"""
FastAPI application exposing the GDPR LLM proxy.

Endpoints:
  GET  /healthz              -- minimal status/liveness endpoint
  GET  /v1/usage/{customer_id} -- current call count for a customer
  POST /v1/proxy/{provider}  -- mask -> forward -> unmask an LLM request

`provider` is "openai" or "anthropic". The request body is expected to be
shaped exactly like the corresponding provider's native request (an OpenAI
chat.completions body, or an Anthropic messages body) -- this proxy is a
drop-in: point your existing SDK's base_url at this service (or send it the
same JSON you'd send upstream) and it does the masking transparently.

Customer/session identification:
  - `X-Customer-Id` header identifies the paying customer for usage tracking.
    Required for /v1/proxy calls (defaults to "anonymous" if omitted, purely
    so the MVP is easy to try locally -- a real deployment should enforce
    this via API-key auth, which is out of scope for the MVP).
  - `X-Session-Id` header (optional) scopes the PII token mapping. If
    omitted, a new session id is generated per-request, meaning tokens are
    not stable across multiple calls (e.g. a multi-turn conversation). Pass
    a stable session id across turns of the same conversation to get
    consistent token reuse and correct unmasking of PII that only appears
    in later provider responses referencing earlier turns.
"""
from __future__ import annotations

import uuid
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from app.config import settings
from app.detection.detector import PiiDetector
from app.proxy.forwarder import LlmForwarder
from app.tokenization.masker import Masker
from app.tokenization.store import build_default_store
from app.usage.tracker import build_default_tracker

app = FastAPI(
    title="gdpr-llm-proxy",
    description=(
        "Middleware that masks EU personal data before it reaches a US-hosted "
        "LLM API, and reverses the masking on the response. See README for "
        "what this does and does not guarantee."
    ),
    version="0.1.0",
)

_detector = PiiDetector()
_store = build_default_store()
_masker = Masker(detector=_detector, store=_store, mapping_ttl_seconds=settings.mapping_ttl_seconds)
_forwarder = LlmForwarder()
_usage_tracker = build_default_tracker()

SUPPORTED_PROVIDERS = {"openai", "anthropic"}


class ProxyRequest(BaseModel):
    """
    Accepts an arbitrary JSON body shaped like the target provider's native
    request. We intentionally do not pin this to a strict schema (OpenAI and
    Anthropic request shapes differ, and providers evolve their APIs) --
    validation of provider-specific fields is left to the upstream provider.
    """

    model_config = {"extra": "allow"}


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {
        "status": "ok",
        "outbound_mode": settings.outbound_mode,
        "store_backend": settings.store_backend,
        "mapping_ttl_seconds": settings.mapping_ttl_seconds,
    }


@app.get("/v1/usage/{customer_id}")
def get_usage(customer_id: str) -> dict[str, Any]:
    return {"customer_id": customer_id, "call_count": _usage_tracker.get_count(customer_id)}


@app.post("/v1/proxy/{provider}")
async def proxy(
    provider: str,
    body: ProxyRequest,
    x_customer_id: str | None = Header(default=None, alias="X-Customer-Id"),
    x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
) -> dict[str, Any]:
    if provider not in SUPPORTED_PROVIDERS:
        raise HTTPException(
            status_code=404,
            detail=f"Unsupported provider '{provider}'. Supported: {sorted(SUPPORTED_PROVIDERS)}",
        )

    customer_id = x_customer_id or "anonymous"
    session_id = x_session_id or str(uuid.uuid4())

    payload = body.model_dump(exclude_none=True)

    try:
        masked_payload = _mask_payload(provider, payload, session_id)
    except Exception as exc:  # pragma: no cover - defensive, detection failures shouldn't 500 silently
        raise HTTPException(status_code=400, detail=f"Failed to process request for PII masking: {exc}")

    try:
        raw_response = await _forwarder.forward(provider, masked_payload)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    unmasked_response = _unmask_payload(provider, raw_response, session_id)

    _usage_tracker.increment(customer_id)

    return unmasked_response


def _mask_payload(provider: str, payload: dict[str, Any], session_id: str) -> dict[str, Any]:
    """Mask all free-text fields in a provider-shaped request payload."""
    counters: dict[str, int] = {}
    masked = dict(payload)

    if provider == "openai":
        messages = masked.get("messages", [])
        masked["messages"] = [_mask_openai_message(m, session_id, counters) for m in messages]
    elif provider == "anthropic":
        # System prompt (Anthropic top-level "system" field)
        if isinstance(masked.get("system"), str):
            masked["system"] = _masker.mask_text(session_id, masked["system"], counters).masked_text
        messages = masked.get("messages", [])
        masked["messages"] = [_mask_anthropic_message(m, session_id, counters) for m in messages]

    return masked


def _mask_openai_message(message: dict[str, Any], session_id: str, counters: dict[str, int]) -> dict[str, Any]:
    content = message.get("content")
    if isinstance(content, str):
        message = dict(message)
        message["content"] = _masker.mask_text(session_id, content, counters).masked_text
    elif isinstance(content, list):
        message = dict(message)
        message["content"] = [_mask_content_block(block, session_id, counters) for block in content]
    return message


def _mask_anthropic_message(message: dict[str, Any], session_id: str, counters: dict[str, int]) -> dict[str, Any]:
    content = message.get("content")
    if isinstance(content, str):
        message = dict(message)
        message["content"] = _masker.mask_text(session_id, content, counters).masked_text
    elif isinstance(content, list):
        message = dict(message)
        message["content"] = [_mask_content_block(block, session_id, counters) for block in content]
    return message


def _mask_content_block(block: Any, session_id: str, counters: dict[str, int]) -> Any:
    if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
        block = dict(block)
        block["text"] = _masker.mask_text(session_id, block["text"], counters).masked_text
    return block


def _unmask_payload(provider: str, payload: dict[str, Any], session_id: str) -> dict[str, Any]:
    """Unmask all free-text fields in a provider-shaped response payload."""
    unmasked = dict(payload)

    if provider == "openai":
        choices = unmasked.get("choices", [])
        new_choices = []
        for choice in choices:
            choice = dict(choice)
            message = dict(choice.get("message", {}))
            content = message.get("content")
            if isinstance(content, str):
                message["content"] = _masker.unmask_text(session_id, content)
            choice["message"] = message
            new_choices.append(choice)
        unmasked["choices"] = new_choices
    elif provider == "anthropic":
        content = unmasked.get("content")
        if isinstance(content, list):
            new_content = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    block = dict(block)
                    block["text"] = _masker.unmask_text(session_id, block.get("text", ""))
                new_content.append(block)
            unmasked["content"] = new_content

    return unmasked
