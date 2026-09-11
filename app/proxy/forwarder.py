"""
Outbound forwarding of the (already-masked) request to the real LLM provider.

Three modes, selected by GLP_OUTBOUND_MODE:
  - "mock":      never makes a network call. Returns a deterministic canned
                 response derived from the masked input, so the full
                 mask -> forward -> unmask pipeline is exercisable and
                 testable with zero external dependencies or API keys. This
                 is the default so the service runs out of the box.
  - "openai":    forwards to OpenAI's Chat Completions API.
  - "anthropic": forwards to Anthropic's Messages API.

Only the already-masked payload is ever sent outbound -- this module never
sees or handles unmasked PII; that boundary is enforced by the caller
(the proxy route), which masks before constructing the forwarder call and
unmasks only after it returns.
"""
from __future__ import annotations

from typing import Any

import httpx

from app.config import settings


class LlmForwarder:
    def __init__(self, mode: str | None = None):
        self.mode = mode or settings.outbound_mode

    async def forward(self, provider: str, masked_payload: dict[str, Any]) -> dict[str, Any]:
        """
        Send `masked_payload` (already PII-masked, shaped like a native
        OpenAI/Anthropic request body) to the given provider and return the
        raw provider response as a dict.

        `provider` is "openai" or "anthropic" and selects both the payload
        shape expected by the caller and, in real modes, the endpoint used.
        """
        if self.mode == "mock":
            return self._mock_response(provider, masked_payload)
        if self.mode == "openai" or provider == "openai":
            return await self._forward_openai(masked_payload)
        if self.mode == "anthropic" or provider == "anthropic":
            return await self._forward_anthropic(masked_payload)
        raise ValueError(f"Unsupported outbound mode/provider combination: {self.mode}/{provider}")

    def _mock_response(self, provider: str, masked_payload: dict[str, Any]) -> dict[str, Any]:
        """
        Deterministic stub used when no real API key/network call is
        configured (default for MVP / CI). Echoes back a fixed
        acknowledgement that references the masked input structure only --
        it never contains real PII since it only ever sees the masked
        payload.
        """
        if provider == "anthropic":
            last_user_text = _last_anthropic_user_text(masked_payload)
            return {
                "id": "mock-msg-001",
                "type": "message",
                "role": "assistant",
                "model": masked_payload.get("model", "mock-model"),
                "content": [
                    {
                        "type": "text",
                        "text": f"[mock outbound call] received: {last_user_text}",
                    }
                ],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }

        # default / openai shape
        last_user_text = _last_openai_user_text(masked_payload)
        return {
            "id": "mock-cmpl-001",
            "object": "chat.completion",
            "model": masked_payload.get("model", "mock-model"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"[mock outbound call] received: {last_user_text}",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    async def _forward_openai(self, masked_payload: dict[str, Any]) -> dict[str, Any]:
        if not settings.openai_api_key:
            raise RuntimeError(
                "GLP_OUTBOUND_MODE=openai but GLP_OPENAI_API_KEY is not set. "
                "Set the key, or use GLP_OUTBOUND_MODE=mock for local development."
            )
        async with httpx.AsyncClient(timeout=settings.outbound_timeout_seconds) as client:
            resp = await client.post(
                f"{settings.openai_base_url}/chat/completions",
                headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                json=masked_payload,
            )
            resp.raise_for_status()
            return resp.json()

    async def _forward_anthropic(self, masked_payload: dict[str, Any]) -> dict[str, Any]:
        if not settings.anthropic_api_key:
            raise RuntimeError(
                "GLP_OUTBOUND_MODE=anthropic but GLP_ANTHROPIC_API_KEY is not set. "
                "Set the key, or use GLP_OUTBOUND_MODE=mock for local development."
            )
        async with httpx.AsyncClient(timeout=settings.outbound_timeout_seconds) as client:
            resp = await client.post(
                f"{settings.anthropic_base_url}/messages",
                headers={
                    "x-api-key": settings.anthropic_api_key,
                    "anthropic-version": "2023-06-01",
                },
                json=masked_payload,
            )
            resp.raise_for_status()
            return resp.json()


def _last_openai_user_text(payload: dict[str, Any]) -> str:
    messages = payload.get("messages", [])
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            return content if isinstance(content, str) else str(content)
    return ""


def _last_anthropic_user_text(payload: dict[str, Any]) -> str:
    messages = payload.get("messages", [])
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts = [block.get("text", "") for block in content if isinstance(block, dict)]
                return " ".join(texts)
    return ""
