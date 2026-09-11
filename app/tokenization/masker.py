"""
Mask -> stable token -> unmask pipeline.

Masking:
  For each detected PII entity, generate a stable synthetic token of the
  form `<<GLP_<ENTITY_TYPE>_<n>>>` where `n` is a per-session, per-entity-type
  counter. "Stable" means: the same original value, if it appears multiple
  times in the same session, is replaced with the *same* token every time
  (so the LLM sees consistent references, e.g. a name used twice stays the
  same placeholder instead of becoming two different ones) -- this matters
  for response quality, not just privacy. The token format is deliberately
  unlike anything a real LLM would organically produce (double angle
  brackets, fixed prefix) to make step 2 (reversal) a simple, unambiguous
  substring search rather than a fuzzy match.

Unmasking:
  Any occurrence of a known token in the LLM's response text is replaced
  back with the original value from the mapping store, scoped to the
  session that produced it. If a token's mapping has expired (see store.py
  TTL policy) or is otherwise missing, the token is left as-is in the
  output rather than raising -- silently guessing at PII would be worse
  than surfacing an inert placeholder, and this failure mode is documented
  in the README rather than hidden.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.detection.detector import PiiDetector
from app.tokenization.store import MappingStore

TOKEN_PATTERN = re.compile(r"<<GLP_[A-Z0-9_]+_\d+>>")


def _make_token(entity_type: str, index: int) -> str:
    safe_type = re.sub(r"[^A-Z0-9]", "_", entity_type.upper())
    return f"<<GLP_{safe_type}_{index}>>"


@dataclass
class MaskResult:
    masked_text: str
    # value -> token, so repeated values in later calls within the same mask()
    # invocation map consistently.
    value_to_token: dict[str, str] = field(default_factory=dict)


class Masker:
    def __init__(self, detector: PiiDetector, store: MappingStore, mapping_ttl_seconds: int):
        self.detector = detector
        self.store = store
        self.mapping_ttl_seconds = mapping_ttl_seconds

    def mask_text(self, session_id: str, text: str, counters: dict[str, int] | None = None) -> MaskResult:
        """
        Detect and replace PII in `text` with stable tokens, persisting the
        token->value mapping in the store under `session_id`.

        `counters` (entity_type -> next index) lets callers keep numbering
        stable/contiguous across multiple text fields (e.g. several chat
        messages) processed within one call to mask_request; pass the same
        dict across calls to continue the sequence.
        """
        counters = counters if counters is not None else {}
        entities = self.detector.detect(text)
        if not entities:
            return MaskResult(masked_text=text)

        existing_values = self.store.all_tokens(session_id)
        value_to_existing_token = {v: t for t, v in existing_values.items()}

        value_to_token: dict[str, str] = {}
        pieces: list[str] = []
        cursor = 0

        for entity in entities:
            original = entity.text
            if original in value_to_existing_token:
                token = value_to_existing_token[original]
            elif original in value_to_token:
                token = value_to_token[original]
            else:
                idx = counters.get(entity.entity_type, 0)
                token = _make_token(entity.entity_type, idx)
                counters[entity.entity_type] = idx + 1
                self.store.set(session_id, token, original, self.mapping_ttl_seconds)
                value_to_token[original] = token

            pieces.append(text[cursor : entity.start])
            pieces.append(token)
            cursor = entity.end

        pieces.append(text[cursor:])
        return MaskResult(masked_text="".join(pieces), value_to_token=value_to_token)

    def unmask_text(self, session_id: str, text: str) -> str:
        """
        Replace every recognizable token in `text` with its original value.
        Tokens with no live mapping (expired or unknown) are left untouched.
        """
        if not text or "<<GLP_" not in text:
            return text

        def _replace(match: re.Match) -> str:
            token = match.group(0)
            original = self.store.get(session_id, token)
            return original if original is not None else token

        return TOKEN_PATTERN.sub(_replace, text)
