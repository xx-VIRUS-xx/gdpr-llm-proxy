"""
Tests for the mask -> stable token -> unmask pipeline (the credibility-
critical core of the product).
"""
from __future__ import annotations

from cryptography.fernet import Fernet

from app.detection.detector import PiiDetector
from app.tokenization.masker import Masker, TOKEN_PATTERN
from app.tokenization.store import InMemoryMappingStore


def make_masker(detector: PiiDetector, ttl_seconds: int = 60) -> Masker:
    store = InMemoryMappingStore(encryption_key=Fernet.generate_key().decode())
    return Masker(detector=detector, store=store, mapping_ttl_seconds=ttl_seconds)


def test_mask_replaces_email_with_token(detector):
    masker = make_masker(detector)
    result = masker.mask_text("session-1", "Contact me at jane.doe@example.com please.")
    assert "jane.doe@example.com" not in result.masked_text
    assert TOKEN_PATTERN.search(result.masked_text)


def test_mask_then_unmask_roundtrips_exactly(detector):
    masker = make_masker(detector)
    original = "Contact Jane Doe at jane.doe@example.com or +1 415-555-0132."
    masked = masker.mask_text("session-1", original).masked_text
    assert original.split()[1] not in masked or "Jane" not in masked  # PII no longer present verbatim (loose check below is stronger)
    unmasked = masker.unmask_text("session-1", masked)
    assert unmasked == original


def test_same_value_gets_stable_token_within_session(detector):
    masker = make_masker(detector)
    text = "Jane Doe called. Later, Jane Doe called again."
    result = masker.mask_text("session-1", text)
    tokens_found = TOKEN_PATTERN.findall(result.masked_text)
    person_tokens = [t for t in tokens_found if "PERSON" in t]
    # Same value "Jane Doe" appearing twice should reuse one token, not mint two.
    assert len(set(person_tokens)) == 1


def test_stable_token_persists_across_separate_mask_calls_same_session(detector):
    masker = make_masker(detector)
    first = masker.mask_text("session-1", "My name is Jane Doe.")
    second = masker.mask_text("session-1", "Jane Doe again.")
    first_tokens = set(TOKEN_PATTERN.findall(first.masked_text))
    second_tokens = set(TOKEN_PATTERN.findall(second.masked_text))
    assert first_tokens & second_tokens, "Expected token reuse for the same value across calls in one session"


def test_different_sessions_do_not_share_tokens_or_leak_values(detector):
    masker = make_masker(detector)
    masker.mask_text("session-1", "Jane Doe lives here.")
    # unmasking in a different, unrelated session must not resolve session-1's tokens
    leaked_text = "<<GLP_PERSON_0>>"
    result = masker.unmask_text("session-2", leaked_text)
    assert result == leaked_text  # left as an inert token, not resolved to "Jane Doe"


def test_unmask_leaves_unknown_token_untouched(detector):
    masker = make_masker(detector)
    text = "Here is a token that was never issued: <<GLP_PERSON_99>>."
    result = masker.unmask_text("session-1", text)
    assert result == text


def test_unmask_after_ttl_expiry_leaves_token_in_place(detector):
    masker = make_masker(detector, ttl_seconds=0)
    original = "Contact Jane Doe for details."
    masked = masker.mask_text("session-1", original).masked_text
    import time

    time.sleep(0.05)
    unmasked = masker.unmask_text("session-1", masked)
    # Mapping expired -- token remains, original PII is NOT reconstructed.
    assert "Jane Doe" not in unmasked
    assert TOKEN_PATTERN.search(unmasked)


def test_mask_text_with_no_pii_is_unchanged(detector):
    masker = make_masker(detector)
    text = "The weather today is sunny with a high of 20 degrees."
    result = masker.mask_text("session-1", text)
    assert result.masked_text == text


def test_response_echoing_token_gets_unmasked(detector):
    """
    Simulates the core product flow: prompt is masked, a (mock) LLM response
    echoes the synthetic token back, and unmasking restores the real value
    before returning to the caller.
    """
    masker = make_masker(detector)
    prompt = "Summarize: Jane Doe's email is jane.doe@example.com."
    masked_prompt = masker.mask_text("session-1", prompt).masked_text

    # Simulate an LLM response that echoes back the masked tokens verbatim.
    fake_llm_response = masked_prompt.replace("Summarize: ", "Summary: ")

    final_response = masker.unmask_text("session-1", fake_llm_response)
    assert "Jane Doe" in final_response
    assert "jane.doe@example.com" in final_response
    assert "<<GLP_" not in final_response


# --- multi-turn sessions: new values must never reuse an earlier token --------

import re as _re

from app.detection.detector import PiiEntity


class StubDetector:
    """Deterministic detector so these tests do not depend on an NLP model."""

    NAMES = ("Alice", "Bob", "Carol")

    def detect(self, text):
        found = []
        for name in self.NAMES:
            for m in _re.finditer(name, text):
                found.append(PiiEntity("PERSON", m.start(), m.end(), 0.9, name))
        for m in _re.finditer(r"[\w.]+@[\w.]+\.\w+", text):
            found.append(PiiEntity("EMAIL_ADDRESS", m.start(), m.end(), 0.9, m.group(0)))
        return sorted(found, key=lambda e: e.start)


def test_new_value_in_a_later_request_does_not_overwrite_an_earlier_token():
    masker = make_masker(StubDetector())
    # Each request starts with fresh counters, as in app.main._mask_payload.
    turn1 = masker.mask_text("s", "hello Alice", {}).masked_text
    turn2 = masker.mask_text("s", "hello Bob", {}).masked_text

    assert TOKEN_PATTERN.findall(turn1) != TOKEN_PATTERN.findall(turn2)
    assert masker.unmask_text("s", turn1) == "hello Alice"
    assert masker.unmask_text("s", turn2) == "hello Bob"


def test_resent_history_with_a_new_person_gets_distinct_tokens():
    masker = make_masker(StubDetector())
    masker.mask_text("s", "hello Alice", {})
    masked = masker.mask_text("s", "Alice said hi to Bob", {}).masked_text

    tokens = TOKEN_PATTERN.findall(masked)
    assert len(tokens) == 2 and len(set(tokens)) == 2
    assert masker.unmask_text("s", masked) == "Alice said hi to Bob"


def test_numbering_is_tracked_per_entity_type_across_requests():
    masker = make_masker(StubDetector())
    masker.mask_text("s", "Alice at alice@example.com", {})
    masked = masker.mask_text("s", "Bob at bob@example.com", {}).masked_text

    assert "<<GLP_PERSON_1>>" in masked
    assert "<<GLP_EMAIL_ADDRESS_1>>" in masked


def test_caller_counters_that_are_higher_still_win():
    masker = make_masker(StubDetector())
    masker.mask_text("s", "hello Alice", {})
    masked = masker.mask_text("s", "hello Bob", {"PERSON": 5}).masked_text
    assert "<<GLP_PERSON_5>>" in masked


def test_sessions_number_independently():
    masker = make_masker(StubDetector())
    masker.mask_text("a", "hello Alice", {})
    masked = masker.mask_text("b", "hello Bob", {}).masked_text
    assert "<<GLP_PERSON_0>>" in masked
