"""
Tests for PII detection covering the MVP-required categories:
  - email
  - phone number
  - national-ID-like pattern (using US_SSN as a stand-in structured ID
    format recognized out of the box by Presidio; IBAN is exercised too as
    an EU-relevant structured identifier)
  - basic name detection (NER-based, explicitly documented as best-effort)

Also covers: no PII text, multiple entities in one string, and overlap
resolution.
"""
from __future__ import annotations

from app.detection.detector import PiiDetector


def _types(entities) -> set[str]:
    return {e.entity_type for e in entities}


def test_detects_email_address(detector: PiiDetector):
    text = "Please contact me at jane.doe@example.com for details."
    entities = detector.detect(text)
    assert "EMAIL_ADDRESS" in _types(entities)
    email_entities = [e for e in entities if e.entity_type == "EMAIL_ADDRESS"]
    assert any(e.text == "jane.doe@example.com" for e in email_entities)


def test_detects_phone_number(detector: PiiDetector):
    text = "You can reach the support line at +1 415-555-0132 anytime."
    entities = detector.detect(text)
    assert "PHONE_NUMBER" in _types(entities)


def test_detects_international_format_phone_numbers(detector: PiiDetector):
    """
    Regression test: Presidio's default PHONE_NUMBER recognizer (without a
    region hint) misses several common international "+<country code>"
    formats -- confirmed by direct testing during a live smoke test of this
    service. A custom pattern recognizer was added specifically to catch
    these, since this proxy's core use case (EU-to-US transfer) makes
    international-format numbers the common case, not the edge case.
    """
    cases = [
        "My number is +1-555-123-4567, call anytime.",
        "Reach me at +15551234567 during business hours.",
        "UK office: +44 20 7946 0958.",
    ]
    for text in cases:
        entities = detector.detect(text)
        assert "PHONE_NUMBER" in _types(entities), f"failed to detect phone number in: {text!r}"


def test_detects_us_ssn_as_structured_id_pattern(detector: PiiDetector):
    text = "His SSN is 523-45-6789 on file."
    entities = detector.detect(text)
    assert "US_SSN" in _types(entities)


def test_detects_iban(detector: PiiDetector):
    text = "Please wire the funds to DE89370400440532013000."
    entities = detector.detect(text)
    assert "IBAN_CODE" in _types(entities)


def test_detects_person_name_basic(detector: PiiDetector):
    text = "My name is John Smith and I live in Berlin."
    entities = detector.detect(text)
    assert "PERSON" in _types(entities)
    names = [e.text for e in entities if e.entity_type == "PERSON"]
    assert any("John" in n or "Smith" in n for n in names)


def test_no_false_positive_on_clean_text(detector: PiiDetector):
    text = "The quarterly report shows a 12% increase in revenue this year."
    entities = detector.detect(text)
    # Clean, PII-free business text should not trigger high-confidence hits.
    assert entities == [] or all(e.score < 0.5 for e in entities)


def test_detects_multiple_entities_in_one_string(detector: PiiDetector):
    text = "Contact Jane Doe at jane.doe@example.com or +1 415-555-0132."
    entities = detector.detect(text)
    types = _types(entities)
    assert "EMAIL_ADDRESS" in types
    assert "PHONE_NUMBER" in types


def test_entities_do_not_overlap(detector: PiiDetector):
    text = "Email jane.doe@example.com or call +1 415-555-0132 about SSN 523-45-6789."
    entities = detector.detect(text)
    sorted_entities = sorted(entities, key=lambda e: e.start)
    for a, b in zip(sorted_entities, sorted_entities[1:]):
        assert a.end <= b.start, f"Overlapping entities: {a} and {b}"


def test_empty_string_returns_no_entities(detector: PiiDetector):
    assert detector.detect("") == []


def test_known_false_negative_risk_documented_by_behavior(detector: PiiDetector):
    """
    This test intentionally documents a known weakness rather than hiding
    it: an uncommon or ambiguous name embedded in a terse, unnatural
    sentence fragment can be missed by NER-based detection. We don't assert
    detection succeeds here -- we assert the system doesn't crash, and note
    in comments (and README) that recall on such inputs is not guaranteed.
    """
    text = "Xk'zyx briefed the team."  # atypical name, no biographical context
    # No assertion on whether PERSON is found -- this is exactly the kind of
    # case the README's "known limitations" section calls out. We only
    # assert detection runs without error on adversarial-ish input.
    entities = detector.detect(text)
    assert isinstance(entities, list)
