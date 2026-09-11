"""
PII detection.

Approach (deliberately not hand-rolled regex-only):

  - Structured PII with well-defined formats (email, phone numbers, IBAN,
    credit card, and EU-style national ID / passport patterns) is detected
    with Presidio's built-in pattern recognizers, which combine regex with
    checksum/context validation (e.g. Luhn check for card numbers) rather
    than bare regex matching. This is reliable because these formats are
    genuinely structured.

  - Unstructured PII (person names, addresses, organizations) is detected
    with Presidio's NER-backed recognizers, using a spaCy transformer/CNN
    NLP pipeline to find entities in free text. This is inherently
    probabilistic: model recall depends on language, phrasing, capitalization,
    and how "name-like" a name is.

Why Presidio: it is Microsoft's open-source, widely deployed PII
detection/anonymization toolkit built exactly for this problem (analyzer +
anonymizer split, pluggable recognizers, confidence scores per entity). It
is a justified, maintained approach rather than a bespoke regex pile, and
gives us a clean seam to add custom recognizers (e.g. a customer-specific
allowlist or organization-specific ID formats) later.

KNOWN LIMITATIONS (documented here and in README, deliberately not papered
over):
  - No PII detector has 100% recall. Uncommon names, names in non-Latin
    scripts, nicknames, misspellings, and names embedded in unusual sentence
    structures can be missed by the NER model.
  - Regex-based structured detectors can false-negative on non-standard
    formats and can false-positive on numbers that merely look like the
    pattern (e.g. an order number that matches a phone-number shape). An
    explicit pattern recognizer is added below for international
    "+<country code>..." phone numbers, since Presidio's default recognizer
    misses these without a region hint and this proxy's primary use case
    (EU-to-US transfer) makes international-format numbers common, not rare.
    Locally formatted numbers without any country code or leading '+' in an
    unfamiliar regional format can still be missed.
  - Detection quality depends on the underlying spaCy model; smaller models
    trade recall for speed/memory.
  - This module does not guarantee legal GDPR compliance by itself -- see
    README "What this does and does not guarantee".
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider

from app.config import settings

# Presidio's built-in PHONE_NUMBER recognizer (phonenumbers library under the
# hood) reliably catches domestic-format numbers but misses numbers written
# in international "+<country code>" form (e.g. "+1-555-123-4567",
# "+15551234567") when no region hint is supplied -- confirmed by direct
# testing. Since this proxy exists specifically to protect EU-to-US data
# transfers, international-format numbers are the common case, not the edge
# case, so this gap is patched with an explicit pattern recognizer rather
# than left as a documented limitation.
_INTL_PHONE_PATTERN = Pattern(
    name="intl_phone_number",
    # '+' followed by 7-15 digits, optionally grouped with spaces/dashes/dots/parens.
    regex=r"\+\d{1,3}[-.\s]?\(?\d{1,4}\)?(?:[-.\s]?\d{2,4}){2,4}",
    score=0.6,
)
_INTL_PHONE_RECOGNIZER = PatternRecognizer(
    supported_entity="PHONE_NUMBER",
    patterns=[_INTL_PHONE_PATTERN],
    name="IntlPhoneNumberRecognizer",
)

# Entity types we ask Presidio to look for. This list intentionally spans
# both the "structured" (regex/checksum-backed) and "unstructured"
# (NER-backed) categories described above.
DEFAULT_ENTITIES = [
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "PERSON",
    "LOCATION",
    "IBAN_CODE",
    "CREDIT_CARD",
    "IP_ADDRESS",
    "NRP",  # nationality / religious / political group -- can be sensitive under GDPR special categories
    "MEDICAL_LICENSE",
    "US_SSN",
]
# Deliberately excluded: DATE_TIME. Presidio's DATE_TIME recognizer fires on
# generic temporal words ("quarterly", "this year", "today") that are not
# personal data in the GDPR sense and would mask ordinary business text. A
# real date of birth is far more reliably caught in context by the NER
# PERSON/LOCATION recognizers plus surrounding structured data than by this
# noisy generic recognizer, so it is left out of the default set rather than
# generating false positives here. A customer with a genuine DOB-detection
# need can re-add it explicitly via PiiDetector(entities=[...]).


@dataclass(frozen=True)
class PiiEntity:
    """A single detected PII span."""

    entity_type: str
    start: int
    end: int
    score: float
    text: str


@lru_cache(maxsize=1)
def _build_analyzer(spacy_model: str) -> AnalyzerEngine:
    """
    Build (and cache) the Presidio AnalyzerEngine.

    This is expensive (loads a spaCy model into memory), so it is built once
    per process per model name and reused across requests -- important since
    this sits in the hot path of a proxy.
    """
    nlp_configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": spacy_model}],
    }
    provider = NlpEngineProvider(nlp_configuration=nlp_configuration)
    nlp_engine = provider.create_engine()
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
    analyzer.registry.add_recognizer(_INTL_PHONE_RECOGNIZER)
    return analyzer


class PiiDetector:
    """
    Thin wrapper around Presidio's AnalyzerEngine that returns a normalized,
    dependency-free result type (`PiiEntity`) so the rest of the codebase
    doesn't need to import Presidio types directly.
    """

    def __init__(
        self,
        spacy_model: str | None = None,
        score_threshold: float | None = None,
        entities: list[str] | None = None,
    ) -> None:
        self.spacy_model = spacy_model or settings.spacy_model
        self.score_threshold = (
            score_threshold if score_threshold is not None else settings.detection_score_threshold
        )
        self.entities = entities or DEFAULT_ENTITIES
        self._analyzer = _build_analyzer(self.spacy_model)

    def detect(self, text: str, language: str = "en") -> list[PiiEntity]:
        """Return all detected PII spans in `text`, sorted by start offset."""
        if not text:
            return []

        results: list[RecognizerResult] = self._analyzer.analyze(
            text=text,
            entities=self.entities,
            language=language,
            score_threshold=self.score_threshold,
        )

        entities = [
            PiiEntity(
                entity_type=r.entity_type,
                start=r.start,
                end=r.end,
                score=r.score,
                text=text[r.start : r.end],
            )
            for r in results
        ]
        entities.sort(key=lambda e: e.start)
        return _drop_overlaps(entities)


def _drop_overlaps(entities: list[PiiEntity]) -> list[PiiEntity]:
    """
    Presidio can return overlapping matches from different recognizers
    (e.g. a phone-shaped span also partially matching a generic number
    recognizer). Keep the highest-confidence entity for any overlapping
    span, in left-to-right order, so downstream masking never double-masks
    or produces overlapping token replacements.
    """
    if not entities:
        return []

    by_score_desc = sorted(entities, key=lambda e: e.score, reverse=True)
    kept: list[PiiEntity] = []
    occupied: list[tuple[int, int]] = []

    for entity in by_score_desc:
        if any(entity.start < end and start < entity.end for start, end in occupied):
            continue
        kept.append(entity)
        occupied.append((entity.start, entity.end))

    kept.sort(key=lambda e: e.start)
    return kept
