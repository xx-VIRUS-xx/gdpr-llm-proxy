import pytest

from app.detection.detector import PiiDetector


@pytest.fixture(scope="session")
def detector() -> PiiDetector:
    """
    Shared PiiDetector instance across the test session -- building it loads
    a spaCy model, which is the slow part of the test suite, so we pay that
    cost once.
    """
    return PiiDetector()
