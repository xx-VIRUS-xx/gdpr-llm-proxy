"""
Startup safety checks for production-facing configuration.

These guard the two ways this service could silently put real PII at risk
in a shared/deployed environment: the source-controlled dev encryption key
being left active, and an unencrypted transport to Redis. Both checks are
scoped to only fire when a Redis-backed store is actually configured --
the in-memory/dev path is deliberately zero-friction and must stay that
way (see test_dev_defaults_are_unaffected).
"""
from __future__ import annotations

import pytest

from app.config import Settings, InsecureConfigurationError

REAL_KEY = "fBnZ9y7x3v1u8t6r4q2p0o9m8l7k6j5h4g3f2d1s0a9="


def test_rejects_default_encryption_key_with_redis_store():
    with pytest.raises(InsecureConfigurationError, match="MAPPING_ENCRYPTION_KEY"):
        Settings(store_backend="redis", redis_url="rediss://localhost:6380/0")


def test_rejects_plain_redis_transport_with_real_key():
    with pytest.raises(InsecureConfigurationError, match="TLS"):
        Settings(
            store_backend="redis",
            redis_url="redis://localhost:6379/0",
            mapping_encryption_key=REAL_KEY,
        )


def test_accepts_tls_redis_transport_with_real_key():
    settings = Settings(
        store_backend="redis",
        redis_url="rediss://localhost:6380/0",
        mapping_encryption_key=REAL_KEY,
    )
    assert settings.store_backend == "redis"


def test_accepts_explicit_insecure_transport_opt_out():
    settings = Settings(
        store_backend="redis",
        redis_url="redis://localhost:6379/0",
        mapping_encryption_key=REAL_KEY,
        allow_insecure_redis_transport=True,
    )
    assert settings.allow_insecure_redis_transport is True


def test_dev_defaults_are_unaffected():
    """The zero-config in-memory/dev path must never require any of this."""
    settings = Settings(store_backend="memory", usage_backend="memory")
    assert settings.store_backend == "memory"


def test_redis_usage_backend_alone_also_triggers_checks():
    """usage_backend=redis reuses redis_url too, so it must be covered even
    if store_backend stays on memory."""
    with pytest.raises(InsecureConfigurationError):
        Settings(store_backend="memory", usage_backend="redis")
