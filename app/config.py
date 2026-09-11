"""
Centralized configuration for the GDPR LLM proxy.

Everything here is overridable via environment variables (see .env.example).
Defaults are chosen to make local development and testing work with zero
external dependencies (in-memory store, mocked outbound LLM call).
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="GLP_", extra="ignore")

    # --- Token mapping store ---
    # "memory"  -> in-process encrypted dict, good for local dev/tests, lost on restart.
    # "redis"   -> encrypted-at-rest via app-level Fernet encryption before writing to Redis,
    #              with Redis TTL as the expiry mechanism. Use this in any shared/deployed env.
    store_backend: str = "memory"
    redis_url: str = "redis://localhost:6379/0"

    # Master key for encrypting token->PII mappings at rest (Fernet key, 32 url-safe base64 bytes).
    # MUST be overridden in any non-local environment. Generate with:
    #   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    mapping_encryption_key: str = "wJ0k8p8b9m3z1f4qz3f0m7f5m1c3v9m2x8b7v6n5m4k="

    # How long a token->PII mapping is retained before it is unrecoverable.
    # This bounds the "reversal window": if the LLM provider's response (streaming
    # or delayed) arrives after this TTL, the token cannot be unmasked and is
    # returned to the caller as-is (fail-closed on privacy, fail-open on token leakage
    # into the response body — documented tradeoff in README).
    mapping_ttl_seconds: int = 600  # 10 minutes

    # --- PII detection ---
    # spaCy model used by Presidio's NER-based recognizers for names/locations/orgs.
    # en_core_web_lg gives materially better recall on names than the small model;
    # en_core_web_sm is a lighter-weight fallback for constrained environments.
    spacy_model: str = "en_core_web_lg"
    detection_score_threshold: float = 0.4

    # --- Outbound LLM forwarding ---
    # "mock"      -> never leaves the process; returns a canned response. Default so the
    #                service runs end-to-end with no API keys configured.
    # "openai"    -> forwards to api.openai.com
    # "anthropic" -> forwards to api.anthropic.com
    outbound_mode: str = "mock"
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    anthropic_api_key: str | None = None
    anthropic_base_url: str = "https://api.anthropic.com/v1"
    outbound_timeout_seconds: float = 30.0

    # --- Usage tracking ---
    usage_backend: str = "memory"  # "memory" or "redis" (reuses redis_url)


settings = Settings()
