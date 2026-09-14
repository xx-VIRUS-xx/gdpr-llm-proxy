"""
Centralized configuration for the GDPR LLM proxy.

Everything here is overridable via environment variables (see .env.example).
Defaults are chosen to make local development and testing work with zero
external dependencies (in-memory store, mocked outbound LLM call).
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import model_validator

# This exact key ships in source control (below) as a zero-config local-dev
# default. If it is still active while talking to a real Redis deployment,
# every PII mapping in that store is trivially decryptable by anyone who has
# read this file -- so its presence is treated as a hard startup error
# outside the in-memory/dev configuration, not just a documented "must
# change" note. See Settings.check_production_safety below.
_DEV_DEFAULT_ENCRYPTION_KEY = "wJ0k8p8b9m3z1f4qz3f0m7f5m1c3v9m2x8b7v6n5m4k="


class InsecureConfigurationError(RuntimeError):
    """Raised when settings would put real PII at risk outside local dev."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="GLP_", extra="ignore")

    # --- Token mapping store ---
    # "memory"  -> in-process encrypted dict, good for local dev/tests, lost on restart.
    # "redis"   -> encrypted-at-rest via app-level Fernet encryption before writing to Redis,
    #              with Redis TTL as the expiry mechanism. Use this in any shared/deployed env.
    store_backend: str = "memory"
    redis_url: str = "redis://localhost:6379/0"
    # Redis connections carry Fernet-encrypted ciphertext, but the TRANSPORT
    # itself (plain "redis://") is unauthenticated and unencrypted by
    # default -- anyone who can reach the port can MITM or read Redis's own
    # AUTH exchange. Any non-local deployment should use "rediss://" (Redis
    # over TLS) or run on a network where that transport is already secured
    # (e.g. a private VPC with its own TLS termination). This flag is an
    # explicit, auditable opt-out for the latter case rather than a silent
    # allowance -- see check_production_safety.
    allow_insecure_redis_transport: bool = False

    # Master key for encrypting token->PII mappings at rest (Fernet key, 32 url-safe base64 bytes).
    # MUST be overridden in any non-local environment. Generate with:
    #   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    mapping_encryption_key: str = _DEV_DEFAULT_ENCRYPTION_KEY

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

    @model_validator(mode="after")
    def check_production_safety(self) -> "Settings":
        """
        Fail closed, at startup, on the two ways this service can silently
        put real PII at risk in a shared/deployed environment. Both checks
        are skipped when everything is still on the in-memory, single-
        process dev backends, since there is no real PII-at-rest exposure
        to protect against in that configuration.
        """
        uses_redis = self.store_backend == "redis" or self.usage_backend == "redis"
        if not uses_redis:
            return self

        if self.mapping_encryption_key == _DEV_DEFAULT_ENCRYPTION_KEY:
            raise InsecureConfigurationError(
                "GLP_MAPPING_ENCRYPTION_KEY is still the source-controlled dev "
                "default while a Redis-backed store is configured. This key is "
                "public (it ships in this repo), so every PII mapping written "
                "with it is trivially decryptable. Generate a real key with: "
                "python -c \"from cryptography.fernet import Fernet; "
                "print(Fernet.generate_key().decode())\" and set it as "
                "GLP_MAPPING_ENCRYPTION_KEY."
            )

        transport_is_tls = self.redis_url.startswith("rediss://")
        if not transport_is_tls and not self.allow_insecure_redis_transport:
            raise InsecureConfigurationError(
                "GLP_REDIS_URL is not using TLS ('rediss://') while a "
                "Redis-backed store is configured. Real PII ciphertext and "
                "Redis's own auth exchange would travel unencrypted. Use a "
                "'rediss://' URL, or if the network path to Redis is already "
                "secured some other way (e.g. a private VPC with its own TLS "
                "termination), set GLP_ALLOW_INSECURE_REDIS_TRANSPORT=true to "
                "acknowledge that explicitly."
            )

        return self


settings = Settings()
