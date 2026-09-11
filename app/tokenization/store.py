"""
Token -> real-PII mapping store.

This is the most sensitive component in the system: it is the only place
where real PII values are held after the initial request is masked. Two
design decisions follow directly from that:

1. Encryption at rest. Every value written to the store is encrypted with
   Fernet (AES-128-CBC + HMAC, from the `cryptography` package) using a
   server-side master key (`GLP_MAPPING_ENCRYPTION_KEY`). This applies to
   both backends: the in-memory backend (defense in depth against, e.g., a
   process memory dump or an accidental debug log of the store's internals)
   and the Redis backend (defense against anyone with read access to Redis
   itself -- a shared cache, a backup, an operator with `redis-cli` access).

2. Short TTL, not indefinite retention. Mappings are stored per
   request/session with an expiry (`GLP_MAPPING_TTL_SECONDS`, default 10
   minutes) chosen to comfortably outlast a single request/response
   round-trip (including realistic LLM latency) without becoming a
   long-lived store of personal data. This directly bounds the "growing
   liability" problem: the store's total size is O(TTL x requests-per-TTL
   window), not O(all requests ever made).

   Cleanup mechanics:
     - Redis backend: TTL is enforced natively by Redis (`SETEX`), so
       expired keys are removed by Redis itself with no separate reaper
       process to run or maintain.
     - In-memory backend (local dev / tests only -- not for production,
       since it doesn't survive a process restart and isn't shared across
       workers): expiry is enforced lazily on read/write (a `_purge_expired`
       sweep), which is sufficient for a single-process dev server. It is
       not a substitute for Redis in any multi-worker or persistent
       deployment.

   Once a mapping expires, unmasking that token becomes impossible by
   design -- see Masker for the fail-safe behavior this implies.

A session_id namespaces all tokens for one logical request/response
exchange (typically one proxied call), so token collisions across
unrelated customers/requests are impossible even though the token alphabet
itself is short.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from cryptography.fernet import Fernet, InvalidToken

from app.config import settings


class MappingStore(ABC):
    """Interface for a token -> original-value store, scoped by session."""

    @abstractmethod
    def set(self, session_id: str, token: str, value: str, ttl_seconds: int) -> None:
        ...

    @abstractmethod
    def get(self, session_id: str, token: str) -> str | None:
        """Return the original value for `token`, or None if absent/expired."""

    @abstractmethod
    def all_tokens(self, session_id: str) -> dict[str, str]:
        """Return every non-expired token->value pair for a session."""

    @abstractmethod
    def delete_session(self, session_id: str) -> None:
        """Explicitly wipe all mappings for a session (e.g. after a successful unmask)."""


def _fernet(key: str | None = None) -> Fernet:
    return Fernet((key or settings.mapping_encryption_key).encode())


def _namespaced_key(session_id: str, token: str) -> str:
    return f"glp:map:{session_id}:{token}"


@dataclass
class _Entry:
    ciphertext: bytes
    expires_at: float


class InMemoryMappingStore(MappingStore):
    """
    Encrypted, TTL-expiring, in-process store.

    Intended for local development and unit tests. Not shared across
    workers/processes and does not survive a restart -- use RedisMappingStore
    for anything resembling a real deployment.
    """

    def __init__(self, encryption_key: str | None = None) -> None:
        self._fernet = _fernet(encryption_key)
        self._data: dict[str, _Entry] = {}

    def _purge_expired(self) -> None:
        now = time.time()
        expired = [k for k, entry in self._data.items() if entry.expires_at <= now]
        for k in expired:
            del self._data[k]

    def set(self, session_id: str, token: str, value: str, ttl_seconds: int) -> None:
        self._purge_expired()
        ciphertext = self._fernet.encrypt(value.encode())
        key = _namespaced_key(session_id, token)
        self._data[key] = _Entry(ciphertext=ciphertext, expires_at=time.time() + ttl_seconds)

    def get(self, session_id: str, token: str) -> str | None:
        self._purge_expired()
        key = _namespaced_key(session_id, token)
        entry = self._data.get(key)
        if entry is None:
            return None
        try:
            return self._fernet.decrypt(entry.ciphertext).decode()
        except InvalidToken:
            return None

    def all_tokens(self, session_id: str) -> dict[str, str]:
        self._purge_expired()
        prefix = f"glp:map:{session_id}:"
        result = {}
        for key, entry in self._data.items():
            if key.startswith(prefix):
                token = key[len(prefix) :]
                try:
                    result[token] = self._fernet.decrypt(entry.ciphertext).decode()
                except InvalidToken:
                    continue
        return result

    def delete_session(self, session_id: str) -> None:
        prefix = f"glp:map:{session_id}:"
        for key in [k for k in self._data if k.startswith(prefix)]:
            del self._data[key]


class RedisMappingStore(MappingStore):
    """
    Encrypted mapping store backed by Redis, using Redis's native key TTL
    (SETEX) for expiry -- no separate cleanup/reaper job needed. Values are
    Fernet-encrypted before being written, so a Redis instance/backup/operator
    never sees plaintext PII, only ciphertext with a bounded, self-expiring
    lifetime.
    """

    def __init__(self, redis_client=None, encryption_key: str | None = None) -> None:
        self._fernet = _fernet(encryption_key)
        if redis_client is None:
            import redis as redis_lib

            redis_client = redis_lib.from_url(settings.redis_url)
        self._redis = redis_client

    def set(self, session_id: str, token: str, value: str, ttl_seconds: int) -> None:
        ciphertext = self._fernet.encrypt(value.encode())
        key = _namespaced_key(session_id, token)
        self._redis.setex(key, ttl_seconds, ciphertext)
        # Track membership so all_tokens()/delete_session() can enumerate a
        # session's tokens without a KEYS scan. This index key shares the
        # longest reasonable TTL so it doesn't outlive the mappings it indexes.
        index_key = f"glp:session:{session_id}"
        self._redis.sadd(index_key, token)
        self._redis.expire(index_key, ttl_seconds)

    def get(self, session_id: str, token: str) -> str | None:
        key = _namespaced_key(session_id, token)
        ciphertext = self._redis.get(key)
        if ciphertext is None:
            return None
        try:
            return self._fernet.decrypt(ciphertext).decode()
        except InvalidToken:
            return None

    def all_tokens(self, session_id: str) -> dict[str, str]:
        index_key = f"glp:session:{session_id}"
        tokens = self._redis.smembers(index_key)
        result = {}
        for raw_token in tokens:
            token = raw_token.decode() if isinstance(raw_token, bytes) else raw_token
            value = self.get(session_id, token)
            if value is not None:
                result[token] = value
        return result

    def delete_session(self, session_id: str) -> None:
        index_key = f"glp:session:{session_id}"
        tokens = self._redis.smembers(index_key)
        pipe = self._redis.pipeline()
        for raw_token in tokens:
            token = raw_token.decode() if isinstance(raw_token, bytes) else raw_token
            pipe.delete(_namespaced_key(session_id, token))
        pipe.delete(index_key)
        pipe.execute()


def build_default_store() -> MappingStore:
    """Factory selecting the configured backend."""
    if settings.store_backend == "redis":
        return RedisMappingStore()
    return InMemoryMappingStore()
