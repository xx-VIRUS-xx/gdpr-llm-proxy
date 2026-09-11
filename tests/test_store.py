"""
Tests for the encrypted, TTL-based mapping store (in-memory backend).
Redis backend logic is exercised via fakeredis to keep tests hermetic.
"""
from __future__ import annotations

import time

import pytest
from cryptography.fernet import Fernet

from app.tokenization.store import InMemoryMappingStore, RedisMappingStore


@pytest.fixture
def encryption_key() -> str:
    return Fernet.generate_key().decode()


def test_set_and_get_roundtrip(encryption_key):
    store = InMemoryMappingStore(encryption_key=encryption_key)
    store.set("session-1", "<<GLP_PERSON_0>>", "Jane Doe", ttl_seconds=60)
    assert store.get("session-1", "<<GLP_PERSON_0>>") == "Jane Doe"


def test_get_missing_token_returns_none(encryption_key):
    store = InMemoryMappingStore(encryption_key=encryption_key)
    assert store.get("session-1", "<<GLP_PERSON_0>>") is None


def test_values_are_encrypted_at_rest(encryption_key):
    store = InMemoryMappingStore(encryption_key=encryption_key)
    store.set("session-1", "<<GLP_PERSON_0>>", "Jane Doe", ttl_seconds=60)
    raw_entry = store._data["glp:map:session-1:<<GLP_PERSON_0>>"]
    assert b"Jane Doe" not in raw_entry.ciphertext
    assert raw_entry.ciphertext != b"Jane Doe"


def test_ttl_expiry_purges_mapping(encryption_key):
    store = InMemoryMappingStore(encryption_key=encryption_key)
    store.set("session-1", "<<GLP_PERSON_0>>", "Jane Doe", ttl_seconds=0)
    time.sleep(0.05)
    assert store.get("session-1", "<<GLP_PERSON_0>>") is None


def test_sessions_are_isolated(encryption_key):
    store = InMemoryMappingStore(encryption_key=encryption_key)
    store.set("session-1", "<<GLP_PERSON_0>>", "Jane Doe", ttl_seconds=60)
    store.set("session-2", "<<GLP_PERSON_0>>", "John Smith", ttl_seconds=60)
    assert store.get("session-1", "<<GLP_PERSON_0>>") == "Jane Doe"
    assert store.get("session-2", "<<GLP_PERSON_0>>") == "John Smith"


def test_all_tokens_returns_only_live_entries(encryption_key):
    store = InMemoryMappingStore(encryption_key=encryption_key)
    store.set("session-1", "<<GLP_PERSON_0>>", "Jane Doe", ttl_seconds=60)
    store.set("session-1", "<<GLP_EMAIL_ADDRESS_0>>", "jane@example.com", ttl_seconds=0)
    time.sleep(0.05)
    tokens = store.all_tokens("session-1")
    assert tokens == {"<<GLP_PERSON_0>>": "Jane Doe"}


def test_delete_session_removes_all_its_mappings(encryption_key):
    store = InMemoryMappingStore(encryption_key=encryption_key)
    store.set("session-1", "<<GLP_PERSON_0>>", "Jane Doe", ttl_seconds=60)
    store.set("session-1", "<<GLP_PERSON_1>>", "John Smith", ttl_seconds=60)
    store.set("session-2", "<<GLP_PERSON_0>>", "Someone Else", ttl_seconds=60)

    store.delete_session("session-1")

    assert store.get("session-1", "<<GLP_PERSON_0>>") is None
    assert store.get("session-1", "<<GLP_PERSON_1>>") is None
    assert store.get("session-2", "<<GLP_PERSON_0>>") == "Someone Else"


def test_wrong_key_cannot_decrypt():
    key_a = Fernet.generate_key().decode()
    key_b = Fernet.generate_key().decode()
    store_a = InMemoryMappingStore(encryption_key=key_a)
    store_a.set("session-1", "<<GLP_PERSON_0>>", "Jane Doe", ttl_seconds=60)

    # Simulate a different process/key trying to read the same ciphertext.
    store_b = InMemoryMappingStore(encryption_key=key_b)
    store_b._data = store_a._data
    assert store_b.get("session-1", "<<GLP_PERSON_0>>") is None


# --- Redis-backed store, via fakeredis ---

fakeredis = pytest.importorskip("fakeredis")


@pytest.fixture
def fake_redis_client():
    return fakeredis.FakeStrictRedis()


def test_redis_store_roundtrip(fake_redis_client, encryption_key):
    store = RedisMappingStore(redis_client=fake_redis_client, encryption_key=encryption_key)
    store.set("session-1", "<<GLP_PERSON_0>>", "Jane Doe", ttl_seconds=60)
    assert store.get("session-1", "<<GLP_PERSON_0>>") == "Jane Doe"


def test_redis_store_ttl_expiry(fake_redis_client, encryption_key):
    store = RedisMappingStore(redis_client=fake_redis_client, encryption_key=encryption_key)
    store.set("session-1", "<<GLP_PERSON_0>>", "Jane Doe", ttl_seconds=1)
    fake_redis_client.pexpire("glp:map:session-1:<<GLP_PERSON_0>>", 1)
    time.sleep(0.05)
    assert store.get("session-1", "<<GLP_PERSON_0>>") is None


def test_redis_store_values_encrypted_at_rest(fake_redis_client, encryption_key):
    store = RedisMappingStore(redis_client=fake_redis_client, encryption_key=encryption_key)
    store.set("session-1", "<<GLP_PERSON_0>>", "Jane Doe", ttl_seconds=60)
    raw = fake_redis_client.get("glp:map:session-1:<<GLP_PERSON_0>>")
    assert b"Jane Doe" not in raw


def test_redis_store_delete_session(fake_redis_client, encryption_key):
    store = RedisMappingStore(redis_client=fake_redis_client, encryption_key=encryption_key)
    store.set("session-1", "<<GLP_PERSON_0>>", "Jane Doe", ttl_seconds=60)
    store.set("session-1", "<<GLP_PERSON_1>>", "John Smith", ttl_seconds=60)
    store.delete_session("session-1")
    assert store.get("session-1", "<<GLP_PERSON_0>>") is None
    assert store.get("session-1", "<<GLP_PERSON_1>>") is None
    assert store.all_tokens("session-1") == {}
