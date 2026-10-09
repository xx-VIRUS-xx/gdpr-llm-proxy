"""
End-to-end tests for the /v1/proxy/{provider} endpoint using FastAPI's
TestClient. Outbound mode is left at the default "mock" (see app.config),
so these tests never make a real network call.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_healthz():
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["outbound_mode"] == "mock"


def test_proxy_openai_masks_pii_before_mock_forward_and_unmasks_response():
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": "My name is Jane Doe, email jane.doe@example.com."}
        ],
    }
    resp = client.post(
        "/v1/proxy/openai",
        json=payload,
        headers={"X-Customer-Id": "cust-1", "X-Session-Id": "sess-1"},
    )
    assert resp.status_code == 200
    body = resp.json()
    content = body["choices"][0]["message"]["content"]
    # The mock forwarder echoes back the masked text; after unmasking the
    # real PII should be restored and no raw token should remain.
    assert "Jane Doe" in content
    assert "jane.doe@example.com" in content
    assert "<<GLP_" not in content


def test_proxy_anthropic_masks_and_unmasks():
    payload = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Contact John Smith at john.smith@example.com."}],
    }
    resp = client.post(
        "/v1/proxy/anthropic",
        json=payload,
        headers={"X-Customer-Id": "cust-2", "X-Session-Id": "sess-2"},
    )
    assert resp.status_code == 200
    body = resp.json()
    text = body["content"][0]["text"]
    assert "John Smith" in text
    assert "john.smith@example.com" in text
    assert "<<GLP_" not in text


def test_proxy_unsupported_provider_returns_404():
    resp = client.post("/v1/proxy/does-not-exist", json={"messages": []})
    assert resp.status_code == 404


def test_proxy_increments_usage_counter():
    headers = {"X-Customer-Id": "cust-usage-test", "X-Session-Id": "sess-usage"}
    payload = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hello there"}]}

    before = client.get("/v1/usage/cust-usage-test").json()["call_count"]
    client.post("/v1/proxy/openai", json=payload, headers=headers)
    client.post("/v1/proxy/openai", json=payload, headers=headers)
    after = client.get("/v1/usage/cust-usage-test").json()["call_count"]

    assert after == before + 2


def test_proxy_defaults_customer_id_when_header_missing():
    payload = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}
    resp = client.post("/v1/proxy/openai", json=payload)
    assert resp.status_code == 200
    usage = client.get("/v1/usage/anonymous").json()
    assert usage["call_count"] >= 1


def test_no_pii_leaves_message_effectively_unchanged():
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
    }
    resp = client.post(
        "/v1/proxy/openai",
        json=payload,
        headers={"X-Customer-Id": "cust-3", "X-Session-Id": "sess-3"},
    )
    assert resp.status_code == 200
    content = resp.json()["choices"][0]["message"]["content"]
    assert "What is the capital of France?" in content


def test_multi_turn_session_keeps_two_people_distinct():
    headers = {"X-Customer-Id": "cust-mt", "X-Session-Id": "sess-multi-turn"}
    first = client.post(
        "/v1/proxy/openai",
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "My name is Jane Doe."}]},
        headers=headers,
    )
    assert first.status_code == 200

    # Second turn of the same session mentions the earlier person and a new one.
    second = client.post(
        "/v1/proxy/openai",
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Jane Doe and John Smith met today."}]},
        headers=headers,
    )
    content = second.json()["choices"][0]["message"]["content"]
    # The mock forwarder echoes the masked text. If both people shared one
    # token, unmasking would turn both into the same name.
    assert "Jane Doe and John Smith" in content
    assert "<<GLP_" not in content
