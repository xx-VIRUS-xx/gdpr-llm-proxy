# Architecture

A stateless-per-request HTTP proxy that masks personal data before an LLM call leaves the EU and reverses the masking on the way back. The only state is a short-lived, encrypted token mapping.

## System context

```
 EU product                gdpr-llm-proxy (FastAPI)                    Provider
 (your backend)   ----->   /v1/proxy/{openai|anthropic}   ----->   OpenAI / Anthropic
                  <-----   unmasked response              <-----   (sees tokens only)
                                  |
                                  +--> Mapping store  (memory | Redis)   encrypted, TTL
                                  +--> Usage tracker  (memory | Redis)   per-customer counter
```

**Trust boundary:** raw PII exists only inside the proxy process and in the encrypted mapping store. Everything past `LlmForwarder` is already masked.

## Components

| Module | Responsibility |
|---|---|
| `app/main.py` | HTTP surface (`/v1/proxy/{provider}`, `/v1/usage/{id}`, `/healthz`); walks provider-shaped payloads and applies mask/unmask to every free-text field (system prompt, string content, `text` blocks) |
| `app/detection/detector.py` | `PiiDetector`: wraps Presidio `AnalyzerEngine` (pattern recognizers + spaCy NER), adds an explicit international `+<cc>` phone recognizer, drops overlapping spans keeping the highest score. Analyzer is built once per process (`lru_cache`) because the spaCy model load is expensive |
| `app/tokenization/masker.py` | Replaces each span with a stable token `<<GLP_<TYPE>_<n>>>`; reuses a token when the same value already exists in the session; reverses tokens in responses with a regex scan |
| `app/tokenization/store.py` | `MappingStore` interface with `InMemoryMappingStore` (dev/tests) and `RedisMappingStore` (shared). Values Fernet-encrypted before storage; expiry via TTL |
| `app/proxy/forwarder.py` | `LlmForwarder`: `mock` (default, no network), `openai`, `anthropic` via `httpx.AsyncClient` |
| `app/usage/tracker.py` | Per-customer call counter (memory or Redis). Not billing |
| `app/config.py` | `GLP_`-prefixed settings and `check_production_safety` startup guard |

## Request flow

1. Resolve `X-Customer-Id` (default `anonymous`) and `X-Session-Id` (default: new UUID).
2. `_mask_payload`: for each text field, `Masker.mask_text` runs the detector, looks up existing tokens for the session, mints new tokens for unseen values, and stores `token -> value` (encrypted, TTL).
3. `LlmForwarder.forward` sends only the masked payload upstream. Provider errors surface as `502`.
4. `_unmask_payload`: every `<<GLP_...>>` in the response text is replaced from the store; tokens with no live mapping are left as inert placeholders.
5. Usage counter increments; the provider-shaped response is returned.

## Key design decisions

- **Presidio instead of hand-rolled regex.** Structured PII gets pattern + validation recognizers; names/locations get NER. One custom recognizer patches a verified gap (international-format phone numbers).
- **Collision-free numbering across turns.** New tokens continue from the highest index already in the session for that entity type, so a later request can never reuse a token and overwrite an earlier mapping (`_next_free_indexes` in `masker.py`).
- **Token format chosen for reversibility.** A fixed, unusual prefix turns unmasking into a literal regex scan, not fuzzy matching.
- **Fail-safe on expiry.** An expired mapping leaves the token in the output rather than guessing a value. Wrong-looking is preferred over mis-attributed PII.
- **Encrypt both backends.** The in-memory store also holds only ciphertext, so a heap dump or debug print of the store does not expose plaintext.
- **TTL instead of retention.** Default 600 s bounds store size to roughly one TTL window of traffic. Redis enforces it natively (`SETEX`); the in-memory store sweeps lazily.
- **Pluggable backends behind interfaces** (`MappingStore`, outbound mode) so the full pipeline runs with zero external services in dev and tests.
- **Fail closed in production.** With a Redis backend the app refuses to start on the public dev encryption key or a non-`rediss://` URL unless explicitly overridden.
- **No vector DB.** Detection and reversal are deterministic; semantic retrieval has no role here.

## Scaling and operational notes

- The proxy process is stateless apart from the store, so it scales horizontally **only with `GLP_STORE_BACKEND=redis`**. The in-memory store is per-process.
- Latency is dominated by the upstream call. The local overhead is the NER pass, which grows with prompt length.
- spaCy model size (`GLP_SPACY_MODEL`) trades recall for memory and speed.
- Detection is English-only for NER by default.

## Known limitations and issues

- No authentication on the proxy: `X-Customer-Id` is trusted as sent.
- `GET /v1/usage/{customer_id}` is unauthenticated.
- Only OpenAI chat-completions and Anthropic messages shapes are handled; streaming and attachments are not.
- Detection is probabilistic and cannot guarantee full recall. See the README for the full statement of what this does and does not guarantee.
