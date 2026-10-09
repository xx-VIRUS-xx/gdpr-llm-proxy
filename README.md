# gdpr-llm-proxy

> System design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

A lightweight middleware/proxy that sits in front of your calls to
US-hosted LLM APIs (OpenAI, Anthropic) and masks EU personal data (PII) in
the outbound request before it leaves the EU, then reverses the masking on
the response. It's sold dev-to-dev, usage-based: you point your existing
LLM calls at this proxy instead of directly at the provider.

This is infrastructure aimed at developers building EU-facing products
that use an LLM, who need a technical control to reduce international
transfer risk under GDPR Article 44. **It is not a law firm and it does
not make your product GDPR-compliant by itself** -- see
["What this does and does not guarantee"](#what-this-does-and-does-not-guarantee)
below, which you should read before relying on this in production.

## How it works

```
Your app                gdpr-llm-proxy                     OpenAI / Anthropic
   |                          |                                    |
   | POST /v1/proxy/openai    |                                    |
   |------------------------->|                                    |
   |                          | 1. detect PII in prompt/messages    |
   |                          | 2. replace each PII span with a     |
   |                          |    stable synthetic token           |
   |                          | 3. store token->PII mapping         |
   |                          |    (encrypted, short TTL)           |
   |                          |------ masked request -------------->|
   |                          |<----- response (may echo tokens) ---|
   |                          | 4. reverse-substitute any tokens    |
   |                          |    found in the response            |
   |<-------------------------|                                    |
   | unmasked response        |                                    |
```

The real PII value never leaves the process boundary of this proxy. Only
synthetic tokens (like `<<GLP_PERSON_0>>`, `<<GLP_EMAIL_ADDRESS_0>>`) are
sent to the US-hosted API.

## Stack, and why

- **Python + FastAPI**: async request handling suited to a service that
  spends most of its time waiting on an outbound HTTP call to the LLM
  provider; FastAPI/Starlette's async model avoids blocking on that I/O.
  Python is also where the best-supported, most mature open-source PII/NER
  tooling lives (see below), so the detection and the request-handling
  layers share a runtime instead of needing a cross-language hop.
- **Presidio (Microsoft's open-source PII detection/anonymization
  toolkit)** for detection, not a hand-rolled regex pile. Presidio's
  `AnalyzerEngine` combines:
  - Pattern recognizers (regex plus checksum/context validation, e.g. Luhn
    check for credit cards) for genuinely structured PII: email addresses,
    phone numbers, IBANs, SSNs/national-ID-like patterns.
  - NER-backed recognizers (via a spaCy pipeline, `en_core_web_lg`) for
    unstructured PII: person names, locations.

  This is a deliberate, justified choice over bare regex: structured
  formats get regex+validation (appropriate, since they *are* structured),
  and names/addresses -- which have no fixed format -- get a real NLP model
  instead of a brittle "capitalized word" heuristic.
- **Redis** (optional, recommended for any real deployment) purely as the
  encrypted, short-TTL token-mapping store. Not used as a general cache or
  queue.
- **No Qdrant.** See ["Why no Qdrant"](#why-no-qdrant) below -- this
  product's value is deterministic detection/masking/reversal, not
  semantic retrieval, and forcing in a vector DB with no real use case
  would be dishonest scope creep.

## PII detection: approach and honest limitations

Detected by default: email addresses, phone numbers, person names,
locations, IBANs, credit card numbers, IP addresses, US SSNs, medical
license numbers, and NRP (nationality/religious/political affiliation --
itself a GDPR "special category" when it appears).

**No PII detector has 100% recall. This proxy cannot guarantee it will
catch every piece of personal data in a prompt.** Concretely, known and
tested-for gaps:

- **Names**: NER-based detection depends on phrasing, capitalization,
  context, and language. Uncommon names, names in non-Latin scripts,
  nicknames, misspellings, or names dropped into terse/unnatural sentence
  fragments (e.g. a name with no surrounding biographical context) can be
  missed. `tests/test_detector.py::test_known_false_negative_risk_documented_by_behavior`
  exercises exactly this case and intentionally does not assert success --
  it documents the failure mode rather than hiding it.
- **Structured formats**: phone numbers and IDs in unusual or
  non-configured regional formats can be missed by the regex/pattern
  recognizers; conversely, numbers that merely *look* like a phone number
  or ID (e.g. an order number) can be false-positively masked. False
  positives are the safer failure direction here (over-masking degrades
  LLM output quality; under-masking leaks PII), but both are possible.
- **Model dependency**: detection quality is a function of the underlying
  spaCy model. `en_core_web_lg` (default) trades memory/speed for better
  recall than `en_core_web_sm`; you can swap models via
  `GLP_SPACY_MODEL` for your latency/accuracy tradeoff.
- **English-only NER by default**: the analyzer is currently configured
  for the `en` language pipeline. Non-English free text will get
  structured-PII detection (email/phone/IBAN patterns are language-agnostic)
  but not reliable name/location NER. Multi-language support is a
  reasonable extension but is out of MVP scope.
- `DATE_TIME` is deliberately **excluded** from the default entity set --
  Presidio's generic date/time recognizer fires on ordinary words like
  "quarterly" or "this year," which are not personal data and would
  needlessly mangle normal business text. See the comment in
  `app/detection/detector.py` for the reasoning; it can be re-enabled per
  deployment if you have a genuine date-of-birth detection need.

Bottom line: treat this as **risk reduction, not a leak-proof filter**.
See the detection module's docstring (`app/detection/detector.py`) and its
test suite (`tests/test_detector.py`) for the exact behavior being relied
on.

## Tokenization and the mapping store

Detected PII is replaced with a stable synthetic token of the form
`<<GLP_<ENTITY_TYPE>_<n>>>`. "Stable" means the same value, if repeated
within a session, always gets the same token -- this both keeps the
masked prompt coherent for the LLM (a name used twice doesn't become two
different placeholders) and lets response-side unmasking work as a simple
literal token search, not fuzzy matching.

**The token->original-value mapping is the most sensitive state in this
system** (`app/tokenization/store.py`), and it is handled accordingly:

- **Encrypted at rest.** Every value is encrypted with Fernet
  (AES-128-CBC + HMAC) under a server-side master key
  (`GLP_MAPPING_ENCRYPTION_KEY`) before being written to either backend --
  the in-memory dev store and the Redis store both only ever hold
  ciphertext.
- **Short TTL, not indefinite retention.** Mappings expire after
  `GLP_MAPPING_TTL_SECONDS` (default: **600 seconds / 10 minutes**), chosen
  to comfortably outlast one request/response round trip (including
  realistic LLM latency) without becoming a long-lived store of personal
  data. This bounds total store size to roughly `(calls per TTL window)`,
  not `(calls ever made)`.
- **Cleanup mechanics, explicitly**:
  - *Redis backend* (recommended for any real deployment): TTL is enforced
    natively via `SETEX` -- Redis itself removes expired keys. No separate
    reaper process is required or run.
  - *In-memory backend* (local dev / tests only -- **not** for production;
    it's per-process and doesn't survive a restart or work across multiple
    workers): expiry is swept lazily on each read/write.
- **Session-scoped.** All tokens are namespaced by a session id, so two
  concurrent customers/conversations can never collide on the same token
  or read each other's mappings.
- **Fail-safe on expiry, not fail-silent.** If a token's mapping has
  expired by the time the response arrives, unmasking leaves the token
  in the output as an inert placeholder instead of guessing -- this
  trades a slightly broken-looking response for never fabricating or
  mis-attributing PII. This is a deliberate, documented tradeoff (see
  `app/tokenization/masker.py`), and callers should treat a raw
  `<<GLP_...>>` token surviving into a final response as a sign the TTL
  was too short for their latency profile.

## Why no Qdrant

The brief for this product explicitly calls out that Qdrant's role here is
smaller than in a typical RAG app, and asks us to be honest about whether
there's a genuine use. There isn't one in the MVP: this proxy's core value
is deterministic detection, masking, and reversal of PII -- not semantic
search or retrieval. The one plausible future use envisioned in the design
(a per-customer allowlist of terms that look like PII but aren't, or
organization-specific ID pattern matching) is not implemented in this MVP
because there is no customer-configuration feature yet for it to serve.
Adding a single-collection Qdrant dependency with nothing genuinely using
it would be scope creep for its own sake, so it is left out. If/when
per-customer allowlists ship, that's the natural point to add it.

## What this does and does not guarantee

**This is a technical control that reduces the risk associated with
international data transfer. It is not a legal guarantee of GDPR
compliance, and using it does not by itself satisfy your Article 44
obligations.**

Specifically:

- It reduces (does not eliminate) the amount of raw EU personal data that
  reaches a US-hosted LLM API, by detecting and substituting common PII
  patterns before the request leaves this service.
- It does **not** guarantee 100% detection recall (see limitations above).
  Some PII may still reach the upstream provider undetected.
- It does **not** provide a legal basis for the transfer, a Data
  Processing Agreement, Standard Contractual Clauses, a Transfer Impact
  Assessment, or any other element of a GDPR-compliant transfer mechanism
  -- those remain your (the deploying developer's/company's) legal
  responsibility.
- It does **not** address every category of GDPR-relevant risk (e.g.
  retention policies at the LLM provider once masked/tokenized text is
  sent, the provider's own data handling practices, or personal data that
  arrives via file uploads/attachments rather than prompt text -- out of
  MVP scope).
- Encryption-at-rest and short TTLs reduce the mapping store's exposure
  window, but this service still necessarily handles real PII in memory at
  request time (to detect it and to reverse the masking on the response)
  -- deploying it does not remove the need for your own data processing
  agreements, access controls, and security review appropriate to your
  risk profile.

Talk to your own legal/compliance counsel about what your specific product
needs; this README will not do that job for you, and no README should.

## API

### `GET /healthz`

Minimal status endpoint. Returns current config summary (outbound mode,
store backend, mapping TTL) -- not a full dashboard, per MVP scope.

### `GET /v1/usage/{customer_id}`

Returns the current call count for a customer (see
[Usage tracking](#usage-tracking) below).

### `POST /v1/proxy/{provider}`

`provider` is `openai` or `anthropic`. Send the exact JSON body you would
send to that provider's native chat/messages endpoint; the proxy masks PII
in it, forwards it, unmasks the response, and returns a response shaped
like that same provider's native response.

Headers:

- `X-Customer-Id` (optional, defaults to `"anonymous"`): identifies the
  paying customer for usage tracking. A real deployment should replace
  this with proper API-key auth (out of MVP scope).
- `X-Session-Id` (optional, a UUID is generated if omitted): scopes token
  stability and the mapping store. **Pass a stable session id across the
  turns of one conversation** so tokens stay consistent and later-turn
  responses that reference earlier-turn PII (e.g. "the phone number I gave
  you") can still be unmasked correctly.

Example:

```bash
curl -X POST http://localhost:8000/v1/proxy/openai \
  -H "Content-Type: application/json" \
  -H "X-Customer-Id: acme-corp" \
  -H "X-Session-Id: conversation-42" \
  -d '{
        "model": "gpt-4o-mini",
        "messages": [
          {"role": "user", "content": "My name is Alice Wong, email alice.wong@example.com."}
        ]
      }'
```

## Usage tracking

`app/usage/tracker.py` implements exactly one thing: a per-customer call
counter, incremented once per successful proxied call. This is
intentionally **not** a billing system -- no invoicing, metering tiers, or
payment integration. It exists so a future billing system has a number to
read. In-memory by default; set `GLP_USAGE_BACKEND=redis` for a shared
counter across workers/restarts.

## Configuration

All configuration is environment-variable driven with a `GLP_` prefix; see
`.env.example` for the full list and defaults, and `app/config.py` for
inline documentation of each setting. Key ones:

| Variable | Default | Purpose |
|---|---|---|
| `GLP_STORE_BACKEND` | `memory` | `memory` (dev) or `redis` (shared/prod) mapping store |
| `GLP_MAPPING_ENCRYPTION_KEY` | (dev placeholder) | Fernet key encrypting mappings at rest -- **must** be changed outside local dev |
| `GLP_REDIS_URL` | `redis://localhost:6379/0` | use `rediss://` (TLS) for any non-local deployment |
| `GLP_ALLOW_INSECURE_REDIS_TRANSPORT` | `false` | explicit opt-out allowing a plain `redis://` URL with a Redis-backed store |
| `GLP_MAPPING_TTL_SECONDS` | `600` | mapping store TTL |
| `GLP_SPACY_MODEL` | `en_core_web_lg` | NER model used by Presidio |
| `GLP_OUTBOUND_MODE` | `mock` | `mock` \| `openai` \| `anthropic` |
| `GLP_OPENAI_API_KEY` / `GLP_ANTHROPIC_API_KEY` | unset | required only if using the corresponding real outbound mode |
| `GLP_USAGE_BACKEND` | `memory` | `memory` or `redis` |

Generate a real encryption key:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

**Startup safety checks:** the app refuses to start (raising
`InsecureConfigurationError`) if `GLP_STORE_BACKEND` or `GLP_USAGE_BACKEND`
is `redis` and either (a) `GLP_MAPPING_ENCRYPTION_KEY` is still the
source-controlled dev default -- that key is public, so every mapping
encrypted with it is trivially decryptable -- or (b) `GLP_REDIS_URL` isn't
`rediss://` and `GLP_ALLOW_INSECURE_REDIS_TRANSPORT` isn't explicitly set.
Both checks are skipped entirely on the in-memory dev path, which stays
zero-config on purpose. See `app/config.py::check_production_safety`.

## Running locally

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m spacy download en_core_web_lg

# Runs fully self-contained: in-memory store, mocked outbound LLM call --
# no Redis, no API keys required.
.venv/bin/uvicorn app.main:app --reload
```

With Redis (closer to a real deployment):

```bash
docker compose up --build
```

## Tests

```bash
.venv/bin/python -m pytest -v
```

42 tests covering:
- `tests/test_detector.py`: structured PII (email, phone, IBAN, SSN-like
  national ID pattern), basic name detection, multi-entity text, overlap
  resolution, and an explicit documented false-negative case.
- `tests/test_masker.py`: mask/unmask round-tripping, stable token reuse
  within and across calls in a session, session isolation, unknown-token
  and TTL-expiry handling, and the full "response echoes token back"
  scenario the whole product depends on.
- `tests/test_store.py`: encryption at rest (including "wrong key can't
  decrypt"), TTL expiry, session isolation/deletion, for both the
  in-memory and Redis (via `fakeredis`) backends.
- `tests/test_proxy.py`: end-to-end HTTP behavior of `/v1/proxy/{provider}`
  for both OpenAI- and Anthropic-shaped payloads, usage counting, and
  error handling for unsupported providers.
- `tests/test_usage_tracker.py`: counter correctness and per-customer
  isolation.

## Explicitly out of scope (MVP)

Multi-region deployment, support for every LLM provider (only OpenAI and
Anthropic request/response shapes are handled), a billing/payment system,
and any frontend/dashboard beyond the `/healthz` status endpoint.
