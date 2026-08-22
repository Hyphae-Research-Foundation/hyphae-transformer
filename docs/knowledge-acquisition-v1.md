# Knowledge Acquisition V1

## Objective

The first product slice is deliberately narrower than TinyMARS channels or learned
Hyphae graph navigation. A frozen language model retrieves tenant-local evidence and
answers only when an external sufficiency policy considers that evidence adequate.
When knowledge is absent, the coordinator durably enqueues an asynchronous,
policy-gated acquisition job before returning:

```text
No poseo este conocimiento, descargando...
```

The model never receives network, filesystem, tenant-routing, credential, or direct
Hyphae write authority.

## Boundaries

- One tenant maps to one isolated Hyphae process and data directory.
- Tenant identity comes from authentication, never model output.
- Retrieval and sufficiency decisions are external and deterministic.
- Gemma may later learn search/open/follow/answer/no-knowledge actions, but policy and
  authorization remain host-owned.
- Automatic acquisition resolves only registered source IDs. Raw model-provided URLs
  are not accepted.
- Downloaded documents are untrusted data and never instructions.
- Unknown licenses, conflicts, blocked authority, malware, or policy failures fail
  closed.
- Knowledge enters Hyphae through idempotent, bounded ingestion. It is not trained
  into frozen model weights.

## Response Contract

Knowledge-ready responses contain evidence handles that must be validated before
generation. Missing knowledge returns a separate job identifier and deduplication
flag; variable job data is not interpolated into the fixed user-facing message.

```json
{
  "answer": "No poseo este conocimiento, descargando...",
  "status": "knowledge_pending",
  "job_id": "job_0123456789abcdef",
  "deduplicated": false
}
```

## Job Lifecycle

```text
queued -> acquiring -> quarantined -> validating -> chunking -> embedding
       -> ingesting -> verifying -> ready -> answering -> notifying -> completed
```

Terminal failures include policy denial, unknown license, security rejection,
ingestion failure, cancellation, and insufficient evidence after ingestion. A job
that remains insufficient does not recursively trigger another acquisition.

## Phase 0 Scope

The current implementation provides typed, content-addressed contracts, a calibrated
sufficiency gate, deterministic job IDs, tenant-isolated in-memory state, allowlisted
source URL validation, bounded quotas, and an explicit lifecycle. It is a simulator
and contract gate, not a production downloader or Hyphae writer.

## Phase 1 Integration

Phase 1 now includes a read-only `HyphaeRetrievalGateway` over the smallest public
Python SDK subset: `search_collection`. It is bound to one tenant, one collection, one
corpus generation, and an optional pinned caller-owned embedding provider. Requests
combine a lexical branch and an exact named-vector query when configured, and insert
an active-generation filter before ranking.

Integrated hits currently return object IDs, scores, and doc values but not the exact
document body. Until Hyphae publishes `SearchDocumentGetMany`, ingestion must place a
bounded body, source ID, source version, content digest, and corpus generation in
declared doc-value fields. The gateway rejects missing fields, body/digest mismatches,
cross-generation hits, malformed snapshots, unexpected response kinds, non-finite
vectors, and cross-tenant calls.

The complete Hyphae snapshot identity is reduced to a stable SHA-256 fingerprint and
stored with every evidence bundle. This binds evidence to the returned retrieval root
but does not create a long-lived read view or prove the result. Public read views,
native abstention results, document-body reads, ingest provenance, snapshot pins, and
atomic corpus-generation publication remain follow-up engine work.

The sufficiency gate remains authoritative outside Gemma. A supported evidence bundle
returns opaque handles for generation; below-threshold or absent evidence creates one
deduplicated asynchronous acquisition job. Conflicts, blocked authority, malformed
Hyphae results, and approximate retrieval under an exact-only policy fail closed.

## Security Gate Before Network Acquisition

No automatic source connector may ship until tests prove HTTPS-only egress,
DNS-rebinding and metadata-service denial, redirect revalidation, compressed and
expanded byte limits, parser sandboxing, malware/PII/secret/license gates,
cross-tenant denial, idempotent crash recovery, rollback, deletion propagation,
external audit anchoring, and strict rate/cost quotas.

## Phase 2 Simulator

Phase 2 executes the complete asynchronous lifecycle without a network fetcher or a
Hyphae writer. A tenant-bound `InMemorySourceConnector` returns only artifacts that
were provisioned explicitly by the test host. The worker verifies tenant/source
binding, byte budget, MIME policy, artifact SHA-256, and license policy before
deterministic UTF-8 chunking.

Each chunk is content-addressed by source digest and byte range, embedded by a pinned
provider interface, and written to an idempotent tenant-local simulated index. The
worker verifies every stored chunk before transitioning to `ready`; verification
failure is terminal. Finalization transitions through `answering` and `notifying`, or
ends as `insufficient_after_ingest` without recursively creating another job.

The simulator proves lifecycle ordering, tenant isolation, deterministic overlap,
source/license/MIME/size rejection, idempotent replay, and failure handling. It does
not claim sandboxing, malware scanning, network security, durable queue semantics,
Hyphae commit receipts, embedding attestation, or production notification delivery.
Those are required before replacing the in-memory source and index with live Phase 3
components.

## Phase 3 Shadow Mode

Phase 3 adds a real HTTPS connector and a bounded Hyphae ingestion adapter, but keeps
publication disabled by default. Source URLs remain immutable policy fields; neither
the model nor a user query can provide or redirect them. The connector enforces HTTPS,
declared host/path, public-only DNS answers, pinned-IP connection with hostname TLS,
peer-IP matching, disabled redirects, response byte bounds, exact MIME allowlists,
artifact SHA-256, and source license metadata.

The `HyphaeShadowIngestor` builds canonical `search_ingest` documents containing the
temporary body hydration fields, source/version/digest, corpus generation, byte
ranges, ordinal, and named embedding vector. `publish=False` is the default: batches
are validated, receipted, and verified without calling Hyphae, and the job terminates
as `shadow_validated`. `publish=True` is an explicit integration opt-in and remains
appropriate only for controlled tests until the full live-publication gate closes.

Phase 3 does not yet include DNS pinning through an egress proxy, redirect support,
content decompression, archive or rich-document parsing, malware/PII/secret scanning,
license discovery, durable job leases, transaction outbox, corpus-generation cutover,
rollback, or external receipt signatures. Automatic production publication stays
disabled until those controls and a live tenant-isolated Hyphae conformance run pass.
