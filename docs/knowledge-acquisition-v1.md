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

## Live Publication Gate

Live publication now requires all of these host-owned controls in addition to
`publish=True`:

- `StrictArtifactValidator` accepts only bounded strict UTF-8 `text/plain`. HTML, PDF,
  archives, executable formats, malformed UTF-8, NUL-bearing text, and expanded output
  above the configured bound fail closed because no sandboxed parser exists for them.
- Mandatory versioned scanners run over raw or parsed bytes for the EICAR malware test
  signature, common PII, common secret formats, and explicit prompt-injection patterns.
  A finding or scanner exception terminates the job as `security_rejected` before
  embedding or ingestion. These deterministic rules are a narrow gate, not a claim of
  comprehensive commercial malware or DLP coverage.
- `DurablePublicationAuthorizer(enabled=True)` creates a content-addressed immutable
  authorization bound to tenant, source, policy, raw and parsed digests, parser and
  scanner versions, chunk IDs and digests, embedding profile, corpus generation, and
  ingest idempotency key.
- `PublicationReceiptStore` writes authorization and ingest receipts as canonical,
  bounded, owner-only files using same-directory temporary files, file `fsync`,
  create-if-absent links, and parent-directory `fsync`. Recovery validates exact
  schemas, duplicate JSON keys, path binding, typed fields, and content-derived IDs.
- `HyphaeShadowIngestor` refuses live construction without a durable store, refuses
  publication without a matching stored authorization, rejects mixed-source batches
  and idempotency conflicts, and binds replay to an operator-supplied immutable backend
  identity, collection, and vector target. It accepts only the exact `search_ingested`
  receipt contract with complete snapshot, document-count, idempotency, strict commit,
  WAL digest, and durability-cohort evidence. The canonical backend receipt and its
  digest are persisted before the job becomes `ready`.

`publish=False` remains the default and requires no publication authority. Production
automation remains disabled until deployment supplies an approved scanner backend,
atomic corpus-generation cutover and rollback, deletion propagation, external audit
anchoring, and rate/cost controls. Redirects, compressed content, archive parsing,
HTML, and PDF remain intentionally unsupported rather than silently under-scanned.

## Durable Acquisition Store

`SQLiteTenantStore` now supplies the durable queue boundary for one tenant per database.
It enables WAL mode, `synchronous=FULL`, foreign keys, a bounded busy timeout, startup
integrity checks, and a durable tenant/schema binding. Enqueue deduplication and rolling
24-hour/active quotas execute in one `BEGIN IMMEDIATE` transaction, so concurrent
workers cannot exceed policy between a count and insert.

Workers claim jobs through expiring leases with monotonically increasing fencing
tokens. Renewal, state mutation, exact outbox staging, receipt recording, and release
all require the current unexpired owner and fence. Expired pre-outbox work can be
requeued explicitly; stale workers cannot mutate local state afterward. Ingesting and
verifying jobs retain their status because they have deterministic durable recovery
material.

`DurableAcquisitionWorker` serializes a versioned `PreparedIngest` before any backend
call. The immutable command includes the tenant/job binding, corpus generation,
idempotency key, mode, complete ordered chunk text and coordinates, exact hexadecimal
embedding values, authorization, and publication target. On restart:

- `ingesting` replays that exact command with the same Hyphae idempotency identity;
- `verifying` adopts the durable typed receipt without reacquiring or re-embedding;
- an expired earlier phase is deliberately requeued and deterministically rebuilt.

SQLite receipt/outbox digests and strict decoders fail closed on malformed or changed
state. One database file must remain on a local filesystem with correct WAL shared
locking; copying only the main file while WAL is active is unsupported. Process-crash
tests do not prove storage-device power-loss behavior, which remains a deployment
qualification requirement.

## Durable Finalization And Scheduling

Schema version 2 adds an immutable notification outbox and an explicit migration from
schema version 1. A `DurableFinalizationWorker` separately claims `ready`, `answering`,
or due `notifying` jobs under the same lease/fence authority. Final answers become
versioned `PreparedNotification` commands bound to tenant, job, query digest, corpus
generation, host-configured sink, evidence handles, and a deterministic notification
identity.

The notification command and `answering -> notifying` transition commit atomically.
The external sink must deduplicate `notification_id`; response loss therefore replays
the same event rather than inventing another delivery. A typed `NotificationReceipt`
must match the complete command before `notifying -> completed`, and receipt storage,
completion, and lease release occur in one transaction. Insufficient evidence instead
ends as `insufficient_after_ingest` without creating a notification.

Transient sink failures retain `notifying`, persist a bounded error and capped
exponential backoff, and release the lease for a later due claim. Expired `answering`
and `notifying` leases are reclaimable with a higher fence. `KnowledgeScheduler.tick()`
automatically requeues bounded expired pre-outbox work and alternates acquisition and
finalization claims to prevent either queue from starving; its run loop sleeps only
when no work was performed. Database leases remain the authority if multiple scheduler
processes overlap.

Finalization callback contracts now receive explicit answer and notification deadlines,
and worker construction requires a lease that covers the largest deadline plus a
safety window. Adapters must enforce those deadlines in their own transport or a
supervised process boundary; Python worker threads are not treated as hard cancellation.
Lease expiry and sink idempotency preserve replay safety but cannot stop an adapter that
ignores its timeout contract.

Typed transient, permanent, and timeout failures drive deterministic SHA-256 equal
jitter, independent retry limits for answering and notifying, and durable schema-v3
dead letters. Unexpected adapter exceptions fail closed as permanent. Answer retries,
notification attempts, due times, bounded errors, and dead-letter evidence survive
restart. `FinalizationQueueSnapshot` exposes authoritative queue/deferred/lease/DLQ
counts and oldest-ready age without adding high-cardinality labels, while scheduler
error history is bounded in memory.

`check_notification_sink` is an in-process adapter smoke check for stable ID, logical
receipt replay, and exact command binding. Provider-side durable deduplication across
adapter/process restart and hard timeout behavior require a separate integration test
against the real provider. Power-loss qualification, jitter/dead-letter alerting
thresholds, and metrics export remain deployment concerns.

## Isolated Hyphae Conformance

Run `scripts/hyphae_knowledge_conformance.py` with the exact Hyphae 2.1.0 Python SDK on
`PYTHONPATH` and an already provisioned, tenant-isolated native endpoint:

```text
PYTHONPATH=/path/to/hyphae/sdks/python/src:src \
  python scripts/hyphae_knowledge_conformance.py \
  --endpoint /tmp/tenant-a/hyphae.sock \
  --api-key-file /tmp/tenant-a/owner.key \
  --collection 13 --receipts /tmp/tenant-a/receipts \
  --routing-database /tmp/tenant-a/routing.sqlite3 \
  --backend-id <sha256-of-persistent-hyphae-directory-lineage> \
  --expected-sdk-version 2.1.0 --expected-runtime-version 2.1.0 \
  --score-scale 0.03278688524590164
```

The collection must declare the `body`, `source_id`, `source_version`,
`content_digest`, `corpus_generation`, `byte_start`, `byte_end`, and `chunk_ordinal`
doc values and an exact two-dimensional vector named `semantic`. The script exercises
the real publication gate, persists strict Hyphae receipt evidence, restarts the
receipt store, executes integrated lexical/vector/filter retrieval, and verifies the
hydrated body digest. The endpoint and receipt directory must both be dedicated to the
same tenant; the script never initializes or mutates another tenant's data directory.
`--backend-id` must be derived from the persistent Hyphae directory lineage, not its
socket path, so recreating a backend cannot inherit local replay authority.
The 2.1.0 certification must additionally retain exact tagged binary and wheel
digests, negotiated protocol identity, capability response, strict receipt, daemon
restart replay, exact `vector_branches` strategy evidence, raw score distribution, and
backend-lineage equality during retrieval. Synthetic and model-only shadow evidence does
not substitute for native-runtime certification.
The certified single-document weighted-RRF maximum is `2 / 61 =
0.03278688524590164`; production `RetrievalConfig` instances targeting the same
one-lexical/one-vector weight-1 fusion must pin that scale. A different branch count,
branch weight, fusion method, or Hyphae release requires a new score calibration gate.
Production generation routing must receive the root-owned
`HYPHAE_210_RETRIEVAL_PROFILE`; `GenerationRoutedRetriever` has no implicit score
profile and validates the exact vector receipt before accepting routed evidence.
The isolated conformance path also verifies the generation manifest from durable live
receipts, activates it, retrieves through the active route under a Hyphae deadline,
produces a deterministic verbatim canary quote, stages one immutable notification, and
reaches `completed`. The quote runtime and process-local sink are canary fixtures, not
production Gemma or provider qualification.
Production finalization can use `SupervisedFrozenGemmaRuntime` to exchange one bounded
canonical request with a digest-pinned executable over stdin/stdout. The host enforces
one wall-clock deadline, process-group termination, frozen identity, exact response
schema, supplied handles, and verbatim quotations. The shipped governed runtime loads
the exact Gemma artifact manifest and published control bundle; the control head may
only select evidence or abstain.

`SQLiteMailboxNotificationSink` provides a separate durable acceptance boundary keyed
by `notification_id`. Replay after process restart returns the original receipt and a
different command under the same identity fails permanently. This proves exactly-once
acceptance into the local mailbox, not exactly-once observation by an external webhook,
email, or messaging provider.

## Production Runtime Boundary

Schema version 4 adds immutable generation manifests, one active routing pointer,
compare-and-swap pause/activate/resume/rollback receipts, unique backend collections,
candidate inventory enforcement, generation-routed retrieval, and claim filtering.
Candidate publication is allowed only into a registered building generation and must
produce durable receipts covering the complete manifest before verification. Cutover
requires paused admission plus drained predecessor/candidate acquisition and predecessor
finalization; activation remains paused until explicit resume. Rollback targets only the
immediate predecessor.

Production adapters now include the ClamAV INSTREAM protocol, strict DLP response
binding, fixed HTTPS notification acknowledgement, append-only audit chaining,
Prometheus text rendering, a bounded process supervisor, and a host-owned frozen-model
orchestrator that accepts only verbatim quotations from supplied evidence handles.
`celiums-knowledge worker-once` loads a root-configured `module:factory` tenant adapter;
the model cannot select that adapter, tenant, source, scanner, Hyphae target, generation,
or notification destination.

These components remain disabled without deployment configuration and real-service
qualification. A real rollout must supply separate tenant Hyphae processes and
collections, ClamAV definitions, authenticated DLP and notification providers,
provider-side idempotency, externally anchored audit, OS sandboxing, pinned Gemma
weights/runtime, monitored metrics, backups, canary cutover and tested rollback.
