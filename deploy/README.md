# Controlled Knowledge Deployment

Production activation remains fail-closed until every tenant has owner-private SQLite,
receipt and audit paths, a dedicated Hyphae process/socket/key, separate candidate and
active collections, real ClamAV/DLP conformance, a notification provider that durably
deduplicates `notification_id`, and a pinned frozen model runtime.

Roll out one tenant at a time. Back up SQLite using its backup API, migrate schema,
bootstrap an operator-reviewed active generation manifest, validate a distinct candidate,
pause claims, drain leases, activate by expected revision, observe metrics/audit, test
rollback, then resume. Never down-migrate SQLite or delete the prior Hyphae collection
during the rollback window.

The provided systemd template runs one bounded attempt through the shipped
`hyphae-knowledge` command. The legacy `celiums-knowledge` alias remains available
during migration. Each root-owned tenant config names an installed
`module:factory` adapter. The factory receives the tenant-bound store and constructs the
approved Hyphae, scanner, model and notification workers; adapter configuration and
credentials remain outside model authority.

Existing installations using `celiums-*` system users and `/var/lib/celiums` must be
migrated tenant by tenant; do not rename ownership or state paths in place while a
worker is active.

For each tenant: stop and disable the legacy timer, wait for or expire its lease, back
up SQLite in WAL-safe mode, install the new package, move configuration and state with
owner-only permissions, change ownership to `hyphae-<tenant>`, run preflight, reload
systemd, enable the Hyphae timer, and verify one bounded attempt before removing the old
unit files. Never run old and new timers concurrently against one tenant database.

The distribution rename from `celiums-rezero` to `hyphae-transformer` requires
uninstalling the old distribution before installing 0.2.0 because both distributions
provide the compatibility import namespace `celiums_rezero`. Legacy CLI entry points
remain in the new distribution, but the two distributions must not be co-installed.
