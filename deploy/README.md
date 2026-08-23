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
`celiums-knowledge` command. Each root-owned tenant config names an installed
`module:factory` adapter. The factory receives the tenant-bound store and constructs the
approved Hyphae, scanner, model and notification workers; adapter configuration and
credentials remain outside model authority.
