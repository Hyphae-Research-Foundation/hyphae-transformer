# DigitalOcean Execution

Cloud execution is a promotion target, not the default development loop. `doctl` is
available and authenticated in the development environment. No resource is created
until a local mini-pilot and pilot complete successfully.

## Design

- Package and tests run locally first.
- Cloud configuration is passed as an immutable Lab manifest.
- Secrets stay in DigitalOcean environment configuration, never in the repository.
- Each job writes its manifest, logs, metrics, and report to external artifact storage.
- Device-hours and cost are reconciled against the preregistered budget.

GPU Droplets and Gradient AI workloads differ by account and region. Before creating a
resource, inspect current availability and pricing with `doctl` and select the smallest
accelerator that satisfies the promoted run.

The latest account preflight found the smallest listed GPU Droplet to be
`gpu-4000adax1-20gb` (RTX 4000 Ada, 20 GB, 8 vCPU, 32 GB RAM) at $0.76/hour. Size
listing does not prove regional capacity, so query availability immediately before a
promoted run. No infrastructure has been provisioned.

Local three-seed WikiText-2 and enwiki8 pilots are complete, but the split-gate
advantage over shared-gate RMS ReZero remains below the minimum effect. Cloud
promotion is therefore deferred; provisioning a paid GPU would not yet answer the
remaining ablation efficiently.

The dedicated split-versus-shared campaign measured only a +0.153% relative effect
for split gates and was inconclusive against its 1% threshold. This closes the local
promotion decision: no paid cloud campaign should be launched for that hypothesis.

## Preflight

```bash
doctl auth list
doctl compute region list
doctl compute size list
doctl compute image list-distribution
```

The first cloud campaign should run the same manifest on a 30M parameter model with
three seeds before larger 60M and 150M rungs are approved.

## Container Execution

Build the pinned runtime and execute exactly one externally supplied manifest:

```bash
docker build -t celiums-rezero:local .
docker run --rm \
  -v "$PWD/manifest.json:/input/manifest.json:ro" \
  -v "$PWD/data:/data:ro" \
  -v "$PWD/runs:/output" \
  celiums-rezero:local run-manifest /input/manifest.json \
  --registry /output --data-root /data
```

For a DigitalOcean worker, mount the prepared corpus root read-only and bind it with
`--data-root`; manifests contain only portable relative locators and content hashes.
Persist the registry directory to Spaces or another durable volume.
The worker must enforce its own hard process timeout in addition to the in-loop
manifest wall-time budget. Provisioning remains an explicit promotion action.
