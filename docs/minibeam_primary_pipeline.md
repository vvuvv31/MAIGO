# Minibeam primary GPU pipeline

`minibeam_primary_pipeline` selects scheduling for the existing homogeneous-water
Urban minibeam specialization. It does not select a physics model or change any
step limit, source ZA, material, package, cut, scoring precision or RNG stream.

```yaml
minibeam_primary_specialization: true
minibeam_primary_pipeline: compact
```

Values:

- `combined` (default): source, beamline and water in one kernel.
- `split`: beamline and water in separate kernels, preserving entry order.
- `compact`: separate kernels; group entries with remaining energy above cutoff
  at the front of the handoff buffer and all other entries at its back.

The split modes require the existing specialization eligibility: unified EM,
minibeam enabled, global Urban and unified water, no CT grid, heterogeneous insert
or slab layers. Invalid eligibility is rejected before launching primary work.
The diagnostic environment variable `CARBON_MINIBEAM_PRIMARY_SPLIT=0/1/2`
overrides the YAML selection with combined/split/compact, respectively.

## Boundary and identity

The first stage completes source sampling, copper/air beamline transport,
beamline descendants, phase-space snapshots and water-entry positioning. The
second stage performs the original water transport and its terminal scoring.

The 64-byte `PrimaryWaterEntryState` contains the global history index, per-spot
RNG history and seed, source/reference energies, remaining energy, position and
direction. Stage-specific Urban, ionization and straggling states remain local
to the same physical stage in which they were initialized before the refactor.
No random draw is added by the handoff or partition.

Both live and terminated histories are transferred. Water-stage output arrays
are indexed by the saved global history, never by the reordered lane. This
preserves stopped-history accounting as well as surviving tracks. Phase-space
records are produced in the beamline stage with the original history ID.

## Ordering and storage

The in-order SYCL queue executes count reset, beamline kernel and water kernel
in sequence for each chunk. The buffer is reused only by the following chunk;
the final event is awaited before freeing it or starting downstream processing.

The compact writer uses two counters: live entries grow from index zero, other
entries grow backward from `chunk_count - 1`. Every valid input lane writes one
entry; the two regions cannot overlap because their total count is bounded by
the chunk size. Padded launch lanes return before reserving an entry. The water
kernel processes all entries, including the terminated group.

Storage is `64 * history_chunk_size` bytes plus two 64-bit counters in compact
mode. There is no host readback of the live count and no extra sorting kernel.
Kernel event timings report beamline and water separately; overall throughput
must still include setup, secondary/neutron transport, readback and output.

## Validation scope

Evidence and benchmark manifests are under
`evidence/proton_primary_split_20260928/`. Initial acceptance uses 170 MeV proton
copper minibeam, full physics and FP32 dose with the established Water_75eV
reference. Transport/EM and neutron integer counts are compared to frozen runs;
quality, energy accounting, queue overflow and dose maps are also checked.
FP32 atomic summation order changes are expected under reordered scheduling.
This does not extend the physical validation of the experimental neutron model.

## Open-field uniform water

```yaml
minibeam: false
water_primary_specialization: true
minibeam_primary_specialization: false
minibeam_primary_pipeline: combined
```

The opt-in open-field route uses the same compile-time uniform-water Urban
transport body, including dynamic projectile ZA. Stage 5 samples the original
source and transports directly in water, without a beamline handoff or entry
partition. The existing source-to-water positioning is retained. It requires
unified EM and unified water with global Urban, no CT, no heterogeneous insert,
no slab layers, and `minibeam=false`. Ineligible configurations fail before the
primary launch. Set `water_primary_specialization: false` to use the generic
kernel. Existing minibeam controls retain their meaning and defaults.

This scheduling option does not enable the currently minibeam-only electron
response or neutron model for open fields. The current open-field benchmark
uses local electron deposition and neutral-product energy accounting.
