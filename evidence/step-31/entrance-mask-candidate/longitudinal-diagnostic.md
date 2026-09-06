# Longitudinal electron diagnostic — preliminary, not a physics table

## Follow-up: binary joint response and terminal energy audit

Local Slurm job 2338 completed (2 CPUs / 4 GB, 12 histories, seed 916204).
Only the ntuple output format changed. Its 3D dose values are exactly equal
to the same-seed ASCII run. Output shrank from 233 MiB to 123 MiB; larger
campaigns still require aggregation or bounded history shards.

ASCII rounded both endpoints of one escaping step to z=220 mm, so the new
outward-direction check correctly refused that input. Native binary retains
the displacement and passes at 1e-8 mm boundary tolerance. Do not weaken
the direction check to accommodate rounded ASCII.

The audit now checks contiguous track steps, secondary birth coverage,
positive-KE terminal placement and outward direction. It rejects re-entry,
non-unit weights and non-C12/electron/photon particles: it is deliberately
restricted to this homogeneous EM slab in vacuum, not general CT geometry.

- Electron root birth energy: 146.739176590 MeV.
- Electron-family deposit: 143.944623071 MeV; escape: 2.794553519 MeV.
- Family relative residual: -1.24e-16; global residual: 2.54e-16.
- 3D dose / step energy: 1.000000000182.
- Joint radial/longitudinal energy histogram sums to all family deposit.
- Nine analyzer tests pass, including binary/ASCII equivalence, truncated
  binary rejection and terminal-energy failure cases.

See `binary12-joint-escape.json` for the joint histogram and hashes.
Reproduce using `binary12_steps.phsp`, `binary12_dose.csv`, and additionally
`--format topas-binary-le --slab-bounds-mm -100 100 -100 100 0 220` in the
command below. Input configs and Slurm scripts have been copied into the
artifact directory without overwriting existing files.

The >0.5 mm forward fraction is stable (28.878%). The strictly-positive
fraction changes from 70.1% to 72.6% when sub-micron offsets are preserved;
the old zero-threshold number must not be used as precise calibration.
No longitudinal GPU correction or new Gamma run is justified yet: this
remains 12 primary histories in one energy/density/finite geometry.

Two independent local Slurm jobs (2336 and 2337), 12 C12 histories each,
200 MeV/u with 1% energy spread, HU -1000 homogeneous slab, EM only.
Each used 2 CPUs / 4 GB. Both completed with exit 0.
Job 2335 was cancelled after its all-step ASCII output grew beyond 14 GB;
its incomplete artifacts are retained and excluded from analysis.

| Quantity | seed 916204 | seed 916307 |
| --- | ---: | ---: |
| Recorded steps | 889442 | 887643 |
| 3D integral / step energy | 0.9999999943 | 0.9999999905 |
| Electron-family fraction of deposited energy | 34.05% | 33.55% |
| Electron energy deposited >0.5 mm forward of root birth | 28.88% | 27.93% |
| Energy-weighted longitudinal 90th percentile | 10.68 mm | 8.26 mm |

The forward fraction denominator is **electron-family deposited energy**,
not beam energy. Descendants are assigned to their original primary-daughter
electron through track ancestry. This demonstrates nonlocal longitudinal
deposition in this geometry; it does not measure how much patient Gamma will
improve. The long tail remains statistically unstable at 12 histories.

## Reproduction

Raw data and analysis JSON (including raw/header/dose SHA256):
`/mnt/sda/wuwei/delta_longitudinal_audit/`.
Prefixes: `smoke12` and `smoke12_seed2`.
Configurations: `/tmp/delta_longitudinal_smoke.txt` and
`/tmp/delta_longitudinal_smoke_seed2.txt`, including the original `run.txt`
in the artifact directory. Slurm scripts share the configuration basenames.

```sh
python3 tools/analyze_electron_deposit_steps.py \
  --steps /mnt/sda/wuwei/delta_longitudinal_audit/smoke12_steps.phsp \
  --dose /mnt/sda/wuwei/delta_longitudinal_audit/smoke12_dose.csv \
  --voxel-volume-mm3 2 --histories 12 \
  --output /tmp/delta_smoke12_recheck.json
python3 tests/test_analyze_electron_deposit_steps.py
```

The analyzer requires a completed 21-column header, exact entry/history
counts, complete ancestry, and 3D/step energy closure within 0.1%. Inputs
above 512 MiB are refused before loading. Six synthetic tests pass.

## Decision and next gate

Keep the entrance-mask fix as a candidate; its paired single-shard results
are in `validation.json`. Do not promote it as a full20 improvement.
Do not turn these finite-slab offsets into a universal convolution kernel:
escaping electron energy is absent, and the offsets are conditioned on
deposition inside this slab. Existing transverse-tail energy must not be
counted a second time.

Next implement a bounded-output electron-response extractor with joint
radial/longitudinal offsets, root birth energy and explicit escape accounting.
Validate energy closure and held-out energy/density/entrance geometries
before changing GPU redistribution. Keep the exact validated v2.1 stack
and frozen full20 results unchanged. No new Gamma or package promotion
is supported by this diagnostic.
