# Authorized unified EM production integration

The user explicitly authorized the exception recorded in AGENTS.md on 2026-09-13.
Package SHA256: `8c5d970b3b639bfca2f448730271bed4fc04721aba73100e2efbe09dffe44855`.

Local RTX 2080 Ti runtime checks:

- Water: 10,000 histories, production pass, zero overflow, zero EM failures.
- Layered Schneider CT: 50,000 histories, production pass, zero overflow, zero EM failures.
- Both quality reports retain `unified_material_em_accuracy_pending`.
- Both production entry configs pass eight positive/negative configuration checks:
  valid mode, package pin, primary/secondary fluctuations, scale, CT faces, old-model
  stacking, and electron-response stacking.
- The v2.1 data verifier and unified EM integrity verifier passed before GPU runs.

The CT run is the b3 layered phantom, **not a completed RT07575 patient validation**.
The RT07575 production entry was configuration-checked; its full patient run and
BODY-only Gamma remain outstanding. Low-density production-cut onset accuracy
also remains pending. `pass` refers to runtime quality, not completion of those
accuracy gates.

All-ion nuclear elastic has a separate existing research-only guard. Production
checks omit that bank; existing research entries retain the combined EM/elastic
model. The new EM exception does not promote the elastic bank.

Exact run configs, executable/config hashes, audit counters and quality reports
are preserved here. Full local outputs are in `scratch/unified_em_production_20260913`.
The 1.29 GiB EM binary remains outside Git and was not uploaded to a Release.
