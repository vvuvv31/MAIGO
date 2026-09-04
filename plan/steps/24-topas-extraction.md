# Step 24 — TOPAS extraction

## Objective

Generate the missing correlated final-state events with TOPAS/Geant4 on the
local machine only.

## Inputs

- Step-22 manifest + Step-23 grids
- TOPAS extension/source/build under `/home/wuwei/topas`
- Thin-target event-extractor configuration used by the v1 campaigns

## Outputs

- Raw event files under
  `/mnt/sda/wuwei/cinel03-campaigns/schneider-secondary-v2/raw/`
  (unique output path per job; deterministic seed per task)
- Per-job logs to `/mnt/sda/wuwei/job_%j.log` / `job_%j.err`

## Rules

- Local `sbatch` only. No remote host, no remote cluster, no remote GPU.
- Total concurrent usage <= 192 CPUs and <= 160 GiB RAM; allocate per
  energy/compute weight (high-energy jobs get more CPUs) so batches finish
  together.
- Check queue + used CPUs/memory before submitting; publish the
  job-to-channel mapping.
- Brief `InvalidAccount` at submit time is rechecked after 1-3 minutes,
  not treated as failure.
- Pure event extractor needs no dose scorer; any dose validation uses a 3D
  scorer with transverse summation only (1D scorer forbidden).

## Acceptance

Every manifest task has a completed job with exit 0 and a non-empty raw
output at its unique path; no v1 raw file overwritten.

## Prohibitions

- Do not overwrite old raw data or old binaries.
- Do not change physics list, Geant4 version, or extractor semantics
  mid-campaign (any change restarts provenance).

## Commit intent

`feat(topas): extract v2 secondary CINEL campaign events`
