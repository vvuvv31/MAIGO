# Step 27 — Package provenance audit

## Objective

Independently re-derive package facts without reusing the compiler's own
judgments.

## Inputs

- Step-26 v2 package + metadata + raw validation report

## Outputs

- `schneider-secondary-v2/package_audit.json` (independent recomputation)

## Hard gates (any failure stops the line, no GPU Gamma)

- channels = 169
- missing = 0
- gap-too-large = 0
- covered = 169
- uncovered demand fraction = 0
- empty nodes = 0
- duplicate nodes = 0
- target alias = false
- all package/raw hashes verified

## Acceptance

All nine gates pass with hashes matching the metadata declarations.

## Prohibitions

- Do not reuse compiler-internal decisions as audit evidence.
- Do not proceed to GPU validation on any gate failure.

## Commit intent

`test(ct): independent audit of v2 secondary package`
