# Step 14 — Replace approximate stopping with TOPAS-derived Schneider tables

## Objective

Provide section-dependent mass stopping power consistent with TOPAS, initially C12 and later compatible with the projectile registry.

## Extraction and product

Extend a deterministic TOPAS extractor to output for C12 x 25 sections x the transport energy grid:

```text
mass stopping power (MeV/mm)/(g/cm3)
linear stopping power at material density
CSDA range or independently integrated range
material/physics provenance
```

Compile `data/schneider/schneider_stopping_v1.bin` plus metadata. Verify mass/linear unit conversion and monotonic energy grid. Do not silently use the simplified `Z/A + I` scaling for a section once an exact table is configured.

## Runtime

Use section ID directly, interpolate on the shared energy grid, multiply by voxel density exactly once, and cap steps at material faces as already required. Preserve legacy fallback only for explicitly named compatibility configs.

## Validation

Compare homogeneous Schneider ranges and primary 3D dose/IDD (transverse sum) at 100/200/300 MeV/u. Required range difference is <0.5 mm or one voxel, whichever is larger. Dose differences must be reported without tuning nuclear physics.

## Do not change

No CINEL final states, no MCS fitting, no water table retuning, no FP64 project.

## Acceptance

Host/device table equivalence passes; exact tables are used in production Schneider mode; all slab range gates pass; P2 attenuation remains within its gates.
