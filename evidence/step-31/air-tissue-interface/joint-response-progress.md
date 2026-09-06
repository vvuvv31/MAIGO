# Joint electron response: diagnostic checkpoint, NOT interface acceptance

2026-09-06. No commit/push, production package replacement or patient run.

Independent data: `/mnt/sda/wuwei/electron_joint_conditional_bulk_candidate_r3_20260906/`.
CSV SHA256 `72b98923283f1e07208e02ce2e2192cf4507f7128fd8549e24a4e19cbcced62d`.
Metadata SHA256 `6f579b9623fd4c7bd08cbb5a53fe822f90d300c20a3bfa8b654010ab92b62b86`.
Source hashes and compiler provenance are inside metadata. Failed earlier compilations retained.

`interface_gpu_report.json`: both directions, 10/20mm buffer, fixed depth gates pass.
`stability_3d_report.json`: independent seed and 0.1mm step also pass depth gates.
However tissue→air radial RMS 2.375mm vs TOPAS3.442mm at first2mm; do not promote.
All GPU candidate quality reports retain `unvalidated_electron_joint_response`.

`/mnt/sda/wuwei/joint_homogeneous_control_20260906/stability_3d_report.json`:
matched homogeneous geometry controls. Slurm2460–2463 all0:0, 96CPU/40GB.
Air RMS agrees within about2.3%; tissue absolute difference about0.005mm.

`path_collapse_audit.json`: direct-root-only geometric diagnostic on existing TOPAS
interface ntuples. It is NOT a full-family response or predicted interface dose.
Collapsed-path energy-weighted position RMS error1.14–1.51mm in tissue→air;
uniform-density algebraic identity holds within1.5e-14mm.

Default-off regression SHA256 unchanged:

- air→tissue: `ab5374973c3a234190d83dab4d2172791b461c037264292c885ccb74b45e7408`
- tissue→air: `68f5f5417199e2f0320e4f083d4c6fcf70c174eb4aa0d320ac31a1f963f53d30`

Next: independently reconstructed ordered paths, first offline, then isolated GPU
only if justified. Geometry primitive alone is not a validated electron model.
