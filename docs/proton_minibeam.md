# Proton source in the copper minibeam path

The 170 MeV research configuration is `config/proton_minibeam_170_fullphysics_urban_v2.yaml`.
Build with `CARBON_ENABLE_MINIBEAM=ON`, `CARBON_DOSE_FP32=ON`, `CARBON_DOSE_FP64=OFF`.
An isolated build for this campaign is `build/proton-minibeam/carbon_mc`.

Source selection remains `primary_particle: proton` (or explicit source Z/A). The minibeam route additionally needs source-specific copper/air stopping, copper elastic and nuclear banks. Non-C12 minibeam currently requires `multiple_scattering_model: urban_v2` and `minibeam_copper_mcs_model: urban_v2`; its pinned Urban package must include natural copper (section -2, actual density) and the source species. The separate legacy water-primary Urban selector must remain off because global Urban already handles that track.

Generalization fixes include source ZA in copper energy fluctuations, actual source mass in copper elastic kinematics, and configured reference charge in the secondary stopping fallback. For packaged copper Urban, nuclear competition and decrement of the collision clock both use accepted true path; geometry may shorten that path. The old CSV-only C12 Urban route is retained.

Benchmark geometry: 170 MeV parallel uniform 30 × 30 mm² incident field; natural copper cylinder radius 60 mm, thickness 60 mm, 15 slits of width 0.5 mm at pitch 3.6 mm, length 50 mm; water starts 60 mm downstream. The water reference is Water_75eV, stopping scale 1.0. Both engines score 0.1 × 100 × 0.5 mm³ voxels over a 100 × 100 × 230 mm³ ROI. Reported transverse profiles integrate over scored y. Normalization is per original incident proton, including histories absorbed in copper.

See `data/proton_copper_20260926/README.md` for package limitations and `evidence/proton_minibeam_170_20260926/` for the 5M-per-engine comparison. The source interface is enabled as a research path; accuracy must be judged from those results, particularly valley dose and finite copper cascade coverage.
