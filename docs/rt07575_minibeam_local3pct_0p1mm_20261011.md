# RT07575 minibeam: local3%/0.1mm rescore, 2026-10-11

Reuse completed normalized GPU/TOPAS dose arrays. No new Monte Carlo, fitted scale, shift or registration. Both plans use prescription4Gy threshold3%=0.12Gy in patient BODY union GTV and GTV; local tolerance remains3% of per-voxel reference dose. Search0.02mm; patient grid XYZ0.5/0.5/2mm, unchanged.

| GPU result | GPU histories | Region | Local3%/0.2mm | Local3%/0.1mm | Change pp |
|---|---:|---|---:|---:|---:|
| proton_patient0p2mm | 236086432 | patient_prescription3pct | 93.370477% | 83.951565% | -9.418912 |
| proton_patient0p2mm | 236086432 | GTV_prescription3pct | 98.489610% | 96.818881% | -1.670729 |
| proton_patient0p2mm | 236086432 | patient_historical_peak10pct | 98.993851% | 96.968619% | -2.025232 |
| proton_patient0p5mm | 236086432 | patient_prescription3pct | 93.278681% | 83.884826% | -9.393856 |
| proton_patient0p5mm | 236086432 | GTV_prescription3pct | 98.380580% | 96.664956% | -1.715623 |
| proton_patient0p5mm | 236086432 | patient_historical_peak10pct | 98.941577% | 96.913111% | -2.028465 |
| carbon_inline | 240000000 | patient_prescription3pct | 93.442270% | 87.279196% | -6.163074 |
| carbon_inline | 240000000 | GTV_prescription3pct | 94.574141% | 91.325680% | -3.248461 |
| carbon_Cu_child_queue | 240000000 | patient_prescription3pct | 93.442270% | 87.279196% | -6.163074 |
| carbon_Cu_child_queue | 240000000 | GTV_prescription3pct | 94.574141% | 91.325680% | -3.248461 |

Carbon inline versus Cu queue has zero changed pass/fail voxels at0.1mm in either frozen region, corroborating the scheduling-only decision.

Proton uses original487active-spot historical TOPAS weighted DIJ (1M/spot); carbon uses400optimized-spot90/270 unthresholded TOPAS replay150711884histories. These are distinct plans/references. Proton historical reference variance unavailable; carbon historical TOPAS lacks GenericIon nuclear elastic. More stringent gamma alone cannot attribute failures to Monte Carlo noise or physics bias. DTA/search spacing is not scoring resolution.

GPU research settings: proton236086432 histories per arm, patient0.2 or0.5mm, Cu0.05mm/air0.2mm, all ion fluctuationsON/native clock/electronOFF. Carbon240M per arm, patient1mm/clock1/16/Cu0.05mm/air0.2mm, primaryfluctON/secondaryrestrictedfluctOFF/electronOFF. No preset/default changes from this rescore.

Artifacts: `out/rt07575_minibeam_local3pct_0p1mm_20261011/report.json`, reference/GPU hashes, per-arm/region passed masks, analyze.py, analysis.log.

## Proton threshold updated to 10% of prescription

User requested proton threshold10% of4Gy prescription=0.4Gy, rather than10% of whole-volume peak. BODY∪GTV includes325988reference voxels; GTV includes all31184voxels. Same236086432histories, physical dose/reference/normalization, local3% tolerance and search0.02mm.

| Proton patient cap | BODY∪GTV local3%/0.2mm | BODY∪GTV local3%/0.1mm | GTV local3%/0.1mm |
|---|---:|---:|---:|
| 0.2mm | 97.538253% | 92.542977% | 96.818881% |
| 0.5mm | 97.440090% | 92.391131% | 96.664956% |

All GTV voxels exceed both0.12Gy and0.4Gy, so GTV gamma is unchanged by this threshold update. Carbon remains at its existing prescription3%=0.12Gy threshold. Artifacts: `proton_prescription10pct_report.json`, script and pass masks in the same output directory. This is postprocessing only.
