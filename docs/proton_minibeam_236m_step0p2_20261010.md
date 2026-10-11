# RT07575 head proton minibeam, 0.2mm, 236086432 histories (2026-10-10)

A6000 sm86 on v@172.17.152.19, direct run (noSlurm). Original487active spots, original optimized weights, source20floats/emittance, CT/placement, Cu60mm/15slits/0.5mm width/3.6mm pitch/50mm length preserved. Patientcap0.2mm; Cu0.05mm; airslit0.2mm; native patientclock; primary/secondary/Cu fluctuationsON; explicit electronsOFF; historychunk262144. Production/defaults untouched. Frozen restored proton-mini transport library from previousscreen, onlyoffline numerical-source driver adapted to24batch count. Allsources/hash/counts verified afterretrieval.

## Result

| Metric | Result |
|---|---:|
| Simulated incident protons | 236,086,432 |
| TOPAS local3%/0.2mm, search0.02mm | **98.993851%** |
| Passed/evaluated voxels | 183,692/185,559 |
| Strict >99% | False |
| GTV mean GPU / TOPAS | 4.922194 / 4.918585Gy |
| GTV mean change | +0.073385% |
| GTV D95 GPU / TOPAS | 3.987790 / 4.000110Gy |
| Evaluation-mask dose sum bias | +0.152173% |
| Evaluation-mask localRMS difference | 1.830199% reference |
| GPU voxelSE RMS | 1.100728% reference |
| Warm throughput | 275,318 primary/s |
| Entire campaign wall | 857.668s (14.29min) |
| Maximum relative energy residual | 7.16499e-08 |

Result98.993851% is below99%; do not round98.99 to99 and callpass. Strict >99% needs 183,704 passedvoxels, 12 morethan observed. This is an observed threshold count, not justificationtoignorefailures or proofstatisticaluncertainty coversbias.

## Reference and uncertainty

Original historical weighted TOPAS1M-per-spot DIJ, AllChargedUrbanPhysics, nofreshTOPAS. Reference SHA256 fcd60f21187c8305ee5a1bf36430b7133143081d99382f473638eb192fbc2645. Patient BODYunionGTV and reference>=10%whole-volume peak gives185559fixedvoxels. Trilinear, sphericalsearch0.02mm atDTA0.2mm, referencegrid0.5×0.5×2mm remainsunchanged; no fittedscale, shift, registration or reflection. Original packed_xneg patient mapping only.

Previous12M0.2mm gamma86.9621%→236M98.9939% withsamephysicalsettings; differentindependentseedcampaign. Increase chieflyshowsstatisticaldependence, not newphysicschange. GPU_SE descriptivefrom24near-equalwholeplanbatches, TOPASvoxelvarianceunknown. RemainingRMS1.8302% versusGPU_SE1.1007% containsreferencevarianceandpossiblemodelbias; cannotattributeentireerror toMC orpatientstep. No matched236M0.1mm arm, cannotclaimthisrun measuresstepbias orspeedgain precisely. NoextraNwasaddedtoforce99%.

## Campaign and audits

Global largest-remainder allocation once across236086432histories, balanced24batches9836934/9836935each, all487spots perbatch. Distinctsource seeds independentfrompilot. Dose scale 1000.0000013444387=originalprescribedincidentprotons/236086432; sum24normalizeddoseoutputs once, noaveragingor×24. EachbatchsameGPUcontext/packages; dose/quality/energy/complete savedindividually. No within-batch checkpoint. Newofflineprogressusesactual24batch/236Mtotals.

All24quality/EM/energy/coverage/completionaccepted; zerooverflow; allCu primarychild taskscompleted; noelectronqueue. FinalGPUstate NVIDIA RTX A6000, 1800 MHz, 84, 122.54 W; memorytracked/runtimeorderingpreserved. Compile/source/auditprovenance inexperimentdirectory; noaliases/retiredphysics/neutraltrackingadded.

## Files

Experiment: out/proton_minibeam_236m_step0p2_20261010/. Results/report.json, dose_4Gy.mhd/raw, TOPAS_local3pct_0p2mm_passed.npy, dose_comparison.png and analysis_complete.json. All24rawbatch/auditoutputs inremote_results/case/gpu/. Remote campaign under /home/v/MAIGO_clean_a6000_20261010/proton_minibeam_236m_step0p2_20261010/. Recompute python3 out/proton_minibeam_236m_step0p2_20261010/analyze.py. BinarySHA256 eaa47107014517fd7f8766f67a7ae4fc6ba0c25e0050ce0d00e5583840826bc9.
