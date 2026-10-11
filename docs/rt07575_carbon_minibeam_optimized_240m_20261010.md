# RT07575 head carbon minibeam optimized weights:240M histories

Completed on RTX A6000 at v@172.17.152.19. Same user-selected research physics as60M: chunk262144, patient1mm/CTclock1/16, Cu0.05mm, air0.2mm, primary fluctuationsON, secondary restricted fluctuationsOFF, explicit electronsOFF. Original400spot90/270 optimized source, weights, CT/placement/normalization preserved. Independent seeds, global largest remainder allocation once,96balanced2.5M batches. Dose scale W/240M; sum both fields/all96batches once. Production defaults unchanged.

Reference: original physical optimization DIJ,100k histories/column; local3%/0.2mm/search0.02mm only.10%reference-peak mask excludes the entireGTV because reference peak55.13Gy; patient>=0.4Gy and GTV are therefore reported separately.

| Scope |60M local gamma|240M local gamma|Change pp|
|---|---:|---:|---:|
|patient_10pct_prescription|80.231212%|81.502245%|+1.271032|
|patient_10pct_reference_peak|87.820484%|88.686958%|+0.866474|
|GTV|70.459210%|75.269369%|+4.810159|

|Scope|60M voxel SE RMS|240M voxel SE RMS|
|---|---:|---:|
|GTV|2.196067%|1.102877%|
|patient_dose_ge_0p4Gy|3.326473%|1.669833%|

GTVmean GPU4.011845Gy vs reference3.979940Gy. Warm73657/s, wall54.46min. All192field audits pass,zerooverflow,maxenergyresidual8.69412e-07.

GPU noise does not include reference DIJ variance or model bias. HigherN does not validate the clock/step/secondary fluctuation approximations. No newTOPAS or fitted dose scale/registration.

Artifacts: /mnt/sdb/wuwei/MAIGO_clean/out/rt07575_carbon_minibeam_optimized_240m_20261010/results/report.json,noise_report.json,comparison_60m.json,dose_GPU_4Gy.mhd/raw and profiles/. Batch audits/doses in remote_results/case/gpu/.
