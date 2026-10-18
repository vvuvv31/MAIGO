# Three-GPU full-plan throughput (2026-10-18)

Same frozen source revision including CT outer-face repair, default physics/numerical settings, all five pipelines enabled. Exact same twelve full-plan fixtures, three consecutive resident batches, same source CSVs/seeds and histories per batch on all devices. Native sm_75/sm_86/sm_120 AOT; CUDA_DISABLE_PTX_JIT=1.

Cold starts a fresh process and includes package reads, context creation and first batch. Warm pools the second/third batches. Compilation excluded; dose output and runtime audits included. Unlocked clocks; no fixed hardware-ratio claim. All 36 device/case runs pass runtime/energy/coverage/completion checks with zero overflow. This is throughput testing, not new TOPAS dose acceptance.

Unit: k primary histories/s. Each cell: cold / warm.

|Case|Mode|Histories/batch|RTX 2080 Ti|RTX A6000|RTX PRO 6000 Blackwell|
|---|---|---:|---:|---:|---:|
|RT07575|proton broadbeam|19278962|740.7 / 756.1|1135.0 / 1213.2|2876.6 / 3261.9|
|RT06423|proton broadbeam|17752335|735.5 / 750.7|1174.5 / 1244.2|3070.0 / 3484.2|
|20022516|proton broadbeam|19002665|573.6 / 587.7|882.6 / 945.1|2588.4 / 2863.4|
|RT07575|carbon broadbeam|4986084|215.2 / 216.8|303.8 / 319.8|928.2 / 1043.1|
|RT06423|carbon broadbeam|4873796|234.3 / 239.3|334.7 / 358.2|991.8 / 1137.9|
|20022516|carbon broadbeam|4921474|210.7 / 218.5|295.9 / 329.8|910.7 / 1050.0|
|RT07575|proton minibeam|20000000|339.9 / 345.4|517.7 / 545.8|1518.6 / 1654.4|
|RT06423|proton minibeam|20000000|377.1 / 383.8|560.8 / 594.9|1655.1 / 1802.0|
|20022516|proton minibeam|20000000|420.1 / 431.2|595.5 / 646.0|1806.4 / 1972.7|
|RT07575|carbon minibeam|2500004|84.2 / 102.3|93.4 / 161.2|217.3 / 373.3|
|RT06423|carbon minibeam|2499983|72.2 / 88.0|85.8 / 142.2|189.6 / 322.5|
|20022516|carbon minibeam|2499980|75.0 / 87.2|82.1 / 129.1|196.1 / 309.9|

Evidence: `tmp/dual_gpu_twelve_modes_20261018/` (build/source/input hashes, per-device results, field-level audits and telemetry). G4 reference: `tmp/urban_ct_outer_face_20261018/`.
