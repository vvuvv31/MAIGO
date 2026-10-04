# Proton CT minibeam joint electron pilot

Experimental integration requested before full CT acceptance. Default proton preset remains on the original response path. FP32 dose/atomics; primary step cap0.25mm; new joint charged native steps0.05mm. Existing fallback bank can contain longer native chords; the pilot does not certify a0.25mm cap for that legacy path.

Selected operator: `out/electron_joint_family_20261003/model065_cutoff`, report SHA256 `32074c6b6dcca1eca65ae1bb58ff2df0cc31dfa080cf9c26b30a0665a406efc8`. This is the unthinned model, **not** `model065_reduced` or `model065_all_electrons`. Native training uses an isolated Geant4 GS displacement-control binary; official voxel reference remains unchanged. Full photon and photon-descendant graphs retained. At0.4MeV,20M GPU packets versus10M native voxel histories gave peak−0.155414%,valley−0.258908%,maximum0.1mm profile difference0.564888%; this is a homogeneous electron screen, not patient acceptance.

The frozen pilot artifact is213144512bytes (203.270MiB), SHA256 `a20883e9f73e9bba5b09bf88654bcd2b9c54a6e8fac76990188d006b52751a73`. It adds native graph pre-directions for exact state handoff. Loader checks the full SHA and baseline proton index pin. Only Schneider section1 response node0.2994157075881958g/cm³ and electron energy up to~0.65MeV are eligible. This is electron KE, not proton beam energy. Original birth energy/direction/weight sampling is retained.

At each charged step, before drawing a new outcome, the complete displacement-support neighborhood must have one physical voxel density and section. At an interface the electron resumes the existing continuous-energy crossing law from its current position/direction/KE, permanently leaving the pilot for that packet. No trajectory rejection, dose normalization, local residual dumping or water-material substitution. Spatial distances follow the existing sampled-density-node convention without adding density rescaling. Photon native segments use exact CT boundary traversal: real patient exit is escaped energy; unsupported photon continuation at a heterogeneous interface is explicitly untracked, as in the original path.

The legacy bank stays loaded for unsupported materials/energies/interfaces and original electron births. Consequently this pilot does **not** reduce whole-process response-bank memory. The new package is retained across batches. Quality reports explicitly identify the new unvalidated CT approximation while keeping transport/energy/queue failures fatal.

Reproduce a matched pair (each output directory must be new):

```bash
bash scripts/build.sh
python3 scripts/run_joint_pilot.py --histories 1000000 --output out/joint_control
python3 scripts/run_joint_pilot.py --histories 1000000 --output out/joint_candidate \
  --package out/proton_joint_pilot_20261003/package/joint_pilot.bin
```

Host boundary/packet/photon tests:

```bash
g++ -std=c++20 -O2 -Iinclude tests/electron_joint_pilot.cpp -o /tmp/check_joint_pilot
/tmp/check_joint_pilot
```

## 实际 CT 首轮结果

RTX2080Ti，肺CT、170MeV proton、group32，固定种子1002450001，各100万primary。最终二进制和源文件哈希见 `out/proton_joint_pilot_20261003/final_identity.json`。结果为 **GPU旧路径 versus GPU新混合路径**，不是新的TOPAS对照。

- 原有电子包8,467,976个；满足节点/能量条件145,056个；实际进入新模型88,473个（全部电子包的1.0448%）。新模型完成1,016,044步；44,713个包沉积，43,735个切回旧路径，25个出现显式未覆盖光子续接，无无效包。受影响剂量的比例不能直接用包数比例代替。
- 主输运/EM审计相同；quality无failure，queue overflow=0。research质量报告显式标注 `unvalidated_material_electron_joint_pilot`，并不等于患者剂量验收。
- 六个深度段（0–20、20–80、80–120、120–180、180–230、230–303.5mm），横向|x|≤15mm、|y|≤15mm，原始绝对剂量不重归一化：峰区合并剂量最大变化0.002587%，谷区0.010705%。按0.1mm横向bin合并y和深度后的剖面最大局部差0.2991%。峰/谷按3.6mm节距到最近峰中心≤0.25mm/≥1.55mm定义；尾部同样是几何位置掩膜，不意味仍有清晰峰谷。
- **原始体素局部差尚未达到1%验收**：reference最大剂量1%阈值下，local3%/0mm通过97.6847%，最大局部差31.828%；0.1%阈值下98.8233%，最大238.24%。这些数据包含不同电子抽样带来的有限统计差异；尚未分离噪声与模型偏差。10%阈值仅剩1453个体素，通过率100%不能作为谷区验证。
- 全网格Gy值之和相对变化+0.00004975%（不是异质网格总沉积能量）；能量残差相对入射总能量约3.17e−7。显式未追踪光子能量由688.967342变为691.532493MeV，其差值不能当作沉积剂量。
- 最终同二进制关闭/启用新路径，电子队列2.20198/2.48917s，观察值约慢13.0%。只有一次配对，非稳定吞吐置信区间。原冻结基线端到端37.96s、最终candidate20.60s主要受加载缓存影响，**不能宣称端到端加速**。
- 关闭开关对冻结旧二进制回归：全部离散审计通过，剂量相对L1差4.64e−8，最大绝对差/全图峰值6.43e−8；无新增/丢失非零体素，无体素相对变化超过1%。

首轮仅证明实验路径已实际运行并完成基础守恒/边界检查。低覆盖率会稀释全图差异；它不能证明全面替换电子包后的精度或速度。默认预设保持旧路径。

结果与图：

- `out/proton_joint_pilot_20261003/comparison/report.json`
- `out/proton_joint_pilot_20261003/comparison/profiles.png`
- `out/proton_joint_pilot_20261003/{baseline_1m,candidate_1m,disabled_1m}/`
- 主机几何/状态交接/光子测试、SHA/截断/作用域拒绝测试均通过；见 `host_tests.log`、`rejections.json`。

连续2×1万primary测试通过：新joint包和旧bank均在第二批命中缓存，两批quality均通过且零overflow。详见 `out/proton_joint_pilot_20261003/batch_check.json`。

## RT07575 full-plan 接入

现已增加已固定SHA的82表RT07575扩展索引兼容。逐一确认其63个原始表条目在解析路径后完全相同（含材料身份、数据及状态记录哈希），其余19表继续走旧路径。未修改joint模型二进制、训练数据或物理参数。

0.1% full plan（236,086,432 primary、487spots、119批）已完成，FP32、group32、单进程。119批质量/守恒/零溢出通过；新模型实际进入0.4353%的电子包。进程1357.96s；local gamma 1%/0.2mm92.0074%、2%/0.2mm97.1416%、3%/0.2mm99.0774%、3%/0mm92.2407%。对照是既有TOPAS加权DIJ，未重跑TOPAS。范围仍是实验混合路径，不能由此认证整个电子库替换或内存/速度改善。

完整报告：`benchmark/protonCT/RT07575_minibeam_GPU_joint_20261003/REPORT_zh.md`；同目录保存119批配置、源文件链接、冻结二进制、索引兼容验证、逐批审计、剂量、gamma及图。
