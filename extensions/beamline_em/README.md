# 铜／空气非受限电子响应

此离线工具使用 Geant4 11.3.2 EM option4 的 `ComputeElectronicDEDX`
提取完整电子阻止本领，并提取原生 Universal/Ion
`Dispersion(tcut=Tmax)`。不提取或抽样独立硬 δ，不包含核阻止本领。
核反应与 Urban 散射包独立保留。工具不链接到 GPU runtime。

从当前铜碎片 Urban 包读取 ZA 清单（包括源质子、C12 及所有带电产物），
然后构建并提取；不要手写一个仅含质子和碳的清单代替碎片覆盖。

```bash
source /path/to/geant4-11.3.2/bin/geant4.sh
cmake -S extensions/beamline_em -B /path/to/work/build \
  -DGeant4_DIR=/path/to/geant4-11.3.2/lib/cmake/Geant4
cmake --build /path/to/work/build -j4
python3 - <<'PY'
from pathlib import Path
import struct
b=Path('data/scattering/urban/copper_fragments.bin').read_bytes()
magic,version,nm,ns,nn,nf,ng,validated=struct.unpack_from('<8s7I',b)
assert magic==b'URBANV22' and version==2 and validated==1
ions=[struct.unpack_from('<II',b,36+nm*20+i*12) for i in range(ns)]
Path('/path/to/work/species.txt').write_text(''.join(f'{z} {a}\n' for z,a in ions))
PY
/path/to/work/build/extract /path/to/work/species.txt \
  /path/to/work/unrestricted.bin /path/to/work/heldout.csv 8193
```

提取完成后将二进制及对应的提取记录安装到 `data/stopping/beamline/`，配置
`minibeam_em_package_file: data/stopping/beamline/unrestricted.bin`，
再更新 `examples/preset_dependencies.json`。
仅需离线提取一次；运行时不依赖 Geant4。

`BLEM0001` 固定包含 G4_Cu、G4_AIR 两种材料、明确 ZA/质量、材料 I/e0、
全电子阻止和两种完整方差。初始能量网格为每物种 8193 个对数节点，
范围 1e-6～6500 MeV/u，并加入原生模型边界及自适应细化。
独立 offset=0.37 探针须满足最大相对误差 ≤5e-4。
运行时按同一分段线性 stopping 的倒数解析积分构造 CSDA 并解析反演。
原生跃迁附近紧邻节点的射程前缀使用 double，避免 FP32 前缀重合；
能量和剂量计分仍为 FP32。首节点以下显式采用 sqrt(E) 延拓。

缺少物种或覆盖、无效／截断／不匹配包均拒绝运行，无旧路径、旧格式或
按电荷平方替代的数据回退。当前仅支持质子/C12 源；通用带电碎片继续保留。
历史资源压缩包不含本包，更新到此模型时必须补装。
