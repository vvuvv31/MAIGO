# Path A TOPAS 提取（补 FRED 3.76 缺表）

FRED 发行包没有 `libEvents_C12_*.dat`。这里用 TOPAS 4.2.p3 / Geant4 11.3.2
在 `wuwei@127.0.0.1` 上抽三套代理数据，放在 `/mnt/sda/wuwei/path_a_topas_extract/`。

| 任务 | slurm | 内容 | 对应缺口 |
|---|---|---|---|
| C-12 水涨落 | `run_fluctuation.slurm` | 8 能量 × 4 面密度，50k/点，EM-only | Vavilov LUT |
| C-12 弹性 | `run_elastic_events.slurm` | H1/C12/O16，XS 0–400 MeV/u + 95/200/300/400 事件 | ENDF C+p 弹性 |
| 薄靶碎裂 | `run_thin_fragment.slurm` | H1/C12/O16 1 mm，INCLXX 相关末态 | `libEvents_C12_*` |

2GR MCS 仍用 FRED `libFred.data` 的 `data/mcs/*_2GR.txt`（不是 TOPAS 过程）。远端已有 `c12_primary_msc_summary_400_emonly` 可作 Highland 标定。

提交（在 127.0.0.1 上，数据在 `/mnt/sda/wuwei`）：

```bash
sbatch /mnt/sda/wuwei/path_a_topas_extract/run_fluctuation.slurm
sbatch /mnt/sda/wuwei/path_a_topas_extract/run_elastic_events.slurm
sbatch /mnt/sda/wuwei/path_a_topas_extract/run_thin_fragment.slurm
```

完成后：涨落点 JSON → `compile_energy_loss_fluctuation_grid.py`；弹性 phsp → 替换 `calculate_sigma_el_H_mb`；薄靶 reaction ntuple → FRED 风格事件库。
