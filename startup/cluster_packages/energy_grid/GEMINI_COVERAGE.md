# Gemini：每个专用能量包能覆盖多大能量范围

Grok 已在集群提交 100/150/200/250/300/350/400 MeV/u 各 1M 的 TOPAS
`CarbonCascadeNtuple` 作业。你的任务是：**等 7 个作业都编完包之后**，
用一级 inelastic 的分箱产额，估计每个专用包沿能量轴能用多宽，并给出
用最少能量点拼 100–400 MeV/u 生产包的方案。

不要重跑 TOPAS，不要改 GPU 输运。只读表、写报告。

## 数据在哪

登录节点：`wuwei@10.10.10.4`

```
/LustreData6/home/wuwei/carbonGPU/maigo_packages_1M/energy_grid/
  slurm_c12egrid_<arrayid>_<0-6>.{out,err}
  run_<E>MeVu_1M/
    topas_<E>MeVu_water_inclxx_1M_primary_reactions.csv.gz
    topas_<E>MeVu_water_inclxx_1M_primary_secondaries.csv.gz
  packages/
    topas_<E>MeVu_water_inclxx_1M_primary_bin_yields.json   ← 主输入
    topas_<E>MeVu_water_inclxx_1M_primary_3d.bin
    topas_<E>MeVu_water_inclxx_1M_primary_3d.compiled.json
    topas_<E>MeVu_water_inclxx_1M_cascade_3d.bin
```

作业是否结束：

```
squeue -u wuwei | grep c12egrid || true
ls packages/topas_*_primary_bin_yields.json
```

7 个 `*_primary_bin_yields.json` 都在才能开始。缺哪个就看对应
`slurm_c12egrid_*_<task>.err`。

`primary_bin_yields.json` 是**真实记录**的一级 inelastic（track-1、
`event_interaction_id=0`），4 MeV/u 一箱。空箱 `n_reactions=0`。
编 `.bin` 时空箱会从最近非空箱复制（`--fill-empty nearest`），
**覆盖度分析必须忽略这些复制，只看 json 里 `n_reactions>0` 的箱。**

## 要回答的问题

对每个源能量 \(E_s\in\{100,150,200,250,300,350,400\}\)：

1. 这个专用束的一级 inelastic 主要落在哪些 MeV/u 箱（占有率）？
2. 以该束**自身靠近 \(E_s\) 的高统计箱**为参考（建议取
   `on_energy_bin`，或 `n_reactions` 最大且上沿 \(\le E_s\) 的箱），
   同一 json 里更低能量的箱，B/C/N/O/He/proton 产额相对参考偏多少？
3. 以 **\(E_s\) 专用包** 的某个箱，对比 **\(E_t\) 专用包** 在同一能量
   窗口的产额（跨文件）。这是覆盖度的主指标。

通道：`proton`（Z=1,A=1）、`helium`、`lithium`、`beryllium`、`boron`、
`carbon`、`nitrogen`、`oxygen`。重点是 **B、C、N、O**（已知 400 长程
切 200 箱时 N/O 会差数倍）。

## 怎么比

对每一对 \((E_s, E_t)\)，\(E_t \le E_s\)（高能束的低能箱去对低能专用束）：

- 取能量重叠箱 \(b\)，两边 `n_reactions` 都 \(\ge 200\)，否则标 `insufficient`。
- 相对误差 \(\delta_k = (Y_s(b,k)-Y_t(b,k))/Y_t(b,k)\)，\(Y_t\) 是
  \(E_t\) 专用包在箱 \(b\) 的 multiplicity。
- 箱合格：B、C、N、O 的 \(|\delta_k|\) 都 \(\le 10\%\)；同时报告 15% 和
  20% 两档。He/p 只记录，不挡合格（除非你发现它们才是限制因素）。
- \(E_s\) 包的覆盖下沿 = 仍连续合格的最低箱下沿。上沿 = \(E_s\)
  （专用束几乎不提供 \(>E_s\) 的一级 inelastic）。

再做一次 **只比 B/C**（忽略 N/O）。若 B/C 覆盖远宽于 N/O，结论里写明：
弹核碎块和靶残核要分开拼。

不要用编出来的 `.bin` 做产额统计。不要把 fill-empty 的箱当真实样本。

## 交付

写一份 markdown，放到：

`/LustreData6/home/wuwei/carbonGPU/maigo_packages_1M/energy_grid/coverage_report.md`

并在本仓库复制一份到 `startup/cluster_packages/energy_grid/coverage_report.md`
（若你在本机工作树）。

必须包含：

1. 7 个源能量的占有箱表：`E_s`、总一级反应数、非空箱数、峰值箱、峰值箱反应数。
2. 覆盖矩阵：行 = 专用包 \(E_s\)，列 = 测试能量 \(E_t\)，单元格 =
   `ok10` / `ok15` / `ok20` / `fail` / `insufficient`，括号里写最差物种和 \(\delta\)。
3. 每个 \(E_s\) 的推荐覆盖区间 \([E_\min, E_s]\)（10% 档），以及限制物种。
4. **最少能量点拼接方案**：用尽量少的 \(E_s\) 盖住 100–400 MeV/u
   （10% 档优先；若点数多到不合理，再给出 15% 方案）。
   写成「生产包第 \(i\) 箱用来自 \(E_s\) 的真实事例」。
5. 若某专用束低能箱全空或 \(n<200\)：写明该束**不能**当更低能量的参考，
   不是物理上 12C 变了。

## 约束

- 中文写结论，表格可用英文列名。
- 用户不要再跑 TOPAS；缺文件就说缺哪个作业，不要编造产额。
- 不要 commit，除非用户明确要求。
