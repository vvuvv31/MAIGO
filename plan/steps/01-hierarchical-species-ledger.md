# 01：Species 分层 Energy/FOV Ledger

## 目标

将每个 isotope/species 的 FOV dose 唯一分解为：

`D_FOV = K_birth × f_dep × f_FOV`

- `K_birth = Σ birth kinetic energy`
- `f_dep = unrestricted deposited energy / K_birth`
- `f_FOV = in-FOV deposited energy / unrestricted deposited energy`

任一 >2% 的误差必须先归类为 birth、transport、FOV、cascade 或统计，不允许直接调参。

## 分层维度

`isotope × generation × projectile(Z,A) × target(H/O) × projectile-energy-bin × birth-depth-bin × channel`。

- Energy bin 使用 runtime support 对齐的 hybrid-log grid，同时输出 50 MeV/u coarse view。
- Depth bin 与 0.5 mm 3D scorer z grid 对齐；高维账本可用 sparse key/shard，不得为全矩阵无界增加显存。

## 每层记录

1. `N_birth`、`ΣE_birth`、birth direction/depth。
2. continuous ionization deposit、NIEL diagnostic、process-local nuclear deposit。
3. parent pre/post KE、energy transferred to charged/neutral/unsupported children。
4. cutoff/end-of-track deposit、longitudinal escape、lateral escape、queue overflow、generation-limit energy。
5. unrestricted deposited energy、in-FOV deposited energy、FOV acceptance。
6. secondary reaction count、lookup query/hit/miss/invalid。

## 执行

1. 先定义 host/device 共享 slot/schema，JSON 写出 layout 和 unit。
2. 以 primary/secondary killed/continued、neutral、unsupported、cutoff、escape、overflow 合成事件做单元测试。
3. 128 histories smoke 检查 sparse-key closure；再跑 200 MeV/u 100k baseline。
4. 按 species 生成 birth→transport→FOV waterfall 表，优先 Be6/7/9/10、Li6/7。

## 当前证据（2026-09-01）

- 已实现 18 isotope × 10 metric 聚合 ledger，JSON 输出固定 schema。
- 200 MeV/u 100k G1 本地 RTX 2080Ti 验证：编译与 2/2 tests 通过；全局 energy balance 与冻结基线一致。
- Be/Li/B/C isotope closure 多数已到 `<0.05%`；p/d/He4 仍有 17–32% 未解释残差，Step 01 尚未完成。
- 正式 80×80 mm FOV acceptance：Be/Li 均约 99.89–100.00%，Be6 为 99.9995%；其积分误差不应归因于 FOV/MCS。
- 诊断 A/B total IDD integral 漂移 `+0.0194%`（原子累加调度次序），无 physics/RNG 参数修改。

## 退出条件

- [ ] 每层 unexplained residual `<0.1%` 的该层 `K_birth`。
- [ ] 全局 numerical closure 保持 `<1e-5`。
- [ ] 每个 >2% species residual 都能归入一个首要阶段。
- [ ] 诊断 on/off 不改变 IDD 物理结果。

## 禁止项

不在本步修 sampler、yield、MCS、stopping 或 straggling。
