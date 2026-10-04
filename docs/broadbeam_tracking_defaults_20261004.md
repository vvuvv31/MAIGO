# 四个 broadbeam 默认电子 tracking 开关（2026-10-04）

按用户明确要求，将四个默认 broadbeam 预设统一设置为开启电子 tracking。用户进一步确认沿用已验收的**低密度区域＋周围5 mm**范围，而非全区域电子输运。

```yaml
broadbeam_electron_tracking: true
broadbeam_electron_density_threshold: 0.1  # g/cm³
broadbeam_electron_halo_mm: 5.0
```

| 入口 | true 的有效行为 | false 的有效行为 |
|---|---|---|
| `proton_water_broadbeam` | 均匀水密度1.0，选区为空；局部沉积，不加载电子包 | 局部沉积，不加载电子包 |
| `carbon_water_broadbeam` | 同上，水参考仍为Water_75eV | 同上 |
| `proton_ct_plan_fe` | 低密度＋5mm出生选区，proton birth＋shared FP32电子/光子传播 | 局部沉积，不加载空间电子包 |
| `carbon_ct_plan_fe_joint` | 低密度＋5mm出生选区，C12 v2 birth＋同一shared传播；旧文件名保留 | 局部沉积，不调用旧joint/tail |

两个CT默认的物理设置与上一轮通过local1%/1mm >99%的5mm路径一致。只接管主proton/C12的电离电子；核次级离子自身产生的电子仍局部沉积。选中的电子及后续光子在离开选区后继续输运，不在选区边界截断。group32抽样和FP32评分不变；该开关不控制核反应或带电核次级输运。

`false` 在配置展开时移除独立出生、joint、charged、photon包引用，同时关闭原CT joint/tail响应并重置电子队列设置。启动SHA预读取也遵守该开关，避免“输运关闭但仍读取大包”。没有删除历史数据或minibeam入口。

四个默认及两个参考trial YAML都使用新开关。历史 `broadbeam_independent_electrons` 仍可读用于复现，但与新开关冲突会报错，避免一个开关覆盖另一个。新开关限于 `broadbeam_fast`。日志和energy ledger分别记录用户请求及有效CT分支；water明确输出 `empty_uniform_water_selection`。

均匀水只是空选区，不使用Schneider软组织冒充水响应。如果调整密度/阈值让水进入选区，程序明确拒绝，需先补齐原生Water_75eV的出生/传播数据。将water配置用CLI直接改为CT也不会静默关闭tracking，必须使用带相应包的CT配置。

## 运行

两个water保持原命令。2080Ti上的CT采用已验证的连续批次，保持原总历史数并只加载一次包：

```bash
./scripts/run.sh proton_water_broadbeam
./scripts/run.sh carbon_water_broadbeam
./scripts/run.sh proton_ct_plan_fe --batch-count 10 --histories-per-batch 1253150 --batch-output out/proton_ct_broadbeam_default
./scripts/run.sh carbon_ct_plan_fe_joint --batch-count 7 --histories-per-batch 925987 --batch-output out/carbon_ct_broadbeam_default
```

输出根目录须新建或为空。此次没有添加自动拆批或减少YAML总历史数；新carbon电子路径的6,481,909单批曾因次级状态分配失败，不应在2080Ti重试同一单批。

上一轮RT07575共用电子剂量验收为proton99.9120%、carbon99.8677%，阈值参考最大剂量10%，全3D local1%/1mm >99%。开启后的持续吞吐约9.0万/5.4万primary/s；关闭后局部沉积更快，但carbon局部沉积local11仅96.2842%，不具有相同剂量验收。上述是上一轮结果，本次开关回归不冒充新的高统计TOPAS对照。

## 验证记录

证据目录：`out/broadbeam_tracking_defaults_20261004/`，保留修改前二进制和配置。构建固定 `CARBON_DOSE_FP32=ON`、`CARBON_DOSE_FP64=OFF`。

配置单测：`tests/broadbeam_electron_tracking.cpp`，检查四预设ON/OFF、规范配置往返、旧joint/tail残留清理、冲突开关/非法boolean/非法邻域拒绝、water材料域和minibeam作用域保护。可执行和日志为证据目录中的 `config_test` / `config_test.log`。

GPU验证脚本为证据目录下 `verify_runtime.py`，结果记录在 `runtime.json`；核查四预设ON/OFF的实际kernel、能量/EM/覆盖/完成数/零溢出、同进程第二批包复用，以及与修改前二进制相同物理分支的剂量对照。关闭及water空选区测试将电子包指向2MiB哨兵文件，并使用strace确认该文件没有被打开，以覆盖启动预读取。

验证已完成：四场景ON/OFF及旧二进制配对共14次进程、28个原生批次（每批25,000 primary）全部通过。EM与源配额一致，CT开启时电子/光子整数工作计数和完成数与先前实现一致；所有质量/能量/覆盖/零溢出检查通过。剂量相对L1最大2.592e-7，属于FP32累加差异。两个CT第二批均命中电子包缓存，6组OFF/空选区的strace均未打开哨兵电子文件。

正式二进制已更新到 `build/carbon_mc`，SHA256 `40d9ec44953844249dad25e78cab9f4cc02752a116788c551a6abcd178cb6c62`。最终校验记录为 `verification.json`，源码/配置身份为 `implementation.json` 和 `defaults.json`。本轮没有更改电子kernel、出生数据或物理采样算法；没有新增待验收的性能候选。

后续出生选区诊断、欧氏邻域和按解剖BODY区分体外空气的研究开关见 [电子出生选区优化](broadbeam_electron_selection_20261004.md)。这些开关默认关闭或采用原box设置，不自动改变上述四预设；本文二进制SHA是当时验证快照，后续二进制身份以该报告为准。
