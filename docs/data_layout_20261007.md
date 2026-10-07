# Release 数据目录整理（2026-10-07）

数据按物理模块、材料和粒子分类。日期和生成器/二进制格式版本保留在 metadata 中；文件路径表达用途。物理模型、二进制格式、数值内容和八个预设的非路径参数不变。

## 当前布局

| 目录 | 内容 |
|---|---|
| `data/materials/{water,schneider}` | 水材料描述、HU 映射、材料辐射长度 |
| `data/em/ions` | continuous26 离子 EM、delta moments、生成辅助 composition bases |
| `data/em/electrons/{shared,proton,carbon}` | strict26 共享电子/光子与入射离子参数 |
| `data/nuclear/schneider/{proton,carbon,secondary}` | 主/次级核反应率、final states、bundle 和 sidecars |
| `data/nuclear/copper/{proton,carbon}` | 铜中的主粒子反应率与级联；共享离子截面在上层 |
| `data/elastic/{schneider,copper}` | 各粒子校准的弹性散射、反冲 stopping |
| `data/stopping/{water,air,copper,schneider}` | 各材料的参考 stopping 表及辅助提取表 |
| `data/scattering/urban` | CT/铜与空气的 Urban 包 |
| `data/ionization/air` | 质子空气电离率 |
| `inputs/rt07575_head/{proton,carbon}` | 独立的两套头部 CT、beam model、spots |
| `inputs/lung_minibeam` | 两个 CT minibeam 共用的独立肺部 benchmark |
| `inputs/water_minibeam/carbon` | 碳水 minibeam 的 TPS spots |

质子和碳离子的铜 fragment stopping 表分别为 `ions_proton.csv`、`ions_carbon.csv`，没有合并。电子包内部 operator 文件名、package ID 和参数硬链接保持原样；这些硬链接属于当前包结构。未创建旧目录的兼容链接或旧路径查找逻辑。

149 个原始文件均有迁移记录，其中 5 个历史生成/清理记录移入 `docs/data_provenance/`；其余 144 个保留为安装资源。另外新增 `data/manifest.json`。辅助生成资源仍被保留，manifest 不宣称其每一个文件都是运行必需输入。

历史验证报告和历史绝对生成路径保留原样。当前配置、源码固定路径、核反应 companion 文件名、bundle 引用、电子 registry、校验脚本和安装清单已同步迁移。

## 发布清单

- `data/manifest.json`：安装资源的路径、字节数、SHA256 和八预设依赖列表；路径相对仓库根目录，manifest 不包含自己的 digest。
- `docs/preset_dependencies.json`：显式配置、固定默认路径、电子 payload/参数链接、bundle 成员和 companion metadata 的展开清单。
- `docs/copied_inputs.json`：当前 `data/` 和 `inputs/` 文件清单。
- `docs/data_layout_migration_20261007.json`：全部新旧路径及原始 digest。
- `docs/data_layout_before_20261007.json`：迁移前安装资源与冻结对照二进制的 digest。

发布校验：

```bash
python3 scripts/check_data_manifest.py
python3 scripts/check_electron_packages.py
```

只有有意修改数据之后才使用 `check_data_manifest.py --write` 重建清单。运行时不增加每批次 digest 检查。

## 验证

构建成功，既有编译警告仍在。82 项退休接口检查、15 项输入契约检查、4 项核反应包契约检查及电子包检查通过。

111 个非 JSON 文件（包含二进制、CSV、HU 文本、NPZ）与迁移前 SHA256 完全相同；38 个 JSON 的数值内容保持一致。八个 YAML 的全部非路径字段保持一致。运行源码、配置和脚本无旧资源路径残留。详见 `data_layout_integrity_20261007.json`。

冻结对照位于 `/mnt/sdb/wuwei/MAIGO_data_layout_before_20261007/`，拥有迁移前的目录、配置、源码和二进制。匹配 GPU 回归结果记录在 `validation_data_layout_20261007/summary.json`；每个预设每个臂连续 2 批，每批 10,000 histories。

八个预设均通过匹配 GPU 回归，源 CSV/种子和离散审计计数一致，全部 16 个候选批次审计通过且队列零溢出。最大剂量相对 L1 差异为 `2.79985540757e-07`（阈值 `1e-5`），最大物理相对能量残差为 `7.26634417886e-06`（阈值 `1e-4`）。

该检查只验证实现和迁移的一致性，不是性能测量或新的 TOPAS 剂量验收；碳 minibeam 剂量精度验收仍待完成。
