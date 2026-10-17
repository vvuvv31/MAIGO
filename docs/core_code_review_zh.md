# 核心 C++ 人工审阅导读

本文描述当前实现的入口、状态交接和需要保持的约束。源码中的 `.inc` 是实际参与
编译的 C++ 实现片段；历史实验、离线提取器和独立检查不属于运行时输运流程。

## 建议阅读顺序

1. `src/main.cpp`、`src/cli.cpp`：命令行入口和运行方式。
2. `src/config.cpp`、`include/carbon/transport_config.hpp`：输入约束、默认值和派生配置。
3. `src/batch_run.cpp`、`src/run_session.cpp`：计划权重、批次划分和上下文复用。
4. `src/transport_sycl.cpp`、`src/pipelines/`：管线选择及其编译期策略。
5. `src/detail/` 中的 setup → primary → secondary → finalize：输运、计分和审计。
6. `src/io.cpp`、`src/run_quality.cpp`：剂量单位转换、输出及运行质量判断。

## `.inc` 的包含上下文与职责

这些文件依赖父编译单元已有的声明，不能当作独立 `.cpp` 编译。尤其是四个
`sycl_pipeline_*.inc`，它们按顺序包含在同一个 `transport_pipeline<Pipeline>()`
函数体内，共享局部变量；调整包含顺序需要重新核对状态和资源生命周期。

| 实现片段 | 包含上下文 | 职责及交接 |
|---|---|---|
| `sycl_transport_context_impl.inc`、`sycl_transport_context_methods.inc` | `namespace carbon` | 保存和管理跨批次复用的 GPU 上下文、包缓存 |
| `sycl_dose_atomic.inc`、`sycl_ion_ledger_device.inc` | `namespace carbon` | 计分类型和设备端能量账目辅助函数 |
| `sycl_copper_primary_task.inc`、`sycl_device_math.inc`、`sycl_urban_global.inc` | 匿名命名空间 | 铜任务类型、物理与几何函数、Urban 步进 |
| `sycl_pipeline_setup.inc` | 输运函数体 | 配置检查、包视图、队列、上传及缓冲区分配 |
| `sycl_pipeline_primary.inc` | setup 之后 | 主粒子输运、束线/介质交接、铜子任务和介质内带电次级出生 |
| `sycl_primary_beamline.inc` | 主粒子设备内核 | 铜/空气输运；产生下游介质入口状态 |
| `sycl_copper_primary_child.inc`、`sycl_copper_ion_fragment_step.inc` | 铜子粒子处理调用点 | 铜反应产物输运；同时服务内联处理与排队任务 |
| `sycl_pipeline_secondary.inc` | primary 之后 | 按代处理带电次级及其新产生的后代 |
| `sycl_pipeline_finalize.inc` | secondary 之后 | 读回计分、汇总能量账目、完成与覆盖审计 |

`SyclTransportContext` 可跨批次持有不可变包；`DeviceMemoryTracker` 管理本次调用的
临时设备分配。设备视图和插值点不拥有包存储。审阅异步上传时，要同时检查宿主源
缓冲区何时释放、设备目标何时可读，以及后续提交是否依赖既有队列顺序。

## 主粒子阶段与暂停/恢复

阶段名称及整数身份集中在 `src/pipelines/primary_stage.hpp`。整数保留原值，
仍用于 `PrimaryTransportKernel` 模板实例化。它们不属于配置文件或物理过程编号。

- `beamline` / `beamline_queued_children`：从源推进铜/空气；后者把铜子粒子工作排队。
- `medium`：从入口记录开始推进水或 CT。
- `full_transport` / `full_transport_nonion`：通用完整输运体及其非离子涨落策略。
- `proton_ct` / `carbon_ct`：CT 宽束专用路径。
- `carbon_ct_shared_cell`：材料和计分网格严格重合时复用体素查询。
- `minibeam_ct_begin` / `minibeam_ct_resume`：CT 微束介质段的首次启动与恢复。
- `proton_ct_shared_cell_begin` / `proton_ct_shared_cell_resume`：质子 CT 宽束对应路径。

`kKernelStage` 区分编译内核和调度策略，`kPrimaryStage` 选择它们复用的物理主体。
检查专用路径时，应先确认宿主 eligibility 条件，再审阅编译期替换是否符合该条件。

暂停点必须在完整物理迭代之间。恢复时需要保留位置、能量、方向、步数、随机数
身份、核光学深度、离化时钟、Urban 状态和尚未提交的计分。能量/物种排序只改变
执行索引，不能重置这些状态，也不能把 GPU 队列槽号作为随机数身份。

## 物理计算的单位与约定

| 量 | 当前约定 |
|---|---|
| 位置、射程和路径长度 | mm；Urban 的真实路径与几何位移路径需分别检查 |
| 轨迹动能 | 总 MeV；查询需要 MeV/u 的表时显式除以质量数 A |
| 密度 | g/cm³ |
| 宏观反应率 | mm⁻¹；质量归一化反应率乘以密度后使用 |
| 光学深度 | 无量纲，沿路径累计 `Σ × length` |
| GPU 剂量计分缓冲区 | 实际累计沉积能量 MeV，原子累加为 FP32 |
| 输出剂量 | Gy；宿主端由能量和介质质量换算 |

质量应遵循各包的显式约定：保留 C12 的 A×amu 校准运动学，不把它统一替换成
核质量。`particle.hpp` 和包加载检查分别说明、核对相应约定。

介质核反应使用共享光学深度：先以弹性与非弹性总危险率定位候选，再按各自危险率
选择过程。跨材料和暂停恢复时延续剩余光学深度；只有新时钟才重新抽样。
`HadronicRateCache` 的非弹性候选率及 post-EM 接受检查是另一层逻辑，不能把弹性
危险率混入其接受分母。材料/密度变化也要按调用点要求使率缓存失效。

`unified_em_view.hpp` 负责 continuous26 的插值、均值、涨落与聚合 delta 能损。
当前不显式输运电子/光子，介质内电离能损局部沉积；受限损失与 delta 能量项仍须
完整保留。铜使用 `copper_fluctuation.hpp` 的统一引擎，但主质子与原生离子碎片
保留独立标定和参数。关闭受限涨落不等于关闭原生铜离子的 hard-delta 抽样。

Urban 的 Geant4 11.3.2 对应函数见 `sycl_device_math.inc` 开头的模型说明；铜涨落
的参数与验证范围另见 `docs/copper_fluctuation_engine.md`。有历史验收结果不代表
新增配置自动具有相同精度，具体结果应查对应的完整计划报告。

## 计分、批次和能量账目

`src/batch_run.cpp` 先对整个 campaign 的权重做一次最大余数分配，再将每个 spot
的粒子分到各批次。分别检查每批粒子数、每个 spot 的总粒子数，以及 seed 范围。
此过程不以进入患者的粒子数替代原始入射粒子数。

`src/io.cpp` 按 `dose_output_scale` 应用输出归一化，并使用
`dose = deposited_MeV × 1.602176634e-13 / mass_kg`。CT 按各体素介质密度计算
质量；不能用水密度替代。批次数不是额外的剂量乘数，跨批次结果的合并需遵守各自
输出尺度与粒子分配，而不能仅因运行了 N 批就把计划剂量乘 N。

审阅局部沉积和终止分支时，应追踪能量最终进入哪个互斥去向：计分、仍在输运的
带电粒子、逃逸、束线移除或未跟踪产物能量。核反应账目的细分项是审计信息，
不能再次加到剂量中。中性产物保持明确的未跟踪能量汇；铜 delta/反冲能量保持
束线移除约定。覆盖失败、无进展、未完成轨迹或队列溢出都不能作为有效剂量接受。

## 修改后的检查边界

纯排版或命名整理首先确认有效 token、数值常量、表达式顺序和模板实例化身份未变，
再编译和运行对应的实现回归。GPU FP32 原子加法存在执行顺序差异，不应要求不同
调度运行的整个剂量数组逐位相同。

`scripts/validate_presets.py` 是匹配源、随机种子、剂量、计数和账目的 GPU 实现检查。
它不替代 TOPAS gamma、物理模型验证或吞吐测试。独立数学检查对应几何、随机流、
共享光学深度和主质子快速路径。GPU 构建必须显式指定目标架构并生成 AOT，
执行时设置 `CUDA_DISABLE_PTX_JIT=1`；具体运行命令见主 README。
