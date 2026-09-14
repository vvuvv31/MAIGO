"""Summarize non-overlapping host/kernel stages and diagnostic lane cycles."""
from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
data = json.loads((ROOT / "results.json").read_text())
base, prof = data["runs"]
s = base["status"]
h = base["host_scopes_seconds"]
parts = {
    "Input/source preparation": h["input_and_source_preparation"],
    "Transport setup (includes uploads)": h["transport_setup_including_upload"],
    "Primary kernels": s["primary_s"],
    "Secondary kernels": s["secondary_s"],
    "Loop host/synchronization/other GPU work": h["transport_loop_including_scoring_and_queue_transfers"] - s["primary_s"] - s["secondary_s"],
    "Readback/host finalization": h["transport_readback_and_host_finalize"],
    "Quality/output": h["quality_and_output"],
}
parts["Other process time"] = s["wall_s"] - sum(parts.values())
if any(v < -0.01 for v in parts.values()):
    raise RuntimeError("Host scope overlap; inspect raw scopes before reporting")
lines = ["# RT07575 runtime breakdown — 2026-09-14", "",
         "当前工作树隔离快照；本地 RTX 2080 Ti / sm_75，100 万原发，完整次级统一 EM、种类分组、原生涨落，生产配置不含独立核弹性。history_chunk_size=34816。", "",
         "源码和配置见 `scratch/runtime_breakdown_20260914`，原始运行见 `scratch/unified_em_perf_20260913/runtime_current_1m_*`。此处不是之前含弹性的 1M 实验。", "",
         f"未插桩基准：{s['throughput']:.1f} histories/s（程序 Elapsed={s['elapsed_s']:.3f} s）；进程墙钟 {s['wall_s']:.3f} s。", "",
         "| 不重叠阶段 | 秒 | 完整进程占比 |", "|---|---:|---:|"]
for name, value in parts.items():
    lines.append(f"| {name} | {value:.4f} | {100*value/s['wall_s']:.2f}% |")
lines += ["", "下列 CUDA 传输时间来自 Nsight 采集，是上述阶段的内部活动，不能再次相加：", "",
          "| CUDA transfer | 秒 | bytes | calls |", "|---|---:|---:|---:|"]
for c in prof["cuda_copies"]:
    lines.append(f"| {c['direction']} | {c['seconds']:.6f} | {c['bytes']} | {c['count']} |")
lines += ["", f"Nsight 程序 Elapsed 增加 {100*data['profiler_elapsed_overhead_fraction']:.2f}%；输入/二进制 SHA、步数和 EM audit 已核对。", "",
          "CT load/preprocess、包加载等嵌套 wall scopes 保存在 results.json，不将其与父 scope 重复求和。此配置读取 packed CT，不包含从原始 DICOM 重新建网格的全流程。没有独立 WET map kernel，CT 材料查询/边界推进在输运内；剂量累加也融合在输运 kernel 中。", ""]
fine = REPO / "scratch/unified_em_perf_20260913/runtime_current_1m_phase_fine/RT07575"
diagnostic = fine if (fine / "status.json").exists() else REPO / "scratch/unified_em_perf_20260913/runtime_current_1m_phase/RT07575"
if (diagnostic / "status.json").exists():
    ds = json.loads((diagnostic / "status.json").read_text())
    cycles = {int(i): (int(c), int(n)) for i,c,n in re.findall(
        r"\[phase-cycles\] (\d+) (\d+) (\d+)", (diagnostic / "gpu.log").read_text())}
    if not ds["complete"] or ds["audit"] != s["audit"] or ds["steps"] != s["steps"]:
        raise RuntimeError("Diagnostic physics audit mismatch")
    if not cycles or not sum(c for c,n in cycles.values()):
        raise RuntimeError("Device clocks not collected")
    lines += ["## Kernel 内采样诊断", "",
              f"诊断构建 Elapsed={ds['elapsed_s']:.3f} s，较未插桩基准变化 {100*(ds['elapsed_s']/s['elapsed_s']-1):.1f}%。每约 2048 个粒子步抽一个 lane，记录 SM clock 周期；不消耗物理 RNG。", "",
              "以下是抽样 lane elapsed cycles 比例，包含 warp 分歧、等待和插桩影响，不是各物理过程的独立 GPU wall seconds；嵌套 EM 子项不与外层相加。", ""]
    groups = [("Primary", [0,1,2,3,4,5], ["Preparation: CT/geometry/EM tables/nuclear rates", "Unified EM loss", "Scoring/electron-response branches", "MCS", "Advance/face handling", "Nuclear resolution/loop bookkeeping"]),
              ("Secondary", [8,9,10,11,12,13], ["Preparation: CT/geometry/EM tables/nuclear rates", "Unified EM loss", "Post-EM geometry/scoring", "MCS", "Inelastic replay/scoring branches", "Elastic/loop bookkeeping"]),
              ("Primary EM nested", [16,17,18], ["Mean loss", "Fluctuations", "Delta clock/acceptance/spectrum"]),
              ("Secondary EM nested", [20,21,22], ["Mean loss", "Fluctuations", "Delta clock/acceptance/spectrum"])]
    if diagnostic == fine:
        groups[0] = ("Primary", [0,6,7,1,2,3,4,5], ["CT/material lookup and initial step state", "EM table/range/rate preparation", "Geometry clamps/nuclear rates/pre-loss work", "Unified EM loss", "Scoring/electron-response branches", "MCS", "Advance/face handling", "Nuclear resolution/loop bookkeeping"])
        groups[1] = ("Secondary", [8,14,15,9,10,11,12,13], ["CT/material/stopping lookup and initial state", "EM table/range/rate preparation", "Geometry clamps/nuclear rates/pre-loss work", "Unified EM loss", "Post-EM geometry/scoring", "MCS", "Inelastic replay/scoring branches", "Elastic/loop bookkeeping"])
    for name, slots, labels in groups:
        total = sum(cycles.get(i,(0,0))[0] for i in slots)
        lines += [f"### {name}", "", "| Region | sampled cycle share | intervals |", "|---|---:|---:|"]
        for i,label in zip(slots,labels):
            c,n = cycles.get(i,(0,0))
            lines.append(f"| {label} | {100*c/total if total else 0:.2f}% | {n} |")
        lines.append("")
    (ROOT / "phase_cycles.json").write_text(json.dumps(cycles,indent=2)+"\n")
else:
    lines += ["Kernel 内各物理过程尚无独立实测时长，不以源代码行数或调用次数替代耗时比例。", ""]
(ROOT / "README.md").write_text("\n".join(lines)+"\n")
(ROOT / "disjoint_stages.json").write_text(json.dumps(parts,indent=2)+"\n")
print("\n".join(lines))
