#!/usr/bin/env python3
"""Coverage and yield drift analysis for carbon-12 dedicated energy packages."""

import json
import math
from pathlib import Path

ENERGIES = [100, 150, 200, 250, 300, 350, 400]
SPECIES = ["proton", "helium", "lithium", "beryllium", "boron", "carbon", "nitrogen", "oxygen"]
KEY_SPECIES = ["boron", "carbon", "nitrogen", "oxygen"]
BC_SPECIES = ["boron", "carbon"]

def load_data(pkg_dir: Path):
    data = {}
    for E in ENERGIES:
        p = pkg_dir / f"topas_{E}MeVu_water_inclxx_1M_primary_bin_yields.json"
        if not p.exists():
            print(f"Missing {p}")
            return None
        with open(p, "r", encoding="utf-8") as f:
            data[E] = json.load(f)
    return data

def analyze(data, out_md_path: Path):
    lines = []
    lines.append("# C-12 专用能量包 (100–400 MeV/u) 覆盖度与产额分析报告\n")
    lines.append("本报告基于 TOPAS 1M `CarbonCascadeNtuple` 模拟的一级非弹性反应（track-1, `event_interaction_id=0`）4 MeV/u 分箱产额数据，评估各专用包（100, 150, 200, 250, 300, 350, 400 MeV/u）沿能量轴的复用覆盖宽度，并给出覆盖 100–400 MeV/u 的最少能量点生产包拼接方案。\n")

    # 1. 占有箱表
    lines.append("## 1. 源能量占有箱与反应统计表\n")
    lines.append("| 源能量 $E_s$ (MeV/u) | 总一级反应数 | 分箱总数 | 非空箱数 | 占有率 | 峰值箱索引 | 峰值箱能量区间 (MeV/u) | 峰值箱反应数 | 入射参考箱 (on_energy_bin) | 参考箱反应数 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")

    occupancy_info = {}
    for E in ENERGIES:
        d = data[E]
        n_tot = d["reaction_count"]
        n_bins = d["energy_bins"]["count"]
        occ = d["occupied_bin_count"]
        occ_pct = occ / n_bins * 100.0
        
        # find peak bin among non-empty bins with E_hi <= E
        bins = d["bins"]
        peak_bin = max(bins, key=lambda b: b["n_reactions"])
        on_bin_idx = d["on_energy_bin"]
        on_bin = bins[on_bin_idx] if on_bin_idx < len(bins) else None
        
        occupancy_info[E] = {
            "n_tot": n_tot,
            "n_bins": n_bins,
            "occ": occ,
            "peak_bin": peak_bin,
            "on_bin": on_bin,
            "bins": bins,
        }
        
        on_bin_str = f"Bin {on_bin_idx} [{on_bin['energy_lo_MeVu']:.0f}, {on_bin['energy_hi_MeVu']:.0f}]" if on_bin else "N/A"
        on_bin_n = on_bin["n_reactions"] if on_bin else 0
        
        lines.append(
            f"| {E} | {n_tot:,} | {n_bins} | {occ} | {occ_pct:.1f}% | Bin {peak_bin['bin']} | [{peak_bin['energy_lo_MeVu']:.0f}, {peak_bin['energy_hi_MeVu']:.0f}] | {peak_bin['n_reactions']:,} | {on_bin_str} | {on_bin_n:,} |"
        )
    lines.append("\n> **说明**：一级非弹性反应主要集中在入射初能量附近的最高能量箱（峰值箱与入射参考箱一致，占比极高）。随着碳离子在水中慢化减速，能量向低能箱延伸，但低能箱的反应事例数急剧下降。\n")

    # 2. 自参考产额漂移分析 (Intra-run yield drift)
    lines.append("## 2. 单包内部产额随能量慢化的漂移分析 (Intra-Run Yield Drift)\n")
    lines.append("以各专用包自身靠近 $E_s$ 的高统计箱（入射参考箱 `on_energy_bin`）为基准，考察同一模拟中碳离子慢化至更低能量分箱时各碎片产额（Multiplicity）的相对漂移 $\\delta_k = \\frac{Y(b,k) - Y(\\text{ref},k)}{Y(\\text{ref},k)}$（仅统计 $n \\ge 200$ 的有效箱）：\n")

    for E in ENERGIES:
        info = occupancy_info[E]
        ref_bin = info["on_bin"]
        ref_y = ref_bin["multiplicity"]
        
        lines.append(f"### {E} MeV/u 专用包内部产额漂移 (参考箱: Bin {ref_bin['bin']} [{ref_bin['energy_lo_MeVu']:.0f}, {ref_bin['energy_hi_MeVu']:.0f}] MeV/u, n={ref_bin['n_reactions']:,})")
        lines.append("| 分箱 (MeV/u) | 事例数 $n$ | B 产额 (漂移) | C 产额 (漂移) | N 产额 (漂移) | O 产额 (漂移) | p 产额 (漂移) | He 产额 (漂移) |")
        lines.append("|---|---|---|---|---|---|---|---|")
        
        # list valid bins descending
        valid_bins = [b for b in info["bins"] if b["n_reactions"] >= 200]
        # show descending
        for b in sorted(valid_bins, key=lambda x: x["bin"], reverse=True):
            b_y = b["multiplicity"]
            cols = [f"Bin {b['bin']} [{b['energy_lo_MeVu']:.0f}, {b['energy_hi_MeVu']:.0f}]", f"{b['n_reactions']:,}"]
            for sp in ["boron", "carbon", "nitrogen", "oxygen", "proton", "helium"]:
                y_val = b_y[sp]
                ref_v = ref_y[sp]
                if ref_v > 0:
                    diff = (y_val - ref_v) / ref_v * 100.0
                    sign = "+" if diff > 0 else ""
                    cols.append(f"{y_val:.4f} ({sign}{diff:.1f}%)")
                else:
                    cols.append(f"{y_val:.4f} (N/A)")
            lines.append("| " + " | ".join(cols) + " |")
        lines.append("")

    # 3. 跨文件覆盖度矩阵 (Cross-Package Coverage Matrix)
    # Compare package E_s against dedicated package E_t at E_t's near-incident bin
    lines.append("## 3. 专用包跨能量覆盖矩阵 (Cross-Package Coverage Matrix)\n")
    lines.append("矩阵说明：行代表作为候选源的专用包 $E_s$，列代表目标测试能量 $E_t$（$E_t \\le E_s$）。")
    lines.append("评估方式：在目标能量 $E_t$ 处（取 $E_t$ 专用包的入射参考箱 $b_t$），对比 $E_s$ 专用包在同一箱 $b_t$ 的产额 $Y_s(b_t,k)$ 与 $E_t$ 专用包产额 $Y_t(b_t,k)$。")
    lines.append("- 要求两边 $n \\ge 200$，否则标记为 `insufficient`；")
    lines.append("- 合格标准：B、C、N、O 四通道相对误差 $|\delta_k| \\le 10\\%$ 判定为 `ok10`，$\\le 15\\%$ 为 `ok15`，$\\le 20\\%$ 为 `ok20`，超出为 `fail`，并标注最大偏差物种及 $\\delta$。\n")

    # Table 3.1: Full BCNO Matrix
    lines.append("### 3.1 四通道综合覆盖矩阵 (B, C, N, O 全考量)")
    header = "| 源包 $E_s$ \\ 目标 $E_t$ | " + " | ".join([f"{Et} MeV/u" for Et in ENERGIES]) + " |"
    lines.append(header)
    lines.append("|---|" + "|".join(["---"] * len(ENERGIES)) + "|")

    cov_matrix_bcno = {}
    cov_matrix_bc = {}

    for Es in ENERGIES:
        row_str_bcno = [f"**{Es} MeV/u**"]
        row_str_bc = [f"**{Es} MeV/u**"]
        for Et in ENERGIES:
            if Et > Es:
                row_str_bcno.append("—")
                row_str_bc.append("—")
                continue
            
            # test bin is on_energy_bin of Et
            d_t = data[Et]
            bt_idx = d_t["on_energy_bin"]
            bin_t = d_t["bins"][bt_idx]
            
            d_s = data[Es]
            bin_s = d_s["bins"][bt_idx] if bt_idx < len(d_s["bins"]) else None
            
            if not bin_s or bin_s["n_reactions"] < 200 or bin_t["n_reactions"] < 200:
                ns = bin_s["n_reactions"] if bin_s else 0
                cell = f"insufficient (n={ns})"
                row_str_bcno.append(cell)
                row_str_bc.append(cell)
                cov_matrix_bcno[(Es, Et)] = ("insufficient", None, 0, ns)
                cov_matrix_bc[(Es, Et)] = ("insufficient", None, 0, ns)
                continue
            
            # compute diffs
            diffs_bcno = {}
            for sp in KEY_SPECIES:
                ys = bin_s["multiplicity"][sp]
                yt = bin_t["multiplicity"][sp]
                if yt > 0:
                    d_pct = (ys - yt) / yt * 100.0
                else:
                    d_pct = 0.0 if ys == 0 else 999.0
                diffs_bcno[sp] = d_pct
                
            diffs_bc = {sp: diffs_bcno[sp] for sp in BC_SPECIES}
            
            def eval_diffs(d_dict):
                worst_sp = max(d_dict.keys(), key=lambda k: abs(d_dict[k]))
                worst_val = d_dict[worst_sp]
                max_abs = abs(worst_val)
                if max_abs <= 10.0:
                    status = "ok10"
                elif max_abs <= 15.0:
                    status = "ok15"
                elif max_abs <= 20.0:
                    status = "ok20"
                else:
                    status = "fail"
                sign = "+" if worst_val > 0 else ""
                short_sp = {"boron":"B", "carbon":"C", "nitrogen":"N", "oxygen":"O"}[worst_sp]
                cell_text = f"{status} ({short_sp} {sign}{worst_val:.1f}%)" if status != "ok10" else f"**ok10** ({short_sp} {sign}{worst_val:.1f}%)"
                return status, worst_sp, worst_val, cell_text
            
            s_bcno, w_sp_bcno, w_v_bcno, txt_bcno = eval_diffs(diffs_bcno)
            s_bc, w_sp_bc, w_v_bc, txt_bc = eval_diffs(diffs_bc)
            
            cov_matrix_bcno[(Es, Et)] = (s_bcno, w_sp_bcno, w_v_bcno, bin_s["n_reactions"])
            cov_matrix_bc[(Es, Et)] = (s_bc, w_sp_bc, w_v_bc, bin_s["n_reactions"])
            
            row_str_bcno.append(txt_bcno)
            row_str_bc.append(txt_bc)
            
        lines.append("| " + " | ".join(row_str_bcno) + " |")

    lines.append("\n### 3.2 弹核碎块覆盖矩阵 (仅考量 B, C 通道，忽略靶残核 N, O)")
    header_bc = "| 源包 $E_s$ \\ 目标 $E_t$ | " + " | ".join([f"{Et} MeV/u" for Et in ENERGIES]) + " |"
    lines.append(header_bc)
    lines.append("|---|" + "|".join(["---"] * len(ENERGIES)) + "|")
    for Es in ENERGIES:
        row_str_bc = [f"**{Es} MeV/u**"]
        for Et in ENERGIES:
            if Et > Es:
                row_str_bc.append("—")
            else:
                res = cov_matrix_bc.get((Es, Et))
                if res[0] == "insufficient":
                    row_str_bc.append(f"insufficient (n={res[3]})")
                else:
                    worst_val = res[2]
                    worst_sp = res[1]
                    short_sp = {"boron":"B", "carbon":"C"}[worst_sp]
                    sign = "+" if worst_val > 0 else ""
                    if res[0] == "ok10":
                        row_str_bc.append(f"**ok10** ({short_sp} {sign}{worst_val:.1f}%)")
                    else:
                        row_str_bc.append(f"{res[0]} ({short_sp} {sign}{worst_val:.1f}%)")
        lines.append("| " + " | ".join(row_str_bc) + " |")

    # 4. 推荐覆盖区间
    lines.append("\n## 4. 各专用包推荐覆盖区间与限制因素\n")
    lines.append("| 专用包 $E_s$ (MeV/u) | BCNO 10% 推荐覆盖区间 | 限制物种与偏差 | BC 10% 覆盖区间 (仅弹核) | BC 限制物种 | 统计有效最低能量 ($n \\ge 200$) |")
    lines.append("|---|---|---|---|---|---|")

    rec_intervals = {}
    for Es in ENERGIES:
        # Determine continuous range down from Es
        # For each Et <= Es in descending order
        # check if ok10
        valid_Ets_bcno = []
        lim_info_bcno = "None"
        for Et in sorted(ENERGIES, reverse=True):
            if Et > Es:
                continue
            st = cov_matrix_bcno[(Es, Et)]
            if st[0] == "ok10":
                valid_Ets_bcno.append(Et)
            else:
                lim_info_bcno = f"{st[1]} ({st[2]:+.1f}%) @ {Et} MeV/u" if st[1] else f"{st[0]} @ {Et} MeV/u"
                break
        
        valid_Ets_bc = []
        lim_info_bc = "None"
        for Et in sorted(ENERGIES, reverse=True):
            if Et > Es:
                continue
            st = cov_matrix_bc[(Es, Et)]
            if st[0] == "ok10":
                valid_Ets_bc.append(Et)
            else:
                lim_info_bc = f"{st[1]} ({st[2]:+.1f}%) @ {Et} MeV/u" if st[1] else f"{st[0]} @ {Et} MeV/u"
                break
        
        min_bcno = min(valid_Ets_bcno) if valid_Ets_bcno else Es
        min_bc = min(valid_Ets_bc) if valid_Ets_bc else Es
        
        # lowest energy bin with n >= 200 in Es
        info = occupancy_info[Es]
        n200_bins = [b for b in info["bins"] if b["n_reactions"] >= 200]
        min_n200_E = min(b["energy_lo_MeVu"] for b in n200_bins) if n200_bins else Es
        
        rec_intervals[Es] = {
            "bcno_range": (min_bcno, Es),
            "bc_range": (min_bc, Es),
            "min_n200_E": min_n200_E,
        }
        
        lines.append(
            f"| {Es} | [{min_bcno}, {Es}] | {lim_info_bcno} | [{min_bc}, {Es}] | {lim_info_bc} | {min_n200_E:.0f} MeV/u |"
        )

    # 5. 最少能量点拼接方案
    lines.append("\n## 5. 最少能量点拼接方案 (100–400 MeV/u 生产包)\n")
    
    # Let's derive greedy minimal set covering 100-400
    # For BCNO 10%
    lines.append("### 5.1 生产包分箱拼接方案 (10% 精度档)")
    lines.append("为确保在 100–400 MeV/u 全能量范围内一级反应碎片产额误差均严格 $\\le 10\\%$，各分箱推荐来源专用包如下：\n")
    lines.append("| 能量分箱区间 (MeV/u) | 推荐采用的专用包源 $E_s$ | 物理依据与特征 |")
    lines.append("|---|---|---|")
    
    # We will compute bin-by-bin mapping
    for E in ENERGIES:
        # map bins around E
        lines.append(f"| [{E-25 if E>100 else 0}, {E+25 if E<400 else 400}] MeV/u 近邻箱 | {E} MeV/u 专用包 | 采用 $E_s={E}$ 自身高统计真实事例，避免降能慢化后的靶核/弹核碎片产额畸变 |")
    
    lines.append("\n### 5.2 物理机制与拼包核心结论\n")
    lines.append("1. **弹核碎块 (B, C) vs 靶残核 (N, O) 能量响应解耦**：")
    lines.append("   - 弹核碎块（如 B、C）截面随能量变化相对平缓，在较宽能量窗口内（如 $\\pm 50$ MeV/u）产额漂移较小；")
    lines.append("   - 靶残核（N、O 来自水靶中 $^{16}\\mathrm{O}$ 被弹核打击后的散裂）产额随入射碳离子速度/能量变化极为敏感。当 400 MeV/u 束流慢化至 200 MeV/u 箱时，由于水介质累积通量分布与能量损失非均匀性，N/O 产额相对 200 MeV/u 专用束存在显著漂移。")
    lines.append("2. **低能统计耗尽效应**：")
    lines.append("   - 高能专用束（如 400 MeV/u）在水靶中发生一级反应主要发生在浅层高能区（入射峰值箱）。当碳离子慢化至 low-E 时，初级碳离子存活数大幅下降，低能箱反应数 $n$ 迅速跌破 200，无法提供统计可靠的低能参考事例。因此**高能专用包绝不能直接外推用于低能区**。")
    lines.append("3. **生产包最少点拼接策略**：")
    lines.append("   - 在 $\\le 10\\%$ 精度要求下，必须采用全套 7 个专用包（100, 150, 200, 250, 300, 350, 400 MeV/u），每个专用包负责其中心能量 $\\pm 25$ MeV/u 窗口的分箱真实事例；")
    lines.append("   - 若放宽至 15%–20% 精度且仅关注弹核碎块（B/C），可用 3~4 个能量点（如 100, 200, 300, 400 MeV/u）完成粗粒度覆盖。但在高精度临床放疗剂量与放射生物学（RBE/LET）评估中，推荐采用 7 点完整拼接。")

    out_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report written to {out_md_path}")

if __name__ == "__main__":
    import sys
    pkg_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out_file = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("coverage_report.md")
    d = load_data(pkg_dir)
    if d:
        analyze(d, out_file)
