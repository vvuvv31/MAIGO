#!/usr/bin/env python3
"""Summarize repeated NCU stage samples without averaging unlike kernels."""
import argparse
import json
from pathlib import Path
import statistics


METRICS = {
    "registers_per_thread": "launch__registers_per_thread",
    "block_size": "launch__block_size",
    "occupancy_limit_register_blocks": "launch__occupancy_limit_registers",
    "active_warps_per_scheduler": "smsp__warps_active.avg.per_cycle_active",
    "eligible_warps_per_scheduler": "smsp__warps_eligible.avg.per_cycle_active",
    "long_scoreboard_cycles_per_issue":
        "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "active_lanes_per_warp": "smsp__thread_inst_executed_per_inst_executed.ratio",
    "predicated_lanes_per_warp":
        "smsp__thread_inst_executed_pred_on_per_inst_executed.ratio",
    "l1_hit_percent": "l1tex__t_sector_hit_rate.pct",
    "l2_hit_percent": "lts__t_sector_hit_rate.pct",
}


def number(value):
    return float(value.replace(",", ""))


def summarize(path):
    rows = json.loads((path / "results.json").read_text())
    grouped = {}
    for row in rows:
        grouped.setdefault(row["stage"], []).append(row)
    result = {}
    for stage, samples in grouped.items():
        item = {"samples": len(samples), "identity": samples[0]["identity"]}
        for label, metric in METRICS.items():
            values = [number(x["metrics"][metric]) for x in samples
                      if x["metrics"].get(metric) not in (None, "")]
            if values:
                item[label] = {
                    "median": statistics.median(values),
                    "minimum": min(values),
                    "maximum": max(values),
                }
        # Local-memory transactions are evidence of spills; stack size alone is not.
        local = {k: [number(x["metrics"][k]) for x in samples]
                 for k in samples[0]["metrics"]
                 if "local" in k and ("sector" in k or "transaction" in k)
                 and all(x["metrics"].get(k) not in (None, "") for x in samples)}
        item["local_memory_metrics"] = {
            k: {"median": statistics.median(v), "minimum": min(v), "maximum": max(v)}
            for k, v in local.items()
        }
        item["kernel_names"] = sorted({x["metrics"]["Kernel Name"] for x in samples})
        result[stage] = item
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    args = parser.parse_args()
    result = summarize(args.profile)
    (args.profile / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
