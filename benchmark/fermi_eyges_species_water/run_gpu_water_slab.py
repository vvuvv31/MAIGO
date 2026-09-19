#!/usr/bin/env python3
"""Run and quality-gate the prepared FE species water replays."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess


DEFAULT_RUN_ROOT = Path(
    "/mnt/sda/wuwei/fe_species_water_em10gev_20260919/gpu_unified_strag_endpoint")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-root", type=Path, default=DEFAULT_RUN_ROOT)
    args = parser.parse_args()
    manifest = json.loads((args.gpu_root / "manifest.json").read_text())
    binary = manifest["binary"]
    env = dict(os.environ)
    env["ONEAPI_DEVICE_SELECTOR"] = "cuda:*"
    env["LD_LIBRARY_PATH"] = (
        "/home/wuwei/sycl_workspace/llvm/build/install/lib:" +
        env.get("LD_LIBRARY_PATH", ""))
    for case in manifest["cases"]:
        directory = Path(case["config"]).parent
        log_path = directory / "gpu.log"
        with log_path.open("w") as log:
            result = subprocess.run(
                [binary, "--config", case["config"], "--device", "cuda"],
                cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT)
        text = log_path.read_text()
        quality_path = directory / "out/gpu/quality_report.json"
        quality = json.loads(quality_path.read_text()) if quality_path.exists() else {}
        audit = [list(map(int, row.split())) for row in
                 re.findall(r"\[unified-em-audit\]([^\n]+)", text)]
        if (result.returncode != 0 or not quality.get("accepted", False) or
                quality.get("failures") or not audit or
                any(row[0] != 0 for row in audit) or
                not Path(case["plane"]).exists()):
            raise RuntimeError(f"GPU validation failed: {case['name']}")
        print(case["name"], "complete", flush=True)


if __name__ == "__main__":
    main()
