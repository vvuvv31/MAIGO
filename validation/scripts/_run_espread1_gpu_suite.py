#!/usr/bin/env python3
"""Run multi-energy 1% espread GPU suite under a hard GPU memory guard (50%).

Kills carbon_mc immediately if Windows reports GPU dedicated usage above 50% of
device memory, to avoid Arc driver lockups / host freezes.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXE = ROOT / "build/oneapi-windows-release-grok/carbon_mc.exe"
OUT = ROOT / "out/multi_energy_espread1"
RES = ROOT / "validation/results/espread1"
GUARD = ROOT / "validation/scripts/gpu_memory_guard.py"
ENERGIES = [100, 150, 200, 250, 300, 350, 400]
# Arc B580 global mem from sycl (~11869 MiB); override with env CARBON_GPU_MIB
DEVICE_MIB = float(os.environ.get("CARBON_GPU_MIB", "11869"))
MAX_FRACTION = float(os.environ.get("CARBON_GPU_MAX_FRACTION", "0.50"))


def kill_leftovers() -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/IM", "carbon_mc.exe", "/T"],
            capture_output=True,
            check=False,
        )
        time.sleep(1.5)


def run_one(e: int) -> None:
    cfg = ROOT / f"config/beam_{e}MeVu_multi_energy_espread1.yaml"
    kill_leftovers()
    print(f"=== GPU espread1 e={e} (VRAM guard {MAX_FRACTION*100:.0f}%) ===", flush=True)
    cmd = [
        sys.executable,
        "-u",
        str(GUARD),
        "--max-fraction",
        str(MAX_FRACTION),
        "--device-mib",
        str(DEVICE_MIB),
        "--poll-s",
        "0.4",
        "--log-every-s",
        "3",
        "--",
        str(EXE),
        "--config",
        str(cfg),
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), check=False)
    if proc.returncode == 3:
        raise SystemExit(
            f"KILLED e={e}: GPU memory exceeded {MAX_FRACTION*100:.0f}% hard limit"
        )
    if proc.returncode != 0:
        raise SystemExit(f"carbon_mc failed for e={e} rc={proc.returncode}")
    src = OUT / f"e{e}_idd.csv"
    # Keep legacy filename for compare_espread1_suite defaults.
    dst = RES / f"gpu_e{e}_idd_100k.csv"
    RES.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    # Also write history-tagged copy when N != 100k.
    try:
        # Best-effort: parse from config is not needed; suite uses 200k now.
        dst2 = RES / f"gpu_e{e}_idd.csv"
        shutil.copy2(src, dst2)
    except OSError:
        pass
    print(f"copied {dst}", flush=True)
    kill_leftovers()


def main() -> int:
    if not EXE.is_file():
        raise SystemExit(f"missing {EXE}")
    if not GUARD.is_file():
        raise SystemExit(f"missing {GUARD}")
    only = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else ENERGIES
    for e in only:
        run_one(e)
        time.sleep(1.0)
    print("=== compare ===", flush=True)
    return subprocess.call(
        [sys.executable, "-u", str(ROOT / "validation/scripts/compare_espread1_suite.py")],
        cwd=str(ROOT),
    )


if __name__ == "__main__":
    raise SystemExit(main())
