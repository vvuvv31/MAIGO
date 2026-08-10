#!/usr/bin/env python3
"""Run CarbonGPU SOBP, tee its log, and record end-to-end wall time."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()

    args.log.parent.mkdir(parents=True, exist_ok=True)
    command = [str(args.executable), "--config", str(args.config), "--device", "gpu"]
    start = time.perf_counter()
    with args.log.open("w", encoding="utf-8", newline="\n") as log:
        process = subprocess.Popen(
            command,
            cwd=args.project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
        elapsed = time.perf_counter() - start
        timing = f"WallElapsedSeconds={elapsed:.6f}\n"
        print(timing, end="")
        log.write(timing)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
