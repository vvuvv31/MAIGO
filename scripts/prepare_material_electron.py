#!/usr/bin/env python3
"""Build immutable, exact decoded CT electron runtime artifacts, one table at a time."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fingerprint():
    files = (ROOT / "scripts/material_prepared_sources.txt").read_text().splitlines()
    text = "".join(f"{name}:{hashlib.sha256((ROOT / name).read_bytes()).hexdigest()}\n" for name in files)
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--packed", action="store_true", help="Proton EM-only graph plus replay certificates")
    mode.add_argument("--carbon-compact", action="store_true", help="C12 exact shared-state and crossing-bin artifact")
    p.add_argument("--begin", type=int, default=0)
    p.add_argument("--end", type=int)
    args = p.parse_args()
    index = args.index.resolve()
    doc = json.loads(index.read_text())
    version = fingerprint()
    build = ROOT / "out/material_prepared_tools" / version
    build.mkdir(parents=True, exist_ok=True)
    exe = build / "prepare"
    if not exe.exists():
        subprocess.run(["g++", "-O3", "-std=c++20", "-DCARBON_OPENSSL_SHA256=1",
                        f'-DCARBON_MATERIAL_PREPARED_VERSION="{version}"', "-I", str(ROOT / "include"),
                        str(ROOT / "scripts/prepare_material_electron.cpp"),
                        str(ROOT / "src/water_electron_response.cpp"), "-lcrypto", "-lzstd", "-pthread", "-o", str(exe)], check=True)
    subprocess.run([str(exe), str(index), hashlib.sha256(index.read_bytes()).hexdigest(), str(args.cache.resolve()),
                    *map(str, doc["projectile"]), str(args.begin), str(args.end if args.end is not None else len(doc["tables"])), *(["carbon_compact"] if args.carbon_compact else ["packed"] if args.packed else [])], check=True)


if __name__ == "__main__":
    main()
