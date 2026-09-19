#!/usr/bin/env python3
"""Run an all-ion elastic campaign sequentially and freeze input provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topas", required=True, type=Path)
    parser.add_argument("--campaign", required=True, type=Path)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    topas = args.topas.resolve()
    metadata = json.loads((campaign / "campaign.json").read_text())
    (campaign / "raw").mkdir(exist_ok=True)
    (campaign / "logs").mkdir(exist_ok=True)
    pins: dict[str, str] = {}
    for name in (campaign / "cases.txt").read_text().splitlines():
        if not name:
            continue
        case = campaign / name
        log = campaign / "logs" / f"{case.stem}.log"
        with log.open("w", encoding="utf-8") as stream:
            subprocess.run([str(topas), str(case)], cwd=campaign,
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
        pins[str(case)] = digest(case)
    pins[str(topas)] = digest(topas)
    pins[str(REPO / "extensions/topas/all_ion_elastic/AllIonElasticDump.cc")] = digest(
        REPO / "extensions/topas/all_ion_elastic/AllIonElasticDump.cc")
    pins[str(REPO / "extensions/topas/all_ion_elastic/AllIonElasticDump.hh")] = digest(
        REPO / "extensions/topas/all_ion_elastic/AllIonElasticDump.hh")
    hu_file = Path(metadata["hu_file"])
    pins[str(hu_file)] = digest(hu_file)
    for path in sorted(Path(metadata["dicom_dir"]).rglob("*")):
        if path.is_file():
            pins[str(path)] = digest(path)
    (campaign / "provenance.json").write_text(
        json.dumps(pins, indent=2) + "\n", encoding="utf-8")
    print(f"completed {len((campaign / 'cases.txt').read_text().splitlines())} cases")


if __name__ == "__main__":
    main()
