#!/usr/bin/env python3
"""Run an all-ion elastic campaign sequentially and freeze input provenance."""
from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--topas', required=True, type=Path)
    parser.add_argument('--campaign', required=True, type=Path)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    topas = args.topas.resolve()
    metadata = json.loads((campaign / 'campaign.json').read_text())
    (campaign / 'raw').mkdir(exist_ok=True)
    (campaign / 'logs').mkdir(exist_ok=True)
    sources = []
    for name in (campaign / 'cases.txt').read_text().splitlines():
        if not name:
            continue
        case = campaign / name
        log = campaign / 'logs' / f'{case.stem}.log'
        with log.open('w', encoding='utf-8') as stream:
            subprocess.run([str(topas), str(case)], cwd=campaign, stdout=stream, stderr=subprocess.STDOUT, check=True)
        sources.append(str(case))
    sources.extend([str(topas), str(REPO / 'extensions/topas/all_ion_elastic/AllIonElasticDump.cc'), str(REPO / 'extensions/topas/all_ion_elastic/AllIonElasticDump.hh')])
    hu_file = Path(metadata['hu_file'])
    sources.append(str(hu_file))
    for path in sorted(Path(metadata['dicom_dir']).rglob('*')):
        if path.is_file():
            sources.append(str(path))
    (campaign / 'provenance.json').write_text(json.dumps(sources, indent=2) + '\n', encoding='utf-8')
    print(f'completed {len((campaign / 'cases.txt').read_text().splitlines())} cases')
if __name__ == '__main__':
    main()
