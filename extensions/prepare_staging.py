#!/usr/bin/env python3
"""Prepare a separate release-asset staging root; never overwrite installed assets."""
import argparse
from pathlib import Path
import shutil

KIT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for folder in ('nuclear/schneider/proton', 'nuclear/schneider/carbon', 'nuclear/schneider/secondary',
                   'elastic/schneider/proton', 'elastic/schneider/carbon', 'materials/schneider',
                   'stopping/schneider', 'em/ions', 'scattering/urban', 'ionization/air'):
        (out/'data'/folder).mkdir(parents=True)
    shutil.copyfile(KIT/'reference/hu_to_material.txt', out/'data/materials/schneider/hu_to_material.txt')
    shutil.copyfile(KIT/'reference/carbon_bundle.json', out/'data/nuclear/schneider/carbon/bundle.json')
    (out/'docs').mkdir()
    print(out)


if __name__ == '__main__':
    main()
