#!/usr/bin/env python3
"""Convert audited ASCII material ntuples into the current release table layout."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import numpy as np


def save_table(path, header, values):
    if not np.isfinite(values).all():
        raise ValueError(f'Nonfinite table: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    np.savetxt(path, values, delimiter=',', header=header, comments='', fmt='%.17g')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True, help='Staged root containing data/')
    parser.add_argument('--delta-reference', type=Path, required=True,
                        help='Explicit calibrated water ions CSV; preserves the measured delta correction')
    args = parser.parse_args()
    out = args.output_root.resolve()/'data'
    with args.delta_reference.open() as stream:
        # Production CSV may have leading provenance comments.
        rows = list(csv.DictReader(line for line in stream if not line.startswith('#')))
    fractions = {}
    for row in rows:
        key = int(row['atomic_number']), int(row['mass_number'])
        fractions.setdefault(key, []).append((float(row['energy_MeVu']), float(row['delta_electron_fraction'])))
    for medium in ('water', 'air', 'copper'):
        for species in ('proton', 'carbon'):
            case = args.campaign/f'{medium}_{species}'
            if medium == 'copper':
                shutil.copyfile(case/'primary_stopping.csv', out_path(out/'stopping/copper'/f'{species}.csv'))
                shutil.copyfile(case/'primary_inelastic_rate.csv', out_path(out/'nuclear/copper'/species/'primary_rate.csv'))
                shutil.copyfile(case/'primary_elastic.csv', out_path(out/'elastic/copper'/species/'primary.csv'))
                shutil.copyfile(case/'primary_metadata.json', out_path(out/'elastic/copper'/species/'primary.metadata.json'))
            else:
                raw = np.atleast_2d(np.loadtxt(case/'primary.phsp'))
                if raw.shape[1] != 5:
                    raise ValueError('Primary stopping columns')
                save_table(out/'stopping'/medium/f'{species}.csv',
                           'energy_MeVu,stopping_power_MeV_per_mm', raw[:, [0, 2]])
            if medium in ('water', 'copper'):
                raw = np.atleast_2d(np.loadtxt(case/'ions.phsp'))
                if raw.shape[1] != 8:
                    raise ValueError('Ion stopping columns')
                correction = np.empty(len(raw))
                for z, a in np.unique(raw[:, :2], axis=0):
                    key = int(z), int(a)
                    if key not in fractions:
                        raise ValueError(f'Missing calibrated delta fraction for {key}')
                    grid, values = np.array(sorted(fractions[key])).T
                    selected = (raw[:, 0] == z) & (raw[:, 1] == a)
                    if raw[selected, 2].min() < grid[0]-1e-5 or raw[selected, 2].max() > grid[-1]+1e-5:
                        raise ValueError(f'Delta reference domain does not cover {key}')
                    correction[selected] = np.interp(raw[selected, 2], grid, values)
                if not np.isfinite(correction).all() or np.any((correction < 0) | (correction >= 1)):
                    raise ValueError('Invalid calibrated delta fractions')
                values = np.column_stack([raw[:, :5], raw[:, 6:8], correction])
                name = f'ions_{species}.csv' if medium == 'copper' else f'ions_{species}_candidate.csv'
                save_table(out/'stopping'/medium/name,
                           'atomic_number,mass_number,energy_MeVu,stopping_power_MeV_per_mm,'
                           'restricted_stopping_power_MeV_per_mm,nuclear_stopping_power_MeV_per_mm,'
                           'raw_delta_electron_fraction,delta_electron_fraction', values)
                if medium == 'copper':
                    rates = np.atleast_2d(np.loadtxt(case/'ion_rates.phsp'))
                    save_table(out/'nuclear/copper'/f'ion_cross_sections_{species}_candidate.csv',
                               'atomic_number,mass_number,energy_MeVu,macroscopic_inelastic_cross_section_per_mm,mean_free_path_mm', rates)
    shutil.copyfile(args.campaign/'water_material/material.json', out_path(out/'materials/water/material.json'))
    report = dict(status='EXTRACTED_REQUIRES_VALIDATION', delta_reference=str(args.delta_reference.resolve()),
                  water_ion_candidates='Compare both source contexts before selecting ions.csv',
                  copper_ion_rate_candidates='Compare contexts before selecting ion_cross_sections.csv')
    (args.output_root/'tables_report.json').write_text(json.dumps(report, indent=2)+'\n')


def out_path(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    return path


if __name__ == '__main__':
    main()
