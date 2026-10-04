#!/usr/bin/env python3
"""Summarize diagnostic-only electron histograms without double-counting ties."""
import argparse
import json
import re
from pathlib import Path

ENERGIES = ['<100 eV', '100 eV–1 keV', '1–10 keV', '10–100 keV', '0.1–1 MeV', '>=1 MeV']
EVENTS = ['continued', 'elastic', 'hard_null', 'hard_accepted', 'soft_or_native_deposit',
          'escaped', 'coverage_missing', 'invalid']
LIMITS = {1: 'max_step', 2: 'mean_loss', 4: 'range', 8: 'voxel_face', 16: 'collision_flight'}


def summarize(log):
    batches, rows = [], []
    for line in log.splitlines():
        m = re.fullmatch(r'\[electron-step-diagnostic\] energy_bin=(\d+) limiter_mask=(\d+) event=(\d+) count=(\d+)', line)
        if m:
            e, mask, event, n = map(int, m.groups())
            if not (0 <= e < 6 and 0 <= mask < 32 and 0 <= event < 8 and n > 0):
                raise ValueError('Invalid histogram entry')
            rows.append((e, mask, event, n))
        m = re.fullmatch(r'\[electron-step-diagnostic-total\] calls=(\d+) counted=(\d+)', line)
        if not m:
            continue
        calls, counted = map(int, m.groups())
        if sum(r[3] for r in rows) != calls or calls != counted:
            raise ValueError('Histogram does not close')
        energy = {label: sum(n for e, _, _, n in rows if e == i) for i, label in enumerate(ENERGIES)}
        outcomes = {label: sum(n for _, _, event, n in rows if event == i) for i, label in enumerate(EVENTS)}
        # Exclusive masks are additive; inclusive bit totals overlap at ties.
        masks = {str(k): sum(n for _, mask, _, n in rows if mask == k) for k in range(32)}
        limits = {label: sum(n for _, mask, _, n in rows if mask & bit) for bit, label in LIMITS.items()}
        batches.append(dict(calls=calls, energy_counts=energy, outcome_counts=outcomes,
                            exclusive_limiter_masks=masks, inclusive_limiter_counts=limits,
                            tied_limiter_calls=sum(n for _, mask, _, n in rows if mask.bit_count() > 1),
                            flight_soft_terminal=sum(n for _, mask, event, n in rows if mask & 16 and event == 4),
                            matrix=[dict(energy_bin=e, limiter_mask=m, event=v, count=n) for e, m, v, n in rows]))
        rows = []
    if rows or not batches:
        raise ValueError('Missing/incomplete histogram')
    return dict(diagnostic_only=True, timing_is_not_performance_evidence=True, batches=batches)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    report = summarize(args.log.read_text())
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    for batch in report['batches']:
        print(json.dumps({k: v for k, v in batch.items() if k != 'matrix'}, ensure_ascii=False))
