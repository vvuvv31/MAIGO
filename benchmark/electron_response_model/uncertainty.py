#!/usr/bin/env python3
"""Add independent training/test history-block uncertainty to the mono pilot."""
from pathlib import Path
import json
import numpy as np
from prototype import extract, reference, NAMES, save, sha

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT/'out/electron_mono_reference_20261002'
OUT = ROOT/'out/electron_mono_model_20261002'


def main():
    source = json.loads((INPUT/'manifest.json').read_text())[0]
    a,b,w,e,events,exposure,audit = extract(Path(source['path']).parent,electron_source=source)
    absolute = []
    for block in range(20):
        take = events % 20 == block
        values = reference(a[take],b[take],w[take],4)
        incident = exposure['electron_energy'][exposure['event'] % 20 == block].sum()
        absolute.append(values*w[take].sum()/incident)
    se = np.asarray(absolute).std(axis=0,ddof=1)/np.sqrt(len(absolute))
    report = json.loads((OUT/'report.json').read_text())
    checks = {}
    for name in ['peak','valley']:
        original = report['independent_absolute_errors'][name]
        combined = float(np.hypot(se[NAMES.index(name)],original['test_block_standard_error']))
        relative_halfwidth = 2.44*combined/abs(original['reference'])
        # Bonferroni-conservative two-ROI 95% interval, t(df=19) rounded up.
        # Also include measured fitting and quadrature differences explicitly.
        representation = abs(report['candidates'][-1]['errors'][name]['relative_difference'])
        quadrature = abs(report['quadrature_4_vs_8'][name]['relative_difference'])
        bound = abs(original['relative_difference'])+relative_halfwidth+representation+quadrature
        checks[name] = dict(point_relative_difference=original['relative_difference'],
                            training_block_standard_error=float(se[NAMES.index(name)]),
                            combined_relative_95_halfwidth=relative_halfwidth,
                            measured_representation_difference=representation,
                            measured_quadrature_difference=quadrature,
                            conservative_error_envelope=bound,within_1pct=bound<.01)
    result = dict(scope='Lung rho0.2994157076 g/cm3,0.65MeV single electrons; ideal infinite periodic0.5mm/3.6mm grating; homogeneous only',
                  criteria=checks,scoped_two_roi_check_pass=all(c['within_1pct'] for c in checks.values()),
                  ct_acceptance=False,source_report_sha256=sha(OUT/'report.json'),
                  note='Intervals rely on independent history blocks and do not bound energy interpolation, source-angle changes or interfaces')
    save(OUT/'uncertainty.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()
