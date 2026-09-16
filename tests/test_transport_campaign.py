"""Acceptance tooling must reject incomplete/mismatched evidence."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'tools'/f'{name}.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m
analyzer = module('analyze_transport_campaign')
profiler = module('profile_transport_stages')

class CampaignTests(unittest.TestCase):
    def test_phase_selection_uses_generation_and_round(self):
        text='\n'.join(f'[secondary-round] begin={begin} round={rnd} active={active} blocks256=4 {tail}'
                       for begin,rnd,active,tail in [(0,0,10000,'segment'),(0,1,9000,'segment'),
                           (0,2,8500,'segment'),(0,3,8000,'finish_tail'),(12000,0,20000,'segment')])
        phases=profiler.stages(text)
        self.assertEqual(phases['secondary_late']['round'],2)
        self.assertEqual(phases['secondary_generation2']['begin'],12000)
        self.assertEqual(phases['secondary_generation2']['launch_skip'],4)
        with self.assertRaises(ValueError):
            profiler.stages('')

    def test_dose_audit_and_timing_are_independent_gates(self):
        with tempfile.TemporaryDirectory() as name:
            out=Path(name)
            rows=[]
            for kind in ('base','cand'):
                for i in range(5):
                    d=out/f'run_{i:02d}_{kind}';d.mkdir()
                    (d/'dose.mhd').write_text('NDims = 3\nElementType = MET_FLOAT\nDimSize = 2 1 1\nElementDataFile = dose.raw\n')
                    np.array([1, .1+i*.001 if kind=='base' else .2],dtype='<f4').tofile(d/'dose.raw')
                    rows.append(dict(kind=kind,index=i,warmup=False,audit=['1' if kind=='base' else '2'],
                        quality=dict(accepted=True,failures=[],queue_overflow_count=0),
                        **{k:10 if kind=='base' else 8 for k in ('wall_s','elapsed_s','primary_s','secondary_s')}))
            (out/'results.json').write_text(json.dumps(rows))
            r=analyzer.analyze(out)
            self.assertTrue(r['throughput_gate_pass'])
            self.assertFalse(r['dose_global_envelope_pass'])
            self.assertFalse(r['audit_identical'])
            self.assertFalse(r['promotion_allowed'])
            rows.pop()
            (out/'results.json').write_text(json.dumps(rows))
            with self.assertRaises(ValueError):analyzer.analyze(out)

    def test_paired_improvement_detects_reversal(self):
        self.assertGreater(analyzer.paired_stats([10]*5,[9]*5)['paired_gain_ci95'][0],.05)
        self.assertLess(analyzer.paired_stats([10]*5,[11]*5)['paired_gain_ci95'][1],0)

if __name__=='__main__':unittest.main()
