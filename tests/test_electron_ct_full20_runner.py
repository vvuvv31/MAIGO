import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import run_electron_ct_full20 as runner
from export_electron_ordered_response import canonical_ceiling_roundoff, swept_bin_lower


class Full20RunnerTest(unittest.TestCase):
    def test_swept_domain_requires_recorded_step(self):
        np.testing.assert_array_equal(swept_bin_lower([.02,5.01],[0.,4.9],[0,1]),[0.,5.])
        np.testing.assert_array_equal(swept_bin_lower([.02],[.01],[0]),[.01])
        with self.assertRaises(ValueError):swept_bin_lower([.02],[-.01],[0])

    def test_roundoff_is_not_physical_clamping(self):
        a=canonical_ceiling_roundoff([495.,500.+3*np.spacing(500.)],500.)
        np.testing.assert_array_equal(a,[495.,500.])
        with self.assertRaises(ValueError):canonical_ceiling_roundoff([500.+1e-8],500.)

    def test_aggregate_and_hard_failure(self):
        for mode in ('clean', 'fail', 'overflow', 'half'):
            bad=mode=='fail'
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); old=root/'old'; old.mkdir(); shards=[]
                binary=root/'binary'; binary.write_bytes(b'fake')
                joint=root/'joint.csv'; joint.write_text('fake')
                joint.with_suffix('.metadata.json').write_text('{}')
                spots=root/'spots.csv'; spots.write_text('spot_id,energy,weight\n1,200,1\n')
                geom='DimSize = 2 2 2\nElementDataFile = gpu_sum.raw\n'
                (old/'gpu_sum.mhd').write_text(geom)
                for name in ('gpu_sum.raw','topas_sum.raw'):
                    np.full((2,2,2),20,dtype='<f4').tofile(old/name)
                for i in range(20):
                    d=old/f'shard_{i:02d}'; d.mkdir()
                    (d/'config.yaml').write_text(yaml.safe_dump(dict(number_of_histories=1,
                        tps_spots_file=str(spots),random_seed=i)))
                    shards.append(dict(directory=str(d),histories=1))
                runner.save(old/'manifest.json', dict(case='synthetic',histories=20,shards=shards,
                    mapping='native',gpu_shape_zyx=[2,2,2],topas_shape_zyx=[2,2,2],spacing_zyx=[1,1,1],
                    reference_sum_sha256=runner.sha(old/'topas_sum.raw')))
                runner.save(old/'execution.json',dict(status='complete',completed=shards,
                    aggregate_sha256=runner.sha(old/'gpu_sum.raw')))
                def fake_run(source,dest,exe,table):
                    dest.mkdir()
                    q=dict(accepted=False,queue_overflow_count=0,queue_overflow_energy_MeV=0,
                        failures=[dict(code='unvalidated_electron_joint_response')])
                    if bad:q['failures'].append(dict(code='electron_joint_lookup_or_geometry_failure'))
                    overflow=mode=='overflow' and dest.name.startswith('shard_')
                    if overflow:q['queue_overflow_count']=1
                    runner.save(dest/'quality_report.json',q)
                    runner.save(dest/'energy_ledger.json',dict(electron_joint_response=dict(
                        patient_experiment=True,ordered_path_replays=1,domain_misses=int(bad),invalid_marches=0)))
                    np.full((2,2,2),999 if overflow else 1,dtype='<f4').tofile(dest/'dose.raw')
                    (dest/'dose.mhd').write_text(geom)
                    runner.save(dest/'run_report.json',dict(binary_sha256=runner.sha(exe),
                        dose_sha256=runner.sha(dest/'dose.raw'),inputs={str(source):runner.sha(source)},histories=1,seconds=1))
                with patch.object(runner,'run',side_effect=fake_run):
                    if bad:
                        with self.assertRaisesRegex(ValueError,'Unexpected quality'):runner.full(old,root/'new',binary,joint)
                        self.assertFalse((root/'new/gamma.json').exists())
                    else:
                        runner.full(old,root/'new',binary,joint,half=mode=='half')
                        result=json.loads((root/'new/gamma.json').read_text())
                        self.assertEqual(set(result['delta_percentage_points'].values()),{0.0})
                        expected=10 if mode=='half' else 20
                        self.assertEqual(result['histories'],expected)
                        self.assertEqual(result['scale'],2.0 if mode=='half' else 1.0)
                        self.assertTrue(np.all(np.fromfile(root/'new/gpu_sum.raw','<f4')==expected))
                        if mode=='overflow':
                            state=json.loads((root/'new/execution.json').read_text())
                            self.assertEqual(len(state['overflow_excluded']),20)


if __name__=='__main__':unittest.main()
