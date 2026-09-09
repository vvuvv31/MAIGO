"""CLI fail-closed checks for the isolated GPU fraction candidate."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import yaml

ROOT=Path(__file__).resolve().parents[1]
BINARY=ROOT/'build/oneapi-nvidia-loss-fraction/carbon_mc'


@unittest.skipUnless(BINARY.exists(), 'Requires candidate SYCL build')
class FractionCandidateConfigTest(unittest.TestCase):
    def test_terminal_em_candidate_rejects_production(self):
        base=yaml.safe_load((ROOT/'benchmark/list/b1_200/em_gaussian_clamped/gpu.yaml').read_text())
        base.update(enable_terminal_generation_em_transport=True)
        with tempfile.TemporaryDirectory() as tmp:
            config=Path(tmp)/'config.yaml';config.write_text(yaml.safe_dump(base))
            env=dict(os.environ,LD_LIBRARY_PATH='/home/wuwei/sycl_workspace/llvm/build/install/lib')
            run=subprocess.run([str(BINARY),'--config',str(config),'--device','cuda'],
                               cwd=tmp,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                               text=True,timeout=30)
            self.assertNotEqual(run.returncode,0)
            self.assertIn('enable_terminal_generation_em_transport requires smoke mode',run.stdout)

    def test_restricted_scope(self):
        base=yaml.safe_load((ROOT/'benchmark/list/b1_200/em_gaussian_clamped/gpu.yaml').read_text())
        base.update(run_mode='smoke',energy_straggling_model='packaged_fluctuation_fraction_hybrid',
                    enable_primary_loss_query_audit=True,
                    energy_straggling_package_file='/tmp/intentionally_missing_fraction_candidate.csv')
        # CT geometry loads before config validation, so an absent CT fixture
        # would only test a missing-file error, not this model's scope guard.
        cases=[dict(run_mode='production'),
               dict(straggling_scale=1.2),dict(enable_secondary_energy_straggling=True),
               dict(enable_step_stable_straggling=True,straggling_sampling_length_mm=.1),
               dict(enable_primary_loss_query_audit=False)]
        env=dict(os.environ,LD_LIBRARY_PATH='/home/wuwei/sycl_workspace/llvm/build/install/lib')
        with tempfile.TemporaryDirectory() as tmp:
            for change in cases:
                with self.subTest(change=change):
                    config=Path(tmp)/'config.yaml'
                    config.write_text(yaml.safe_dump(dict(base,**change)))
                    run=subprocess.run([str(BINARY),'--config',str(config),'--device','cuda'],
                                       cwd=tmp,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                       text=True,timeout=30)
                    self.assertNotEqual(run.returncode,0)
                    self.assertNotIn('Loaded packaged C-12',run.stdout)
                    self.assertTrue(any(s in run.stdout for s in
                                        ['requires smoke mode','smoke-only C12 water candidate',
                                         'requires enable_primary_loss_query_audit']),run.stdout)


if __name__=='__main__':unittest.main()
