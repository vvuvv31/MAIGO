import sys
import tempfile
import unittest
from pathlib import Path
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from run_topas10x_gpu_benchmark import config_write


class ConfigWriteTest(unittest.TestCase):
    def test_disabled_outputs_survive_two_writes(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'config.yaml'
            config_write(p,{'charged_origin_voxel_output_file':'','flag':True,'step':.25,'pos':[1.,2.,3.]})
            first=p.read_text()
            self.assertIsNone(yaml.safe_load(first)['charged_origin_voxel_output_file'])
            config_write(p,yaml.safe_load(first))
            self.assertEqual(p.read_text(),first)
            self.assertNotIn('None',p.read_text())

    def test_existing_scalar_spelling_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'config.yaml'
            config_write(p,{'a':False,'b':2,'c':.5,'file':'x.bin'})
            self.assertEqual(p.read_text(),'a: false\nb: 2\nc: 0.5\nfile: x.bin\n')


if __name__=='__main__':unittest.main()
