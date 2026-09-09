import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

REPO=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('water_verifier',REPO/'tools/verify_unified_water_stopping.py')
verifier=importlib.util.module_from_spec(spec);spec.loader.exec_module(verifier)

class WaterStoppingVerifierTests(unittest.TestCase):
    def test_real_candidate(self):
        self.assertEqual(verifier.verify(REPO)['energy_max_MeVu'],400.01)

    def test_corruption_and_missing_file_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);target=root/'data/water_unified';target.mkdir(parents=True)
            for p in (REPO/'data/water_unified').glob('g4_water78*'):
                shutil.copy2(p,target/p.name)
            meta=target/'g4_water78_stopping_v1.metadata.json'
            original=meta.read_text();obj=json.loads(original)
            obj['files'].pop(next(iter(obj['files'])))
            meta.write_text(json.dumps(obj))
            with self.assertRaisesRegex(ValueError,'Wrong file set'):verifier.verify(root)
            meta.write_text(original)
            p=target/'g4_water78_primary_v1.csv'
            with p.open('a') as f:f.write('\n')
            with self.assertRaisesRegex(ValueError,'SHA mismatch'):verifier.verify(root)

if __name__=='__main__':unittest.main()
