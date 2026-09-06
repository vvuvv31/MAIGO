import json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from audit_ordered_response_source_statistics import audit
from analyze_longitudinal_holdout import sha

class SourceStatisticsTests(unittest.TestCase):
    def test_exposure_weighting_determinism_and_pins(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);sources={}
            for i,(exposure,fraction) in enumerate(((1.,.1),(9.,.9))):
                p=root/f'{i}.npz';loss=np.zeros(37);loss[33:36]=exposure
                np.savez(p,loss=loss,bins=np.array([33,34,35]),weights=np.full(3,exposure*fraction))
                sources[p.name]=dict(hu=100,payload_sha256=sha(p),raw_path=f'/campaign/tissue/seed{i}/steps.phsp')
            manifest=root/'manifest.json';manifest.write_text(json.dumps(dict(sources=sources)))
            result=audit(root,replicates=100)
            self.assertEqual(result,audit(root,replicates=100))
            for b in result['cases']['100']['bins']:
                self.assertAlmostEqual(b['pooled_nonlocal_fraction'],.82)
            sources['0.npz']['payload_sha256']='0'*64
            manifest.write_text(json.dumps(dict(sources=sources)))
            with self.assertRaisesRegex(ValueError,'SHA'):audit(root)

if __name__=='__main__':unittest.main()
