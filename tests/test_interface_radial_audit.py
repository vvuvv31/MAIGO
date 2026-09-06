import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from audit_joint_interface_stability import radial_metrics

class RadialAuditTests(unittest.TestCase):
    def test_weighted_moments_and_strict_outer_boundary(self):
        r2=np.array([[0.,100.,400.]])
        got=radial_metrics(np.array([[1.,2.,1.]]),r2)
        self.assertAlmostEqual(got['radial_rms_mm'],np.sqrt(150.))
        self.assertEqual(got['outer_10mm_fraction'],.25)
        self.assertEqual(got,radial_metrics(np.array([[10.,20.,10.]]),r2))

    def test_invalid_dose_rejected(self):
        for v in (np.array([[0.]]),np.array([[-1.]]),np.array([[np.nan]]),np.ones((2,2))):
            with self.assertRaises(ValueError):radial_metrics(v,np.array([[1.]]))

if __name__=='__main__':unittest.main()
