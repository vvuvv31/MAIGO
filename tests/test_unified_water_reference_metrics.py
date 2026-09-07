import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from compare_unified_water_reference import depth_metrics


class WaterReferenceMetricsTest(unittest.TestCase):
    def test_distal_crossing_not_entrance(self):
        result = depth_metrics(np.array([1., 8., 10., 9., 7., 0.]))
        self.assertEqual(result['peak_mm'], 1.25)
        self.assertEqual(result['r80_mm'], 2.0)

    def test_truncated_range_rejected(self):
        with self.assertRaises(ValueError):
            depth_metrics(np.array([1., 2., 3.]))

    def test_physical_conversion(self):
        # 1 Gy in one 2 mm3 water voxel, per 50k histories.
        self.assertAlmostEqual(2e-6 * 6.241509074e12 / 50000, 249.66036296)

    def test_long_phantom_retains_distal_position(self):
        profile=np.zeros(800)
        profile[450:454]=[8.,10.,9.,7.]
        result=depth_metrics(profile)
        self.assertEqual(result['peak_mm'],225.75)
        self.assertEqual(result['r80_mm'],226.5)

    def test_r80_invariant_under_history_scaling(self):
        profile=np.array([1.,8.,10.,9.,7.,0.])
        self.assertEqual(depth_metrics(profile),depth_metrics(profile*20))


if __name__ == '__main__':
    unittest.main()
