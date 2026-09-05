import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from plot_topas10x_isocenter_profiles import extract_profile, isocenter_index


class ProfileTests(unittest.TestCase):
    def test_fractional_isocenter_on_linear_dose(self):
        z, y, x = np.indices((5, 7, 9))
        dose = (x + 10*y + 100*z).astype(float)
        iso = np.array([3.25, 2.6, 1.4])
        spacing = np.array([.5, .5, 2])
        for axis, factor in enumerate([1, 10, 100]):
            distance, values = extract_profile(dose, iso, spacing, axis)
            expected = iso @ np.array([1, 10, 100]) + distance / spacing[axis] * factor
            np.testing.assert_allclose(values, expected, atol=1e-12)

    def test_passive_topas_rotation_and_voxel_centers(self):
        text = "\n".join([f"d:Ge/Patient/Trans{a} = {v} mm" for a, v in zip("XYZ", [-12.3588, -5.559, .3945])] + [f"d:Ge/Patient/Rot{a} = {v} deg" for a, v in zip("XYZ", [0, 0, 90])])
        dims = np.array([440, 440, 34])
        spacing = np.array([.5, .5, 2])
        index, local = isocenter_index(text, dims, spacing)
        np.testing.assert_allclose(local, [-5.559, 12.3588, -.3945], atol=1e-12)
        np.testing.assert_allclose(index, [208.382, 244.2176, 16.30275])

    def test_outside_isocenter_rejected(self):
        dose = np.ones((3, 3, 3))
        with self.assertRaises(ValueError):
            extract_profile(dose, [1, -1, 1], np.ones(3), 0)


if __name__ == "__main__":
    unittest.main()
