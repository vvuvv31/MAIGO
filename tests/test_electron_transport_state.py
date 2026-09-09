import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from electron_transport_state import validate_states, displacement_length_audit, COLUMNS_V4, DTYPE_V4


def sample():
    d = np.zeros((1, 46))
    d[0, 17:19] = [1., .9]
    d[0, 20] = 1.
    d[0, 21] = 4
    d[0, 36] = 1.
    d[0, 37] = 1.  # Outgoing direction intentionally differs from the chord.
    d[0, 15] = .1
    d[0, 40:46] = [.11, .01, 5, 17, 1, 0]
    return d


class StateTest(unittest.TestCase):
    def test_state_not_chord(self):
        d = sample()
        self.assertIs(validate_states(d), d)
        self.assertEqual(len(COLUMNS_V4), 46)
        self.assertEqual(DTYPE_V4.itemsize, 312)

    def test_reject_invalid(self):
        for col, value in [(21, 3), (36, .5), (37, .5), (40, -.01),
                           (43, -1), (41, 0), (44, 8), (45, 6), (42, 1.5)]:
            with self.subTest(col=col):
                d = sample(); d[0, col] = value
                with self.assertRaises(ValueError): validate_states(d)

    def test_world_exit(self):
        d = sample(); d[0, 41:45] = [0, 5, -1, 0]
        validate_states(d)

    def test_msc_displacement_is_not_true_step_length(self):
        d = sample()
        d[0, 15] = 3.6952037e-6
        d[0, 40] = 9.75333017e-8
        original = d.copy()
        validate_states(d)
        report = displacement_length_audit(d)
        self.assertEqual(report['chord_exceeds_step_rows'], 1)
        np.testing.assert_array_equal(d, original)


if __name__ == '__main__': unittest.main()
