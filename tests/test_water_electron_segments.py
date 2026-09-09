import sys
import json
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from compile_water_electron_segments import audit_records, frame_segments
from water_electron_path_graph import build_path_graph, backward_points, NONE


def fixture():
    d = np.zeros((2, 34))
    d[:, 19:22] = [1, 1, 3]
    d[:, 5] = 1
    d[0, 2:5] = [1, 0, 1000060120]
    d[0, 6:9] = d[0, 10:13] = [0, 0, 20]
    d[0, 13:16] = [0, 0, 21]
    d[0, 9] = d[0, 17] = 10
    d[0, 16] = 9
    d[1, 2:5] = [2, 1, 11]
    d[1, 6:9] = d[1, 10:13] = [0, 0, 21]
    d[1, 13:16] = [1, 0, 22]
    d[1, 9] = d[1, 16] = d[1, 17] = 1
    d[1, 22:24] = [1, 10]
    d[1, 26:30] = [1, 1, 1, 1]
    return d


class WaterElectronSegments(unittest.TestCase):
    def audit(self, d):
        return audit_records(d, 1, [[-80, 80], [-80, 80], [0, 350]])

    def test_embedded_source_and_family_closure(self):
        *_, report = self.audit(fixture())
        self.assertEqual(report['families'], 1)
        self.assertEqual(report['max_family_relative_residual'], 0)

    def test_generating_step_mismatch(self):
        d = fixture(); d[1, 23] = 9
        with self.assertRaisesRegex(ValueError, 'Generating parent-step'):
            self.audit(d)

    def test_explicit_homogeneous_density(self):
        d=fixture();d[:,20]=1.07829
        bounds=[[-80,80],[-80,80],[0,350]]
        audit_records(d,1,bounds,expected_density_g_cm3=1.07829)
        with self.assertRaises(ValueError):
            self.audit(d)
        d[1,20]=1.06871
        with self.assertRaises(ValueError):
            audit_records(d,1,bounds,expected_density_g_cm3=1.07829)
        for bad in (0,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):
                audit_records(fixture(),1,bounds,expected_density_g_cm3=bad)

    def test_generating_step_mismatch_density_independent(self):
        d = fixture(); d[1, 23] = 9; d[:,20]=1.07829
        with self.assertRaisesRegex(ValueError, 'Generating parent-step'):
            audit_records(d,1,[[-80,80],[-80,80],[0,350]],expected_density_g_cm3=1.07829)

    def test_missing_steps(self):
        d = fixture(); d[1, 5] = 2
        with self.assertRaisesRegex(ValueError, 'Missing/duplicate'):
            self.audit(d)

    def test_energy_consistent_but_wrong_parent_step_position(self):
        d = fixture(); d[1, 8] = d[1, 12] = 20.5
        with self.assertRaisesRegex(ValueError, 'birth position mismatch'):
            self.audit(d)

    def test_no_family_renormalization(self):
        d = fixture(); d[1, 16] = .9
        with self.assertRaisesRegex(ValueError, 'family closure'):
            self.audit(d)

    def test_charged_escape_rejected(self):
        d = fixture(); d[1, 16] = .5; d[1, 18] = .5; d[1, 15] = 350
        with self.assertRaisesRegex(ValueError, 'Charged escape'):
            self.audit(d)

    def test_positive_terminal_inside_rejected(self):
        d = fixture(); d[1, 16] = .5; d[1, 18] = .5
        with self.assertRaisesRegex(ValueError, 'positive-KE terminal'):
            self.audit(d)

    def test_photon_escape_explicit(self):
        d = fixture(); d[1, 16] = .5
        photon = d[1].copy(); photon[2:5] = [3, 2, 22]
        photon[6:9] = photon[10:13] = [1, 0, 22]
        photon[13:16] = [1, 0, 350]
        photon[9] = photon[17] = photon[18] = .5; photon[16] = 0
        *_, report = self.audit(np.vstack([d, photon]))
        self.assertEqual(report['photon_escaped_MeV'], .5)

    def test_segment_not_collapsed_to_endpoint(self):
        d = fixture()[1:]
        s = frame_segments(d, d)
        self.assertEqual(s.shape, (1, 6))
        np.testing.assert_array_equal(s[0, :3], [0, 0, 0])
        self.assertAlmostEqual(s[0, 5], 1)
        self.assertAlmostEqual(np.linalg.norm(s[0, 3:]), np.sqrt(2))

    def test_photon_deposit_is_at_interaction_not_along_flight(self):
        d = fixture()[1:]; d[0, 4] = 22
        s = frame_segments(d, d)
        np.testing.assert_array_equal(s[:, :3], s[:, 3:])

    def test_graph_preserves_earlier_excursion(self):
        d = fixture()
        d[1, 13:16] = [0, 0, 24]
        later = d[1].copy(); later[5] = 2
        later[10:13] = [0, 0, 24]; later[13:16] = [0, 0, 22]
        d = np.vstack([d, later])
        g, pre, report = build_path_graph(d, d[:2], np.array([-1, 1, 1]))
        json.dumps(report)
        points = backward_points(g, pre[2])
        np.testing.assert_array_equal(points[:, 2], [3, 0])
        # A final point at z=1 is inside [0,2], but the earlier z=3 exits.
        self.assertGreater(points[:, 2].max(), 2)

    def test_descendant_starts_at_partial_parent_step(self):
        d = fixture(); d[1, 13:16] = [0, 0, 25]
        child = d[1].copy(); child[2:5] = [3, 2, 22]
        child[6:9] = child[10:13] = [0, 0, 23]
        child[13:16] = [0, 0, 24]
        d = np.vstack([d, child])
        g, pre, _ = build_path_graph(d, d, np.array([-1, 1, 1]))
        points = backward_points(g, pre[2])
        np.testing.assert_array_equal(points[:, 2], [2, 0])
        self.assertNotIn(4, points[:, 2])  # parent's future endpoint


if __name__ == '__main__':
    unittest.main()
