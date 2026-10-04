#!/usr/bin/env python3
"""Independent analytic geometry/lineage checks for the research operator."""
import unittest
import numpy as np
from joint_family import transport, box_fraction, DEPOSIT, PHOTON_DEPOSIT, ESCAPE, COVERAGE


def fixture(delta, fraction, pdg=11, child=-1, inside=True):
    fraction = np.asarray([fraction], float)
    graph = dict(delta=np.asarray([delta], float), length=np.array([min(.05, np.linalg.norm(delta))]),
                 energy=np.ones(1), pdg=np.array([pdg]), inside=np.array([inside]),
                 fraction=fraction, child=np.array([[child]], int), next=np.array([-1]))
    entry = dict(edges=np.ones(2), nearest=np.array([0]), offsets=np.array([0,1]), cdf=np.ones(1),
                 delta=np.zeros((1,3)), length=np.zeros(1), fraction=np.array([[0.,0.,1.]]),
                 child=np.zeros((1,1), int), source_direction=np.array([[0.,0.,1.]]),
                 source_energy=np.ones(1), postdir=np.array([[0.,0.,1.]]))
    return dict(graph=graph, entry=entry, continuation=entry)


class Checks(unittest.TestCase):
    def run_model(self, model, n=10000):
        return transport(model, 1., n, np.random.default_rng(17073))

    def test_photon_deposition_is_at_interaction(self):
        # Every photon interacts at z=.25+5; it must not smear dose along
        # its free flight, and W must not be multiplied by a birth fraction.
        pos, status, _, _ = self.run_model(fixture([0,0,5], [1,0,0], 22))
        self.assertTrue(np.all(status == PHOTON_DEPOSIT))
        np.testing.assert_array_equal(pos[:,2], np.full(len(pos), 5.25))

    def test_photon_crossing_before_interaction(self):
        pos, status, _, _ = self.run_model(fixture([0,0,50], [1,0,0], 22))
        self.assertTrue(np.all(status == ESCAPE))
        np.testing.assert_array_equal(pos[:,2], np.full(len(pos), 20.))

    def test_native_source_end_is_not_escape_or_dose(self):
        _, status, _, _ = self.run_model(fixture([0,0,.01], [0,1,0]))
        self.assertTrue(np.all(status == COVERAGE))
        _, status, _, _ = self.run_model(fixture([0,0,.01], [1,0,0], inside=False))
        self.assertTrue(np.all(status == COVERAGE))

    def test_electron_uniform_chord(self):
        pos, status, _, _ = self.run_model(fixture([0,0,.05], [1,0,0]))
        self.assertTrue(np.all(status == DEPOSIT))
        self.assertLess(abs(pos[:,2].mean()-.275), .001)
        self.assertTrue(np.all((pos[:,2] >= .25) & (pos[:,2] <= .30)))

    def test_first_boundary_not_endpoint_material(self):
        pos = np.array([[0.,0.,0.], [19.9,0.,0.], [0.,0.,0.], [-20.,0.,0.]])
        delta = np.array([[50.,80.,0.], [.2,0.,0.], [0.,0.,0.], [-.01,0.,0.]])
        np.testing.assert_allclose(box_fraction(pos, delta, 20.), [.25,.5,1.,0.], atol=1e-13)

    def test_electron_step_cap_is_enforced(self):
        model = fixture([0,0,.3], [1,0,0])
        model['graph']['length'][0] = .3
        with self.assertRaises(AssertionError):
            self.run_model(model)

    def test_rotation_preserves_child_frame(self):
        # Native source axis x mapped to lab z; source-space child +x
        # displacement must arrive at lab z+.05, regardless of azimuth.
        model = fixture([.05,0,0], [1,0,0], 22)
        model['entry']['source_direction'][0] = [1.,0.,0.]
        pos, status, _, _ = self.run_model(model)
        self.assertTrue(np.all(status == PHOTON_DEPOSIT))
        np.testing.assert_allclose(pos[:,2], .30, atol=1e-14)


if __name__ == '__main__':
    unittest.main()
