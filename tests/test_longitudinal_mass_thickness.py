import sys
from pathlib import Path
import unittest
import numpy as np
from scipy.integrate import quad
from scipy.special import exp1
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from analyze_longitudinal_mass_thickness import integrated_cdf,weights,predict

class MassThicknessTests(unittest.TestCase):
    def test_primitive_against_quadrature(self):
        for lam in (4.,24.):
            for s in (.01,.5,5.,100.):
                def cdf(t):
                    x=t/lam
                    return 1-np.exp(-x)+x*exp1(x) if x>0 else 0.
                expected=quad(cdf,0,s,epsabs=1e-12)[0]
                self.assertAlmostEqual(float(integrated_cdf(s,lam)),expected,places=9)

    def test_probability_and_conservation(self):
        w=weights(10000,.5,24.)
        self.assertTrue(np.all(w>=0))
        self.assertAlmostEqual(float(w.sum()),1.,places=9)
        source=np.linspace(1,2,440)
        output,escape=predict(source,.084,24)
        self.assertAlmostEqual(float(output.sum()+escape),float(source.sum()),places=10)
        self.assertGreater(escape,0)

    def test_refinement_and_mass_scaling(self):
        x=np.linspace(1,2,100)
        coarse,_=predict(x,.084,8)
        fine,_=predict(np.repeat(x/2,2),.084,8,.25)
        np.testing.assert_allclose(coarse,fine.reshape(-1,2).sum(axis=1),rtol=1e-11,atol=1e-11)
        np.testing.assert_allclose(weights(100,.5,24),weights(100,.25,12),rtol=1e-12,atol=1e-12)

    def test_invalid(self):
        for lam in (0,-1,float("nan")):
            with self.assertRaises(ValueError):
                weights(10,.5,lam)
        with self.assertRaises(ValueError):
            predict([1,-1],.1,24)
        np.testing.assert_array_equal(predict([1,2],0,24)[0],[1,2])

if __name__=="__main__":
    unittest.main()
