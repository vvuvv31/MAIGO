import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from audit_electron_path_collapse import endpoint

class PathCollapse(unittest.TestCase):
    def test_uniform_identity(self):
        b=np.array([1.,2.,-1.]);v=np.array([3.,4.,5.])
        np.testing.assert_allclose(endpoint(b,v,0,(2.,2.)),b+v/2,rtol=0,atol=1e-14)

    def test_straight_forward_and_reverse(self):
        np.testing.assert_allclose(endpoint(np.array([0.,0.,-1.]),np.array([11.,0.,11.]),0,(1.,10.)),[2,0,1])
        np.testing.assert_allclose(endpoint(np.array([2.,0.,1.]),np.array([-11.,0.,-11.]),0,(1.,10.)),[0,0,-1],rtol=0,atol=1e-14)

    def test_collapsing_curved_path_loses_material_order(self):
        # Path (0,0,-1)->(2,0,1)->(2,0,-1) has rho-weighted net
        # vector (11,0,0). Replaying that vector loses TWO crossings.
        actual=np.array([2.,0.,-1.])
        collapsed=endpoint(np.array([0.,0.,-1.]),np.array([11.,0.,0.]),0,(1.,10.))
        self.assertAlmostEqual(np.linalg.norm(collapsed-actual),9.)

if __name__=='__main__':unittest.main()
