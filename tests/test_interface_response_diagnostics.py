import sys,tempfile,unittest,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from predict_bulk_response_interface import predict,kernel
from analyze_longitudinal_holdout import sha
from prepare_bulk_electron_response import prepare

class ResponseDiagnosticsTests(unittest.TestCase):
    def test_exact_mass_transfer(self):
        point=lambda x:(np.array([x]),np.array([1.]),0.)
        out,esc=predict(np.array([0.,.1,.3]),np.array([1.,2.]),['air','tissue'],
                        {'air':point(.1),'tissue':point(0.)})
        np.testing.assert_allclose(out,[0,3],atol=1e-14);self.assertAlmostEqual(esc,0)
    def test_reverse_escape(self):
        point=lambda x:(np.array([x]),np.array([1.]),0.)
        out,esc=predict(np.array([0.,.1,.3]),np.array([1.,2.]),['air','tissue'],
                        {'air':point(0.),'tissue':point(-.2)})
        np.testing.assert_allclose(out,[2,0],atol=1e-14);self.assertAlmostEqual(esc,1)
    def test_zero_displacement_identity(self):
        edges=np.array([0,.001,.08,.09,1.]);energy=np.array([1.,3.,7.,2.])
        out,esc=predict(edges,energy,['a']*4,{'a':(np.array([0.]),np.array([1.]),0.)})
        np.testing.assert_array_equal(out,energy);self.assertAlmostEqual(esc,0)
    def test_missing_energy_not_renormalized(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            np.savez(p/'air_1.npz',energy_MeV=np.array([7.]),centroid_mass_g_cm2=np.array([[.1,.2]]))
            (p/'report.json').write_text(json.dumps({'cases':{'air/1':{'loss_MeV':10,'local_MeV':2,'escaped_MeV':0,'response_sha256':sha(p/'air_1.npz')}}}))
            with self.assertRaisesRegex(ValueError,'normalization'):kernel(p,'air')

if __name__=='__main__':unittest.main()
