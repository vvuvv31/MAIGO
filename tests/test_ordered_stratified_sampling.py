import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from compile_electron_ordered_pilot import stratified_sample

class SamplingTests(unittest.TestCase):
    def test_rare_tail_preserved_without_energy_rescaling(self):
        ids,w=stratified_sample(np.array([0,1]),np.array([999.,1.]),np.array([-20,0]),256,np.random.default_rng(7))
        self.assertEqual(len(ids),256)
        self.assertAlmostEqual(w[ids==0].sum(),999.)
        self.assertAlmostEqual(w[ids==1].sum(),1.)
        self.assertAlmostEqual(w.sum(),1000.)

    def test_within_stratum_weighted_mean_unbiased(self):
        estimates=[]
        for seed in range(500):
            ids,w=stratified_sample(np.array([0,1,2]),np.array([1.,3.,.01]),np.array([0,0,10]),64,np.random.default_rng(seed))
            estimates.append(np.dot(w,np.array([1.,4.,100.])[ids]))
        self.assertAlmostEqual(np.mean(estimates),14.,delta=.12)

    def test_invalid_or_insufficient_budget(self):
        for w,n in ((np.array([1.,-1.]),2),(np.array([1.,np.nan]),2),(np.ones(2),1)):
            with self.assertRaises(ValueError):stratified_sample(np.arange(2),w,np.arange(2),n,np.random.default_rng(1))

if __name__=='__main__':unittest.main()
