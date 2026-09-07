import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from analyze_water_roi_rates import compare_rates
from analyze_water_actual_birth import clamped_hist

class RateComparisonTest(unittest.TestCase):
    def test_birth_endpoint_clamping(self):
        h=clamped_hist([0.,1.,2.,399.,400.,430.],0.,2.,200)
        self.assertEqual(h[0],2);self.assertEqual(h[1],1);self.assertEqual(h[-1],3)
        self.assertEqual(int(h.sum()),6)
    def test_nonfinite_birth_rejected(self):
        with self.assertRaises(ValueError):clamped_hist([float('nan')],-1.,.1,20)
    def arrays(self):
        ref=np.zeros((401,5));ref[:,0]=2;ref[:,1]=4;ref[:,2]=np.arange(401)+.01;ref[:,3]=.003;ref[:,4]=1/.003
        run=np.zeros(401,dtype=[(k,'f8') for k in ['Z','A','E_MeVu','total_per_mm','H_per_mm','O_per_mm','HO_in_domain']])
        run['Z']=2;run['A']=4;run['E_MeVu']=ref[:,2];run['total_per_mm']=.003
        run['H_per_mm']=.001;run['O_per_mm']=.002;run['HO_in_domain']=1
        return ref,run
    def test_exact(self):
        r=compare_rates(*self.arrays())['2,4'];self.assertEqual(r['fully_covered_points'],401)
        self.assertEqual(r['maximum_relative_difference'],0.)
    def test_masked_points_not_claimed_covered(self):
        ref,run=self.arrays();run['HO_in_domain'][:3]=0
        r=compare_rates(ref,run)['2,4'];self.assertEqual(r['fully_covered_points'],398)
        self.assertEqual(r['masked_or_reference_zero_points'],3)
    def test_bad_grid_rejected(self):
        ref,run=self.arrays();ref[1,2]=ref[0,2]
        with self.assertRaises(ValueError):compare_rates(ref,run)
    def test_bad_partial_sum_rejected(self):
        ref,run=self.arrays();run['total_per_mm'][5]=.01
        with self.assertRaises(ValueError):compare_rates(ref,run)

if __name__=='__main__':unittest.main()
