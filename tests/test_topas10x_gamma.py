import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from evaluate_topas10x_gpu_gamma import pass_mask


class GammaTests(unittest.TestCase):
    def test_identical_and_constant_bias(self):
        ref=np.ones((7,7,7),dtype=np.float32)
        pts=np.array([[3,3,3],[2,2,2]])
        for dta in (0,1,2,3):
            for local in (False,True):
                self.assertTrue(pass_mask(ref,ref,pts,np.ones(3),1,dta,local).all())
                self.assertFalse(pass_mask(ref*1.1,ref,pts,np.ones(3),3,dta,local).any())

    def test_zero_dta_is_dose_only(self):
        ref=np.zeros((7,7,7),dtype=np.float32);ref[3,3,3]=1
        evaluation=np.zeros_like(ref);evaluation[3,3,4]=1
        pts=np.array([[3,3,3]])
        self.assertFalse(pass_mask(evaluation,ref,pts,np.ones(3),3,0,False)[0])
        self.assertTrue(pass_mask(evaluation,ref,pts,np.ones(3),3,1,False)[0])
        # Same index shift along a 2 mm axis must not pass the 1 mm gate.
        self.assertFalse(pass_mask(evaluation,ref,pts,np.array([1,1,2]),3,1,False)[0])

    def test_local_global_normalization(self):
        ref=np.ones((3,3,3),dtype=np.float32);ref[1,1,1]=.1
        evaluation=ref.copy();evaluation[1,1,1]=.12
        pts=np.array([[1,1,1]])
        self.assertTrue(pass_mask(evaluation,ref,pts,np.ones(3),3,0,False)[0])
        self.assertFalse(pass_mask(evaluation,ref,pts,np.ones(3),3,0,True)[0])

    def test_spot_remainder_conservation(self):
        counts=np.array([0,1,19,20,21,99,10000001],dtype=np.int64)
        shards=np.array([counts//20+(((i-np.arange(len(counts)))%20)<counts%20) for i in range(20)])
        np.testing.assert_array_equal(shards.sum(axis=0),counts)
        self.assertTrue((shards>=0).all())


if __name__=="__main__":unittest.main()
