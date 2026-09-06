import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from electron_ordered_paths import OrderedPaths
from project_electron_ordered_pilot import move

def row(track,parent,step,birth,pre,post,pdg=11):
    r=np.zeros(34);r[2:6]=track,parent,pdg,step;r[6:9]=birth;r[10:13]=pre;r[13:16]=post
    return r

class Paths(unittest.TestCase):
    def fixture(self):
        return np.array([row(1,0,1,[0,0,0],[0,0,0],[0,0,1],1000060120),
            row(2,1,1,[0,0,.5],[0,0,.5],[1,0,.5]),
            row(2,1,2,[0,0,.5],[1,0,.5],[1,0,1.5]),
            row(3,2,1,[1,0,1],[1,0,1],[2,0,1],22),
            row(4,3,1,[2,0,1],[2,0,1],[2,1,1])])
    def test_full_family_and_midstep_birth(self):
        g=OrderedPaths(self.fixture())
        np.testing.assert_allclose(g.path(4,.5).sum(axis=0),[2,.5,.5])
        np.testing.assert_allclose(g.path(3,1).sum(axis=0),[2,0,.5])
    def test_missing_parent_and_disconnected_birth(self):
        r=self.fixture();r[4,3]=99
        with self.assertRaises(ValueError):OrderedPaths(r).path(4,.5)
        r=self.fixture();r[4,6:9]=[9,0,1];r[4,10:13]=[9,0,1]
        with self.assertRaises(ValueError):OrderedPaths(r).path(4,.5)
    def test_track_gap(self):
        r=self.fixture();r[2,10]=2
        with self.assertRaises(ValueError):OrderedPaths(r).path(2,.5)
    def test_vectorized_mass_path_two_crossings(self):
        birth=np.array([[0.,0.,-1.]])
        v=np.array([[.1,0,.1],[1,0,1],[0,0,-1],[0,0,-.1]])
        pos=birth.copy()
        for step in v:pos=move(pos,step[None,:],0,(1.,10.))
        np.testing.assert_allclose(pos,[[2,0,-1]],rtol=0,atol=1e-13)
        np.testing.assert_allclose(move(birth,v.sum(axis=0)[None,:],0,(1.,10.)),[[11,0,-1]],rtol=0,atol=1e-13)

if __name__=='__main__':unittest.main()
