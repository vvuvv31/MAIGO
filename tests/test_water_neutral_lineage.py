"""Check emitted filter sets are disjoint and exhaustive under TOPAS semantics."""
import re,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from water_neutral_lineage import GROUPS
import numpy as np
from analyze_water_lineage_roundoff import closes

def accepts(filters,ancestry):
    for line in filters:
        names=set(re.findall(r'"([^"]+)"',line))
        intersects=bool(names & ancestry)
        if 'AncestorNotNamed' in line:
            if intersects:return False
        elif not intersects:return False
    return True

class LineagePartitionTest(unittest.TestCase):
    def test_independent_output_rounding(self):
        parts=[np.array([.1,.003],dtype=np.float32).astype(float),
               np.array([.2,.006],dtype=np.float32).astype(float),
               np.array([.7,.001],dtype=np.float32).astype(float)]
        total=np.array([1.,.01],dtype=np.float32).astype(float)
        self.assertTrue(closes(total,parts))
        parts[0][1]+=.00001
        self.assertFalse(closes(total,parts))

    def test_all_ancestry_combinations(self):
        for ancestry,expected in [(set(),'neither'),({'gamma'},'gamma_no_neutron'),
                                  ({'neutron'},'neutron'),({'gamma','neutron'},'neutron')]:
            hits=[name for name,filters in GROUPS.items() if name!='total' and accepts(filters,ancestry)]
            self.assertEqual(hits,[expected])
            self.assertTrue(accepts(GROUPS['total'],ancestry))

if __name__=='__main__':unittest.main()
