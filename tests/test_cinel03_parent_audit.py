import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from audit_cinel03_parent_state import dtype_for

class ParentAuditTest(unittest.TestCase):
    def test_packed_wire_layout(self):
        source=(Path(__file__).resolve().parents[1]/'include/carbon/inelastic_package_v3.hpp').read_text()
        self.assertEqual(dtype_for(source,'Cinel03PackageHeader').itemsize,136)
        dtype=dtype_for(source,'Cinel03InteractionRecord')
        self.assertEqual(dtype.itemsize,476)
        self.assertEqual(dtype.fields['parent_energy_MeV'][1],152)
        sample=np.zeros(3,dtype=dtype)
        sample['parent_status']=[0,2,0];sample['parent_energy_MeV']=[12,0,0]
        selected=(sample['parent_status']==0)&(sample['parent_energy_MeV']>1e-4)
        self.assertEqual(selected.tolist(),[True,False,False])

if __name__=='__main__':unittest.main()
