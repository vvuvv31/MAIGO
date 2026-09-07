import sys,unittest,tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from analyze_water_helium_birth import dose_identity,rows,key

class DoseIdentityTest(unittest.TestCase):
    def test_row_schema(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'rows'
            values=['product']+['0']*29
            values[1:5]=['1','2','3','4']
            p.write_text('# comment\n'+' '.join(values)+'\n')
            self.assertEqual(key(next(rows(p))),(1,2,3,4))
            p.write_text(' '.join(values[:-1])+'\n')
            with self.assertRaises(ValueError):list(rows(p))

    def test_diagnostic_has_no_cross_section_query(self):
        source=(Path(__file__).resolve().parents[1]/'tools/topas_helium_birth/CarbonBirthOnlyNtuple.cc').read_text()
        self.assertNotIn('G4HadronicProcessStore',source)
        self.assertNotIn('GetInelasticCrossSectionPerVolume',source)
        self.assertIn('macroscopic_inelastic_per_mm_ = 0.0;',source)

    def test_exact(self):
        a=np.array([0.,1.,2.])
        self.assertTrue(dose_identity(a,a.copy())['total_unchanged'])

    def test_equal_integral_is_not_identity(self):
        r=dose_identity(np.array([1.,2.]),np.array([2.,1.]))
        self.assertFalse(r['total_unchanged'])
        self.assertEqual(r['relative_integral_change'],0.)
        self.assertEqual(r['changed_voxels'],2)

    def test_small_difference_is_not_silently_accepted(self):
        a=np.array([1.]);b=np.nextafter(a,np.inf)
        self.assertFalse(dose_identity(a,b)['total_unchanged'])

    def test_invalid(self):
        for a,b in [([],[]),([1.],[1.,2.]),([1.],[float('nan')]),([1.],[-1.]),([0.],[0.])]:
            with self.assertRaises(ValueError):dose_identity(np.array(a),np.array(b))

if __name__=='__main__':unittest.main()
