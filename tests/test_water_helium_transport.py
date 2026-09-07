import sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from analyze_water_helium_transport import nuclear_summary
from verify_water_species_tally_fix import energy_closure

def row(kind,track=7,z=2,mass=4,ke=400):
    r=['0']*30;r[0]=kind;r[1:5]=['0','1','2','3']
    r[5]='7';r[6]=str(track);r[10]=str(z);r[11]=str(mass)
    r[13]=str(ke);r[15]='2';r[25]='1'
    return ' '.join(r)+'\n'

class HeliumTransportAuditTest(unittest.TestCase):
    def test_queued_energy_already_contains_import(self):
        names=['queued_birth_kinetic','continuous_deposit_all','nuclear_local_deposit_all',
               'terminal_deposit_all','boundary_escape_kinetic','reaction_export_kinetic',
               'step_limit_escape_kinetic','reaction_import_kinetic','continuous_deposit_fov']
        ledger=dict(cinel02_species_transport_ledger_layout=dict(shape=[1,9],species=['4He'],metric=names),
                    cinel02_species_transport_ledger_MeV=[10,7,0,0,0,3,0,4,6])
        self.assertEqual(energy_closure(ledger)['4He'],0.)
        ledger['cinel02_species_transport_ledger_MeV'][1]=6
        self.assertAlmostEqual(energy_closure(ledger)['4He'],-.1)

    def run_rows(self,text):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'cascade';p.write_text(text)
            return nuclear_summary(p,10)

    def test_same_track_continuation(self):
        d=self.run_rows(row('interaction')+row('product',ke=10))[4]
        self.assertEqual(d['pre_step_KE'],40.)
        self.assertEqual(d['continuation_count'],1)
        self.assertEqual(d['pre_step_minus_continuation_KE'],39.)

    def test_other_child_is_not_continuation(self):
        d=self.run_rows(row('interaction')+row('product',track=8,ke=10))[4]
        self.assertEqual(d['continuation_count'],0)

    def test_identity_change_is_not_same_species(self):
        d=self.run_rows(row('interaction')+row('product',mass=3,ke=10))[4]
        self.assertEqual(d['continuation_count'],0)

    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError):self.run_rows(row('interaction')*2)
        with self.assertRaises(ValueError):self.run_rows(row('interaction')+row('product')*2)

if __name__=='__main__':unittest.main()
