import sys
from pathlib import Path
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from prepare_longitudinal_density_pilot import density, prepare
from analyze_longitudinal_density_pilot import inactive_scope

class DensityPilotTests(unittest.TestCase):
    def test_formula(self):
        table=Path(__file__).resolve().parents[1]/"data/HUtoMaterialSchneider.txt"
        self.assertAlmostEqual(density(-1000,table),0.01131606474518776,places=10)
        self.assertAlmostEqual(density(-975,table),0.03932345286011696,places=10)
        self.assertAlmostEqual(density(-951,table),0.06621015816926956,places=10)
        with self.assertRaises(ValueError):
            density(-950,table)

    def test_invalid_grid_creates_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            grid=tmp/"bad.cctg"
            grid.write_bytes(bytes(44))
            out=tmp/"output"
            with self.assertRaises(ValueError):
                prepare(tmp,out,tmp/"unused",grid)
            self.assertFalse(out.exists())

    def test_scope_gate(self):
        ledger=dict(E_schneider_primary_delta_longitudinal_moved_MeV=0,
                    E_schneider_primary_delta_longitudinal_escaped_scorer_MeV=0,
                    longitudinal_candidate=dict(invalid_marches=0))
        self.assertTrue(inactive_scope("same","same",ledger))
        self.assertFalse(inactive_scope("base","changed",ledger))
        ledger["E_schneider_primary_delta_longitudinal_moved_MeV"]=1e-6
        self.assertFalse(inactive_scope("same","same",ledger))

if __name__=="__main__":
    unittest.main()
