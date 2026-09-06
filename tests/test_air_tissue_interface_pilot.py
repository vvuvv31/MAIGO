import unittest,tempfile,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from analyze_air_tissue_interface_pilot import topas
from prepare_air_tissue_interface_pilot import material
class InterfacePilotTests(unittest.TestCase):
    def test_material_identity(self):
        table=Path(__file__).resolve().parents[1]/"data/HUtoMaterialSchneider.txt"
        rho,sec=material(-1000,table)
        self.assertEqual(sec,0)
        self.assertAlmostEqual(rho,.01131606474518776)
        rho,sec=material(100,table)
        self.assertEqual(sec,8)
        self.assertAlmostEqual(rho,1.0787997245788574)
    def test_3d_reader(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"dose.csv"
            rows=np.array([[x,y,0,1.] for x in range(100) for y in range(100)])
            header="TOPAS\nParameter\nScorer\nComponent\nX in 100 bins of 0.2 cm\nY in 100 bins of 0.2 cm\nZ in 1 bins of 0.05 cm\nDoseToMedium ( Gy ) : Sum"
            np.savetxt(p,rows,delimiter=",",header=header)
            self.assertEqual(topas(p,1)[0],10000)
            rows[-1]=rows[0]
            np.savetxt(p,rows,delimiter=",",header=header)
            with self.assertRaises(ValueError):topas(p,1)
if __name__=="__main__":unittest.main()
