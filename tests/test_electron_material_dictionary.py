import json
from pathlib import Path
import sys
import unittest
import copy
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from electron_material_dictionary import MARKER,read_material_dictionary,bind_schneider_materials
from audit_schneider_response_scope import schneider_identity


def material():
    return dict(index=1100,name='PatientTissueFromHU99',density_g_cm3=1.07829,
                mean_excitation_eV=75.,elements=[dict(Z=1,A_g_mol=1.008,mass_fraction=1.)])


class DictionaryTest(unittest.TestCase):
    def test_binding_rejects_density_alias_and_wrong_composition(self):
        source=Path(__file__).resolve().parents[1]/'data/HUtoMaterialSchneider.txt'
        identity=schneider_identity(source,-1000)
        rho=float(format(identity['density_g_cm3'],'.6g'))
        m=dict(index=4000,name='PatientTissueFromHUNegative1000',density_g_cm3=rho,
               mean_excitation_eV=85.,elements=[dict(Z=z,A_g_mol=a,mass_fraction=f)
                    for z,a,f in [(7,14.007,.755),(8,15.999,.232),(18,39.948,.013)]])
        states=np.zeros((1,46));states[0,[20,41]]=rho;states[0,[42,43]]=4000
        bound=bind_schneider_materials({4000:m},states,source)
        self.assertEqual(bound[4000]['material_section'],0)
        json.dumps(bound,allow_nan=False)
        for field,value in [('name','Water_75eV'),('density_g_cm3',rho*1.01)]:
            wrong=copy.deepcopy(m);wrong[field]=value
            with self.assertRaises(ValueError):bind_schneider_materials({4000:wrong},states,source)
        wrong=copy.deepcopy(m);wrong['elements'][0]['mass_fraction']+=.001
        with self.assertRaises(ValueError):bind_schneider_materials({4000:wrong},states,source)
        with self.assertRaises(ValueError):bind_schneider_materials({},states,source)

    def test_worker_duplicates(self):
        m=material();line=MARKER+json.dumps(m)
        self.assertEqual(read_material_dictionary('G4WT0 > '+line+'\nG4WT1 > '+line),{1100:m})

    def test_conflicting_workers(self):
        a=material();b=material();b['density_g_cm3']=1.
        with self.assertRaises(ValueError):read_material_dictionary(MARKER+json.dumps(a)+'\n'+MARKER+json.dumps(b))

    def test_missing_or_corrupt(self):
        with self.assertRaises(ValueError):read_material_dictionary('no dictionary')
        for field,value in [('index',1.5),('density_g_cm3',float('nan')),('elements',[])]:
            m=material();m[field]=value
            with self.assertRaises(ValueError):read_material_dictionary(MARKER+json.dumps(m))


if __name__=='__main__':unittest.main()
