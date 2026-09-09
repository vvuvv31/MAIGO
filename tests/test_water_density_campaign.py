"""Density pilots must change material density, not cut, step or energy."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT/'benchmark/benchmark20260908'
sys.path.insert(0, str(BENCH))
spec = importlib.util.spec_from_file_location('water_density_campaign', BENCH/'water_density_campaign.py')
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


def fixture():
    lines = []
    for component, values in {
        'World': {'HLX': 2100, 'HLY': 2100, 'HLZ': 1500},
        'Water': {'HLX': 2000, 'HLY': 2000, 'HLZ': 500, 'TransZ': 500},
        'ROI': {'HLX': 2000, 'HLY': 2000, 'HLZ': 500},
        'Source': {'TransZ': 200},
    }.items():
        lines.extend(f'd:Ge/{component}/{key} = {value} mm' for key, value in values.items())
    lines.extend(['s:Ge/Water/Material = "Water_75eV"', 's:Ge/ROI/Material = "Water_75eV"',
                  'i:Sc/Dose/XBins = 160', 'i:Sc/Dose/YBins = 160', 'i:Sc/Dose/ZBins = 2000',
                  'd:Ge/ROI/MaxStepSize = 0.005 mm', 'd:Ph/Default/CutForAllParticles = 0.05 mm',
                  'd:So/Beam/BeamEnergy = 5400 MeV', 'd:Ph/Default/EMRangeMax = 10 GeV'])
    return '\n'.join(lines)+'\n'


class DensityCampaignTest(unittest.TestCase):
    def test_material_and_fixed_physics(self):
        for rho in (.25, .5, 1., 1.5, 2.):
            with self.subTest(rho=rho):
                text, factor = campaign.prepare_text(fixture(), Path('/new'), Path('/old'), rho)
                self.assertEqual(text.count('Material = "ResponseWater75"'), 2)
                self.assertIn('BaseMaterial = "Water_75eV"', text)
                self.assertIn(f'Density = {rho:g} g/cm3', text)
                for token in ('MeanExcitationEnergy = 75 eV', 'MaxStepSize = 0.005 mm',
                              'CutForAllParticles = 0.05 mm', 'BeamEnergy = 5400 MeV',
                              'EMRangeMax = 10 GeV'):
                    self.assertIn(token, text)
                self.assertEqual(factor, max(1., 1./rho))
                self.assertIn(f'd:Ge/Water/HLZ = {500*factor:g} mm', text)

    def test_outside_plan_rejected(self):
        for rho in (0, -1, .1, 3, float('nan')):
            with self.assertRaises(ValueError):
                campaign.prepare_text(fixture(), Path('/new'), Path('/old'), rho)

    def test_changed_template_rejected(self):
        for text in (fixture().replace('HLX = 2100', 'HLX = 2101'),
                     fixture()+'d:Ge/World/HLX = 2100 mm\n'):
            with self.assertRaises(ValueError):
                campaign.prepare_text(text, Path('/new'), Path('/old'), 1.)


if __name__ == '__main__':
    unittest.main()
