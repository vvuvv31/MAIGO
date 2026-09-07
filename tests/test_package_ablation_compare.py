import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from compare_package_ablation import compare, check_ledger_preservation


class AblationComparisonTests(unittest.TestCase):
    def test_ledger_preservation(self):
        before = dict(cinel02_diagnostics=[0, 0],
                      cinel02_species_transport_ledger_MeV=[10., 20.],
                      E_dep_in_grid_MeV=30.)
        self.assertTrue(check_ledger_preservation(before, before))
        for update in (dict(cinel02_diagnostics=[0, 1]),
                       dict(cinel02_species_transport_ledger_MeV=[0., 0.]),
                       dict(E_dep_in_grid_MeV=15.)):
            with self.assertRaises(RuntimeError):
                check_ledger_preservation(before, dict(before, **update))
        with self.assertRaises(RuntimeError):
            check_ledger_preservation(before, {})

    def test_retired_runtime_routes_do_not_return(self):
        root = Path(__file__).resolve().parents[1]
        source = (root/'src/transport_sycl.cpp').read_text()
        for retired in ('use_cinel02', 'InelasticPackageV2Table::from_binary',
                        'cinel02_select_water_target_device',
                        'cinel02_select_material_target_device',
                        'cinel02_diag_device',
                        'cinel02_replay_status_counts_device',
                        'cinel02_secondary_exposure_sums_device',
                        'secondary_projectile_index_device',
                        'sample_secondary_target_device'):
            self.assertNotIn(retired, source)
        for required in ('schneider_ct_device_ctx.secondary_rates',
                         'cinel02_species_energy_device',
                         'cinel02_species_terminal_device'):
            self.assertIn(required, source)

    def test_fail_closed_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = [Path(tmp)/name for name in ('a', 'b')]
            for d in (a, b):
                d.mkdir()
                np.array([1, 2, 3], dtype='<f4').tofile(d/'dose.raw')
                (d/'energy_ledger.json').write_text(json.dumps(
                    dict(histories=10, schneider_diagnostics=dict(primary_hazards=2))))
                (d/'quality_report.json').write_text(json.dumps(
                    dict(queue_overflow_count=0, queue_overflow_energy_MeV=0)))
            self.assertTrue(compare(a, b)['bitwise_dose_equal'])
            np.array([1, 2, 4], dtype='<f4').tofile(b/'dose.raw')
            with self.assertRaisesRegex(RuntimeError, 'changed dose'):
                compare(a, b)
            np.array([1, 2, 3], dtype='<f4').tofile(b/'dose.raw')
            (b/'energy_ledger.json').write_text(json.dumps(
                dict(histories=10, schneider_diagnostics=dict(primary_hazards=3))))
            with self.assertRaisesRegex(RuntimeError, 'counts changed'):
                compare(a, b)
            (b/'energy_ledger.json').write_text((a/'energy_ledger.json').read_text())
            (b/'quality_report.json').write_text(json.dumps(
                dict(queue_overflow_count=1, queue_overflow_energy_MeV=1)))
            with self.assertRaisesRegex(RuntimeError, 'Overflow'):
                compare(a, b)


if __name__ == '__main__':
    unittest.main()
