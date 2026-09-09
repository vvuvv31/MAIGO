"""Host loader regression for the isolated fraction-axis candidate."""
import pathlib
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class FractionLoaderTest(unittest.TestCase):
    def test_axis_and_complete_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source = root / 'check.cpp'
            source.write_text('''
#include "carbon/energy_loss_fluctuation.hpp"
#include <exception>
int main(int argc, char** argv) {
    try {
        auto t = carbon::EnergyLossFluctuationTable::from_csv(argv[1], argv[2][0]=='1');
        return t.sample_loss_ratio(150, .15, .5) == 1.0 ? 0 : 3;
    } catch (const std::exception&) { return 2; }
}
''')
            exe = root / 'check'
            subprocess.run(['g++', '-std=c++20', '-O0', '-I'+str(ROOT/'include'),
                            str(source), str(ROOT/'src/energy_loss_fluctuation.cpp'),
                            '-o', str(exe)], check=True)
            def check(axis, rows, fraction_mode, expected):
                path = root / 'table.csv'
                path.write_text('projectile_Z,projectile_A,material,energy_MeV_per_u,'+
                                axis+',q_0,q_0.5,q_1\n'+''.join(
                                    f'6,12,Water_75eV,{e},{f},0,1,2\n' for e, f in rows))
                result = subprocess.run([str(exe), str(path), str(int(fraction_mode))])
                self.assertEqual(result.returncode, expected)
            rows = [(e, f) for e in (100, 200) for f in (.1, .2)]
            check('mean_loss_fraction', rows, True, 0)
            check('mean_loss_fraction', rows, False, 2)
            check('areal_density_g_per_cm2', rows, False, 0)
            check('areal_density_g_per_cm2', rows, True, 2)
            check('mean_loss_fraction', rows + [(100, .3)], True, 2)
            check('mean_loss_fraction', rows + [(100, .1)], True, 2)
            check('mean_loss_fraction', [(e, f*10) for e, f in rows], True, 2)
            check('mean_loss_fraction', [(e, 0 if f == .1 else f) for e, f in rows], True, 2)


if __name__ == '__main__':
    unittest.main()
