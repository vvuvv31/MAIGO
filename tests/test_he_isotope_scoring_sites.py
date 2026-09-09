"""Structural guard; real GPU partition closure is checked separately."""
from pathlib import Path
import re
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


class HeIsotopeSites(unittest.TestCase):
    def test_compiled_isotope_category_contract(self):
        source = '''#include "carbon/particle.hpp"
static_assert(carbon::he_isotope_origin_category_count == 3);
static_assert(carbon::he_isotope_origin_category(2,3) == 0);
static_assert(carbon::he_isotope_origin_category(2,4) == 1);
static_assert(carbon::he_isotope_origin_category(2,6) == 2);
static_assert(carbon::he_isotope_origin_category(2,8) == 2);
static_assert(carbon::he_isotope_origin_category(1,3) == 3);
static_assert(carbon::he_isotope_origin_category(4,6) == 3);
static_assert(carbon::be_isotope_origin_category(4,6) == 0);
static_assert(carbon::be_isotope_origin_category(2,6) == 4);
'''
        subprocess.run(['c++', '-std=c++17', '-fsyntax-only', '-x', 'c++',
                        '-I', str(ROOT/'include'), '-'], input=source,
                       text=True, check=True)

    def test_each_existing_be_site_has_exactly_one_he_partner(self):
        source = (ROOT/'src/transport_sycl.cpp').read_text()
        sites = list(re.finditer(
            r'score_be_isotope_origin_voxel_device\([\s\S]*?;', source))
        self.assertEqual(len(sites), 5)
        self.assertEqual(source.count('score_he_isotope_origin_voxel_device('), 5)
        for site in sites:
            expected = site.group().replace('be_isotope', 'he_isotope')
            self.assertTrue(source[site.end():].lstrip().startswith(expected))


if __name__ == '__main__':
    unittest.main()
