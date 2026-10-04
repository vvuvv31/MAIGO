#!/usr/bin/env python3
"""Native homogeneous lung electron fixture with the actual CT voxel sizes.

Independent uniform 0.5 mm slit source; score laboratory x without assuming
azimuthal symmetry of a finite voxel grid. Keep photon-sink scope explicit.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
import numpy as np
from run_reference import BINARY, ROOT, sha
from prototype import save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--histories', type=int, default=100000)
    p.add_argument('--seed', type=int, default=1002320001)
    p.add_argument('--source-z', type=float, default=.25)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    template = (ROOT/'out/electron_macro_sphere_lung065_r20_step025_20261002/train/topas.txt').read_text()
    material = re.search(r's:Ge/ROI/Material = "([^"]+)"', template)[1]
    env = dict(os.environ, TOPAS_G4_DATA_DIR='/software/geant4-11.3.2/share/Geant4/data',
               LD_LIBRARY_PATH='/software/topas/lib:/software/geant4-11.3.2/lib:'+os.environ.get('LD_LIBRARY_PATH', ''))
    tags = a.out/'tags.bin'; np.zeros((80, 20, 400), dtype='<i2').tofile(tags)
    cases = []
    for name, seed in [('train', a.seed), ('test', a.seed+1)]:
        d = a.out/name; d.mkdir()
        source = re.sub(r'i:Ts/Seed = \d+', f'i:Ts/Seed = {seed}', template)
        source = re.sub(r'i:So/Beam/NumberOfHistoriesInRun = \d+', f'i:So/Beam/NumberOfHistoriesInRun = {a.histories}', source)
        source = source.replace('s:Ge/ROI/Type = "TsSphere"', 's:Ge/ROI/Type = "TsImageCube"')
        source = re.sub(r'd:Ge/ROI/(RMin|RMax|SPhi|DPhi|STheta|DTheta) = .*\n', '', source)
        source = re.sub(r's:Ge/ROI/Material = .*\n', '', source)
        source += (f'\ns:Ge/ROI/InputDirectory = "{a.out.resolve()}/"\n'
                   's:Ge/ROI/InputFile = "tags.bin"\n'
                   's:Ge/ROI/ImagingToMaterialConverter = "MaterialTagNumber"\n'
                   'i:Ge/ROI/NumberOfVoxelsX = 400\ni:Ge/ROI/NumberOfVoxelsY = 20\ni:Ge/ROI/NumberOfVoxelsZ = 80\n'
                   'd:Ge/ROI/VoxelSizeX = 0.1 mm\nd:Ge/ROI/VoxelSizeY = 2 mm\nd:Ge/ROI/VoxelSizeZ = 0.5 mm\n'
                   'iv:Ge/ROI/MaterialTagNumbers = 1 0\n'
                   f'sv:Ge/ROI/MaterialNames = 1 "{material}"\n')
        source = source.replace('s:So/Beam/BeamPositionDistribution = "None"',
                                's:So/Beam/BeamPositionDistribution = "Flat"\n'
                                's:So/Beam/BeamPositionCutoffShape = "Rectangle"\n'
                                'd:So/Beam/BeamPositionCutoffX = 0.25 mm\n'
                                'd:So/Beam/BeamPositionCutoffY = 1 mm')
        source = re.sub(r'd:Ge/BeamPosition/TransZ = .*', f'd:Ge/BeamPosition/TransZ = {a.source_z:.17g} mm', source)
        (d/'topas.txt').write_text(source)
        start = time.monotonic()
        with (d/'topas.log').open('x') as log:
            code = subprocess.run([str(BINARY), 'topas.txt'], cwd=d, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        elapsed = time.monotonic()-start; log = (d/'topas.log').read_text()
        counts = re.findall(r'Particle source Beam: Total number of histories: (\d+)', log)
        unscored = sum(map(float, re.findall(r'Total amount of energy not scored for this reason: ([\d.eE+-]+) MeV', log)))
        path = (d/'steps.phsp').resolve()
        accepted = code == 0 and counts and int(counts[-1]) == a.histories and path.exists() and unscored < .65*a.histories*1e-6
        status = dict(name=name, seed=seed, accepted=bool(accepted), exit_code=code,
                      histories=a.histories, particle='e-', energy_MeV=.65, table='s1_mid9',
                      geometry='voxel_box_in_vacuum', half_lengths_mm=[20.,20.,20.],
                      source_slit_width_mm=.5, source_y_halfwidth_mm=1., source_z_mm=a.source_z, voxel_size_mm=[.1,2.,.5],
                      records_include_vacuum=True, requested_max_step_mm=.25, production_cut_mm=.05,
                      density_g_cm3=.2994157075881958, path=str(path),
                      input_sha256=sha(d/'topas.txt'), binary_sha256=sha(BINARY),
                      sha256=sha(path) if path.exists() else None, tags_sha256=sha(tags),
                      unscored_MeV=unscored, seconds=elapsed, ct_acceptance=False)
        cases.append(status); save(d/'status.json', status); save(a.out/'manifest.json', cases)
        print(json.dumps(status), flush=True)
        if not accepted:
            raise RuntimeError('Voxel reference did not complete')


if __name__ == '__main__':
    main()
