"""Convert TOPAS water ntuples without fitted scaling or old-water correction."""
import argparse,json
from pathlib import Path
import numpy as np
import yaml
from run_topas10x_gpu_benchmark import sha,config_write

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,required=True)
    root=p.parse_args().raw.resolve()
    manifest=json.loads((root/'manifest.json').read_text())
    for name,h in manifest['pins'].items():
        if sha(Path(name))!=h:raise ValueError('Source pin mismatch: '+name)
    c=np.loadtxt(root/'carbon.phsp'); ions=np.loadtxt(root/'ions.phsp')
    if c.shape!=(4001,5) or ions.shape!=(128032,8):raise ValueError('Incomplete extraction')
    if not np.isfinite(c).all() or not np.isfinite(ions).all():raise ValueError('Nonfinite data')
    if np.any(c[:,2]<=0) or np.any(ions[:,3]<=0):raise ValueError('Nonpositive stopping')
    for block in ions.reshape(32,4001,8):
        if not np.array_equal(block[:,2],c[:,0]):raise ValueError('Grid mismatch')
        if not np.all(block[:,:2]==block[0,:2]):raise ValueError('Species order')
    carbon=ions[(ions[:,0]==6)&(ions[:,1]==12),3]
    if not np.allclose(carbon,c[:,2],rtol=2e-5,atol=0):raise ValueError('Independent C12 columns disagree')
    out=root/'compiled_v2';out.mkdir(exist_ok=False)
    np.savetxt(out/'primary.csv',c[:,[0,2]],delimiter=',',fmt='%.12g',header='G4_WATER 78eV; unrestricted electronic stopping\nenergy_MeVu,stopping_power_MeV_per_mm')
    np.savetxt(out/'ions.csv',ions[:,[0,1,2,3,4,6,7,7]],delimiter=',',fmt='%.12g',header='G4_WATER 78eV; unscaled electronic stopping, delta fraction equals raw\natomic_number,mass_number,energy_MeVu,stopping_power_MeV_per_mm,restricted_stopping_power_MeV_per_mm,nuclear_stopping_power_MeV_per_mm,raw_delta_electron_fraction,delta_electron_fraction')
    cfg=yaml.safe_load(Path('config/unified_water_cinel03_smoke.yaml').read_text())
    cfg.update(primary_stopping_power_file=str(out/'primary.csv'),particle_stopping_power_file=str(out/'ions.csv'))
    config_write(out/'candidate.yaml',cfg)
    pins={str(f):sha(f) for f in [root/'carbon.phsp',root/'ions.phsp',out/'primary.csv',out/'ions.csv',out/'candidate.yaml']}
    (out/'manifest.json').write_text(json.dumps(dict(status='CANDIDATE_NOT_PRODUCTION',pins=pins,energy_max_MeVu=400.01),indent=2))
    print(out/'candidate.yaml')

if __name__=='__main__':main()
