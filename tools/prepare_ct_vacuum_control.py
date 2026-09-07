"""Matched upstream-vacuum control, leaving patient CT unchanged."""
import json
from pathlib import Path
import yaml
from run_topas10x_gpu_benchmark import config_write,sha

def main():
    old=Path('/mnt/sda/wuwei/ct_ideal_pencil_20260906/em')
    root=Path('/mnt/sda/wuwei/ct_upstream_vacuum_20260906');root.mkdir(exist_ok=False)
    d=root/'em';d.mkdir()
    c=yaml.safe_load((old/'gpu.yaml').read_text())
    c['spots_enable_upstream_air_energy_loss']=False
    config_write(d/'gpu.yaml',c);config_write(d/'gpu_step025.yaml',dict(c,maximum_step_mm=.25))
    t=(old/'topas.txt').read_text().replace(str(old/'dose_topas'),str(d/'dose_topas'))
    t=t.replace('s:Ge/World/Material  = "Air"','s:Ge/World/Material  = "G4_Galactic"')
    assert 's:Ge/World/Material  = "G4_Galactic"' in t
    (d/'topas.txt').write_text(t)
    sl=(old/'run.slurm').read_text().replace(str(old),str(d)).replace('ideal_ct_em','ct_vacuum')
    (d/'run.slurm').write_text(sl)
    pins={str(p):sha(p) for p in (old/'topas.txt',old/'gpu.yaml',Path(c['ct_grid_file']),Path(c['tps_spots_file']),Path(c['tps_beam_model_file']),Path('/home/wuwei/topas/topas-build/topas'))}
    pins.update({str(p):sha(p) for p in root.rglob('*') if p.is_file()})
    (root/'manifest.json').write_text(json.dumps(dict(cases=[dict(label='em',directory=str(d),histories=50000)],inputs=pins,status='UPSTREAM_VACUUM_CONTROL'),indent=2))

if __name__=='__main__':main()
