#!/usr/bin/env python3
"""Publish the C12 CT preset only after all C12 response tables are compiled."""
from pathlib import Path
import argparse,hashlib,json
import yaml

ROOT=Path(__file__).resolve().parents[3]
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'config/carbon_ct_minibeam.yaml')
    parser.add_argument('--transport-only',action='store_true',help='Isolated handoff test, never a dose acceptance preset');args=parser.parse_args()
    bank=ROOT/'data/ct_electron/carbon';index=bank/'index.json'
    if not args.transport_only:
        m=json.loads(index.read_text());assert m['projectile']==[6,12] and len(m['tables'])==63
    else:
        assert (ROOT/'out').resolve() in args.output.resolve().parents
    c=yaml.safe_load((ROOT/'config/proton_ct_minibeam.yaml').read_text())
    w=yaml.safe_load((ROOT/'config/carbon_water_minibeam.yaml').read_text())
    for key in ['primary_particle','initial_energy_MeV','minibeam_air_ionization_rate_file','minibeam_air_ionization_rate_sha256']:
        c.pop(key,None)
    for key in ['ct_schneider_primary_rate_file','ct_schneider_primary_cinel03_file','ct_schneider_physics_bundle_file',
                'primary_stopping_power_file','minibeam_copper_stopping_power_file','minibeam_air_stopping_power_file',
                'minibeam_copper_cross_section_file','minibeam_copper_elastic_file','minibeam_copper_inclxx_file',
                'minibeam_copper_ion_stopping_power_file','all_ion_elastic_file','all_ion_elastic_sha256',
                'elastic_recoil_stopping_file','elastic_recoil_stopping_sha256','minibeam_copper_loss_range_file']:
        c[key]=w[key]
    c.update(primary_atomic_number=6,primary_mass_number=12,initial_energy_MeVu=250,
        number_of_histories=100000,random_seed=1002111001,history_chunk_size=16384,
        minibeam_copper_proton_straggling_model='legacy_clipped_gaussian',
        minibeam_air_slit_max_step_mm=.05,
        material_electron_response_index_file='data/ct_electron/carbon/index.json',
        material_electron_response_index_sha256=sha(index) if not args.transport_only else '',material_electron_response_host_budget_MiB=98304,
        material_electron_response_hot_device_budget_MiB=1024,material_electron_queue_chunk=16384)
    if args.transport_only:
        for key in list(c):
            if key.startswith('material_electron_'):c.pop(key)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(('# TRANSPORT HANDOFF DIAGNOSTIC ONLY; local electrons, no dose acceptance.\n' if args.transport_only else '')+
        '# C12 lung CT minibeam research preset; acceptance in progress.\n'
        '# 250 MeV/u, monoenergetic 3x3 cm flat source; not the water TPS/emittance source.\n'
        '# C12-specific responses; shared CT beamline handoff and group32 electron queue.\n'+yaml.safe_dump(c,sort_keys=False))
    print(args.output)
if __name__=='__main__':main()
