#!/usr/bin/env python3
"""Ensure the restricted runtime rejects old interfaces before transport."""
import argparse,json,os,subprocess,tempfile
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
KEYS=['broadbeam_electron_birth_order', 'broadbeam_electron_density_threshold', 'broadbeam_electron_halo_mm', 'broadbeam_electron_low_density_survival', 'broadbeam_electron_tissue_survival', 'broadbeam_electron_tracking', 'enable_electron_transport', 'material_electron_birth_file', 'material_electron_charged_index_file', 'material_electron_group_size', 'material_electron_joint_pilot_file', 'material_electron_queue_budget_MiB', 'material_electron_queue_chunk', 'material_electron_queue_lanes', 'material_electron_retain_bank', 'material_photon_response_index_file']+['tps_isocenter_mm','tps_isocenter_x_mm','tps_isocenter_y_mm','primary_particle','initial_energy_MeV','primary_rest_mass_MeV','beam_energy_spread_percent','primary_urban_reference_particle','primary_urban_ionisation_process','minibeam_copper_loss_range_file','scorer_mode','enable_fragment_species_scoring','cinel02_max_secondary_inelastic_generations','ct_dicom_origin_mode','scorerLET','enable_let_scoring','output_file','dose_output_file','topas_spots_file','topas_spots_files','spot_weights_file','spots_geometry_mode','ion_physics_file','ct_validation_mode','tps_source','ct_schneider_cross_section_file','minibeam_collimator_width_mm','minibeam_collimator_length_mm','primary_inelastic_package_v2_file','material_electron_rate26_file','broadbeam_electron_region_diagnostics','enable_layered_phantom']
KEYS += ['enable_neutron_transport', 'neutron_birth_output_file', 'neutron_birth_capacity', 'neutron_package_file', 'neutron_world_half_width_mm', 'neutron_max_collisions', 'neutron_recoil_stopping_manifest', 'neutron_elastic_two_body', 'neutron_water_recoil_urban', 'neutron_exact_plane_boundaries', 'neutron_proton_birth_output_file', 'neutron_proton_birth_capacity', 'neutral_origin_voxel_output_file']
KEYS += ['enable_charged_origin_voxel_scoring', 'minibeam_phase_space_at_copper_exit']
KEYS += ['phantom_length_mm', 'depth_bin_width_mm', 'minibeam_copper_proton_straggling_model']
KEYS += ['physics_profile','spots_lateral_yz_skew','spots_lateral_yz_skew_auto_pivot','spots_lateral_yz_skew_pivot_mm','spots_lateral_yz_rotation_deg','spots_lateral_yz_rotation_pivot_y_mm','mass_number','stopping_power_file','nuclear_cross_section_file','reaction_package_file','tps_particle_type']
KEYS += ['ct_schneider_radiation_length_file']
FLAGS=['--plan-manifest','--sequential-spots','--scorer-let','--dose-output','--straggling-scale']
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--binary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cfg=yaml.safe_load((ROOT/'examples/proton_water_broadbeam.yaml').read_text())
    env={k:v for k,v in os.environ.items() if not k.startswith('CARBON_')}
    env['CUDA_DISABLE_PTX_JIT']='1'
    reports=[]
    with tempfile.TemporaryDirectory(prefix='maigo_clean_interfaces_') as tmp:
        for key in KEYS:
            c=dict(cfg);c[key]='';f=Path(tmp)/'case.yaml';f.write_text(yaml.safe_dump(c,sort_keys=False))
            r=subprocess.run([str(a.binary.resolve()),'--config',str(f)],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            ok=r.returncode!=0 and 'Unsupported preset configuration key' in r.stdout and '[unified-em-audit]' not in r.stdout
            reports.append(dict(key=key,passed=ok));assert ok,(key,r.stdout)
        for pipeline in ['combined','carbon_broadbeam','proton_minibeam','']:
            c=dict(cfg);c['transport_pipeline']=pipeline
            f=Path(tmp)/'case.yaml';f.write_text(yaml.safe_dump(c,sort_keys=False))
            r=subprocess.run([str(a.binary.resolve()),'--config',str(f)],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            ok=r.returncode!=0 and 'transport_pipeline' in r.stdout and '[unified-em-audit]' not in r.stdout
            reports.append(dict(transport_pipeline=pipeline,passed=ok));assert ok,(pipeline,r.stdout)
        for device in ['gpu','cpu','default','nvidia','level_zero','arc','opencl','hip','native_cpu','']:
            c=dict(cfg);c['device']=device
            f=Path(tmp)/'case.yaml';f.write_text(yaml.safe_dump(c,sort_keys=False))
            r=subprocess.run([str(a.binary.resolve()),'--config',str(f)],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            ok=r.returncode!=0 and 'device must be cuda or intel' in r.stdout and '[unified-em-audit]' not in r.stdout
            reports.append(dict(device=device,passed=ok));assert ok,(device,r.stdout)
        for missing in [('primary_atomic_number',), ('primary_mass_number',), ('primary_atomic_number','primary_mass_number')]:
            c=dict(cfg)
            for key in missing:c.pop(key,None)
            f=Path(tmp)/'case.yaml';f.write_text(yaml.safe_dump(c,sort_keys=False))
            r=subprocess.run([str(a.binary.resolve()),'--config',str(f)],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            ok=r.returncode!=0 and 'Specify primary_atomic_number and primary_mass_number' in r.stdout
            reports.append(dict(missing=list(missing),passed=ok));assert ok,r.stdout
        for key in ['voxel_bins_z', 'voxel_size_z_mm']:
            c=dict(cfg);c.pop(key)
            f=Path(tmp)/'case.yaml';f.write_text(yaml.safe_dump(c,sort_keys=False))
            r=subprocess.run([str(a.binary.resolve()),'--config',str(f)],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            ok=r.returncode!=0 and 'Water geometry requires explicit voxel_bins_z and voxel_size_z_mm' in r.stdout
            reports.append(dict(missing=key,passed=ok));assert ok,r.stdout
        for key in ['voxel_bins_z', 'voxel_size_z_mm']:
            c=dict(cfg);c[key]=0
            f=Path(tmp)/'case.yaml';f.write_text(yaml.safe_dump(c,sort_keys=False))
            r=subprocess.run([str(a.binary.resolve()),'--config',str(f)],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            ok=r.returncode!=0 and '[unified-em-audit]' not in r.stdout
            reports.append(dict(key=key,value=0,passed=ok));assert ok,r.stdout
        # Compact is unconditional; the redundant selector is no longer accepted.
        for value in ['compact', 'combined']:
            c=yaml.safe_load((ROOT/'examples/carbon_water_minibeam.yaml').read_text())
            c['minibeam_primary_pipeline']=value
            f=Path(tmp)/'case.yaml';f.write_text(yaml.safe_dump(c,sort_keys=False))
            r=subprocess.run([str(a.binary.resolve()),'--config',str(f)],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            ok=r.returncode!=0 and 'Unsupported preset configuration key' in r.stdout and '[unified-em-audit]' not in r.stdout
            reports.append(dict(key='minibeam_primary_pipeline',value=value,passed=ok));assert ok,r.stdout
        # Secondary mean-loss defaults must not relax the primary requirement.
        c=dict(cfg);c.update(run_mode='production',enable_energy_straggling=False,
                            enable_secondary_energy_straggling=False)
        c.pop('all_ion_elastic_file',None);c.pop('elastic_recoil_stopping_file',None)
        f=Path(tmp)/'case.yaml';f.write_text(yaml.safe_dump(c,sort_keys=False))
        r=subprocess.run([str(a.binary.resolve()),'--config',str(f)],cwd=ROOT,env=env,
                         stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        ok=r.returncode!=0 and 'Production unified EM requires primary fluctuations' in r.stdout and '[unified-em-audit]' not in r.stdout
        reports.append(dict(check='production_primary_fluctuations_required',passed=ok));assert ok,r.stdout
        for flag in FLAGS:
            r=subprocess.run([str(a.binary.resolve()),flag,'1'],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            ok=r.returncode!=0 and 'Unknown argument' in r.stdout;reports.append(dict(flag=flag,passed=ok));assert ok,(flag,r.stdout)
    a.output.write_text(json.dumps(dict(passed=True,cases=reports),indent=2)+'\n');print('Passed',len(reports),'retired interface checks')
if __name__=='__main__':main()
