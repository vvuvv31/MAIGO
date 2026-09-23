from pathlib import Path
import csv, hashlib, json, re, shutil, subprocess
import yaml

R = Path('/mnt/sdb/wuwei/MAIGO_review_fix_20260923')
O = R / 'evidence/field3cm_emonly_20260923'
REF = Path('/mnt/sda/wuwei/minibeam_single_center_em_only_water75ev_e250_10m/topas_7378')
EXPECTED_GPU = '003249623e409d386c85c233c27286d57a8ffca15271acfd20aeede3050ab4f5'
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

for sub in ['inputs', 'cases', 'bin', 'results']:
    (O / sub).mkdir(exist_ok=True)
plan = O / 'inputs/spots_3cm.csv'
shutil.copy2('/mnt/sda/wuwei/minibeam_energy_scan_inputs/e250/spots_3cm.csv', plan)
beam = O / 'inputs/beam_model.csv'
shutil.copy2('/mnt/sda/wuwei/minibeam_phase_scan_e250_6255_1/beam_model.csv', beam)
rows = list(csv.DictReader(plan.open()))
axis = list(range(-15, 16, 2))
assert len(rows) == 256
assert {(float(v['x']), float(v['y'])) for v in rows} == {(x,y) for x in axis for y in axis}
assert all(float(v['energy']) == 3000 and float(v['weight']) == 100000 for v in rows)
assert [int(v['spot_id']) for v in rows] == list(range(256))
base = yaml.safe_load((R / 'config/review_fullchain_topas_matched_1m.yaml').read_text())
optics = list(csv.DictReader(beam.open()))[0]
for key, bk in [('emittance_sigma_x_mm','sigma_x_mm'),('emittance_sigma_x_prime','sigma_xp_rad'),
                ('emittance_correlation_x','corr_x'),('emittance_sigma_y_mm','sigma_y_mm'),
                ('emittance_sigma_y_prime','sigma_yp_rad'),('emittance_correlation_y','corr_y')]:
    assert base[key] == float(optics[bk])
assert base['beam_energy_spread'] == float(optics['energy_spread_percent']) / 100
assert base['initial_energy_MeVu'] * base['primary_mass_number'] == float(optics['energy'])
assert base['minibeam_copper_max_step_mm'] == base['minibeam_water_primary_urban_max_step_mm'] == .05
assert not base['enable_inelastic'] and not base['minibeam_copper_enable_nuclear_attenuation']
assert not base['minibeam_copper_enable_elastic']
cache = (R / 'build/oneapi-nvidia-minibeam/CMakeCache.txt').read_text()
for required in ['CARBON_DOSE_FP32:BOOL=ON','CARBON_DOSE_FP64:BOOL=OFF','CARBON_DISABLE_INTEGRITY_CHECKS:BOOL=OFF']:
    assert required in cache
assert sha(R / 'build/oneapi-nvidia-minibeam/carbon_mc') == EXPECTED_GPU
shutil.copy2(R / 'build/oneapi-nvidia-minibeam/carbon_mc', O / 'bin/carbon_mc')

common = '''includeFile = run_energy_scan.txt
s:Ge/Box/Material = "Water_75eV"
sv:Ph/Default/Modules = 2 "g4em-standard_opt4" "g4decay"
d:Ph/Default/CutForAllParticles = 0.05 mm
d:Ph/Default/EMRangeMax = 6 GeV
b:Ts/PauseBeforeQuit = "False"
b:So/CarbonPBS/DijMode = "False"
s:So/CarbonPBS/WeightMode = "Histories"
s:So/CarbonPBS/SpotCoordinateConvention = "TPS"
s:Sc/DoseAtPhantomP/IfOutputFileAlreadyExists = "Exit"
i:Ts/ShowHistoryCountAtInterval = 1000000
'''

def topas_case(name, scale, seed, threads=128, plan_path=plan, beam_path=beam, extra=''):
    d = O / 'cases' / name
    d.mkdir(exist_ok=False)
    for fn in ['run1.txt','run_energy_scan.txt','aperture.txt']:
        s = (REF / fn).read_text()
        if fn == 'run1.txt':
            s = re.sub(r'^sv:Ph/Default/Modules\s*=.*$', 'sv:Ph/Default/Modules = 2 "g4em-standard_opt4" "g4decay"', s, flags=re.M)
            # GPU has one homogeneous water volume. Remove the TOPAS-only
            # same-water daughter and its otherwise unmatched interfaces.
            s = re.sub(r'^[a-z]+:Ge/Box2/[^\n]*\n?', '', s, flags=re.M)
        (d / fn).write_text(s)
    s = common + f's:So/CarbonPBS/SpotPlanFile = "{plan_path}"\ns:So/CarbonPBS/BeamModelFile = "{beam_path}"\nu:So/CarbonPBS/HistoriesScale = {scale}\ni:Ts/Seed = {seed}\ni:Ts/NumberOfThreads = {threads}\n' + extra
    (d / 'run.txt').write_text(s)
    return str(d)

def gpu_case(name, scale, seed):
    d = O / 'cases' / name
    d.mkdir(exist_ok=False)
    cfg = dict(base)
    cfg.update(tps_spots_file=str(plan), tps_histories_scale=scale,
               number_of_histories=int(25600000*scale), random_seed=seed)
    cp = d / (name + '.yaml')
    cp.write_text(yaml.safe_dump(cfg, sort_keys=False))
    assert not (R / 'out' / name).exists()
    return str(cp)

cases = {}
for i in range(1, 5):
    cases[f'topas_b{i}'] = topas_case(f'topas_b{i}', .125, 2026101000+i)
    cases[f'gpu_b{i}'] = gpu_case(f'field3cm_emonly_b{i}', .125, 2026102000+i)
cases['topas_smoke'] = topas_case('topas_smoke', .001, 2026101000)
cases['gpu_smoke'] = gpu_case('field3cm_emonly_smoke', .001, 2026102000)

# Deterministic source audit: nonuniform history counts exercise prefix boundaries.
audit_plan = O / 'inputs/source_audit_spots.csv'
weights = [100 + i % 13 for i in range(256)]
with audit_plan.open('w', newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
    for row, n in zip(rows,weights):w.writerow(dict(row,weight=n))
audit_beam = O / 'inputs/source_audit_beam.csv'
audit_beam.write_text('energy,sigma_x_mm,sigma_xp_rad,corr_x,sigma_y_mm,sigma_yp_rad,corr_y,energy_spread_percent\n3000,0,0,0,0,0,0,0\n')
extra='''s:Ge/World/Material = "Vacuum"
s:Ge/SourceAudit/Type = "TsBox"
s:Ge/SourceAudit/Parent = "World"
s:Ge/SourceAudit/Material = "Vacuum"
d:Ge/SourceAudit/HLX = 100 mm
d:Ge/SourceAudit/HLY = 0.1 mm
d:Ge/SourceAudit/HLZ = 100 mm
d:Ge/SourceAudit/TransY = -449 mm
s:Sc/SourceAudit/Quantity = "PhaseSpace"
s:Sc/SourceAudit/Surface = "SourceAudit/YMinusSurface"
s:Sc/SourceAudit/OnlyIncludeParticlesGoing = "In"
sv:Sc/SourceAudit/OnlyIncludeParticlesNamed = 1 "GenericIon(6,12,*)"
s:Sc/SourceAudit/OutputType = "ASCII"
s:Sc/SourceAudit/OutputFile = "source_audit"
s:Sc/SourceAudit/IfOutputFileAlreadyExists = "Exit"
i:Sc/SourceAudit/OutputBufferSize = 100000
b:Sc/SourceAudit/IncludeRunID = "True"
b:Sc/SourceAudit/IncludeEventID = "True"
b:Sc/SourceAudit/IncludeTrackID = "True"
b:Sc/SourceAudit/IncludeParentID = "True"
b:Sc/SourceAudit/KillAfterPhaseSpace = "True"
'''
cases['source_audit'] = topas_case('source_audit',1,2026100901,128,audit_plan,audit_beam,extra)

manifest = dict(
    extension_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=O/'topas_tps_source_extension',text=True).strip(),
    gpu_binary_sha256=EXPECTED_GPU, topas_version='4.2.3', geant4_version='11.3.2',
    cases=cases, histories_per_batch=3200000, batches=4, histories_per_engine=12800000,
    plan=dict(spots=256,axis_centers_mm=axis,spacing_mm=2,energy_MeVu=250,energy_total_MeV=3000,
              per_spot_histories_per_batch=12500,per_spot_histories_total=50000),
    source_audit=dict(histories=sum(weights),weights=weights,threads=128,plane_world_y_mm=-449.1),
    matched=dict(materials=['Copper','Air','Water_75eV'],water_I_eV=75,nuclear_reactions=False,
                 copper_max_step_mm=.05,water_urban_max_step_mm=.05,production_cut_mm=.05,
                 water_dimensions_mm=[100,100,250],water_internal_daughter_volumes=False,grid_shape_depth_y_x=[1000,1,1000],
                 grid_spacing_x_y_depth_mm=[.1,100,.25],beam_optics=optics,
                 source_to_isocenter_mm=450,virtual_magnets_mm=[6228.28,7007.64],
                 slit_width_mm=.5,slit_pitch_mm=3.6,slits=15,copper_thickness_mm=60,
                 copper_radius_mm=60,water_entrance_world_y_mm=60,physical_dose_normalization='Gy per original source history'),
    implementation_limits=['GPU local electron deposition remains the established main comparison mode; TOPAS transports electrons explicitly.',
                           'GPU upstream/slit air uses mean energy loss without air MSC.',
                           'GPU terminal tracking cutoff is 0.1 MeV total; TOPAS has no identical total-ion cutoff in this input.',
                           'Same physical input parameters do not imply identical approximate transport algorithms; this run measures that discrepancy.'],
    files=[])
for p in sorted(list((O/'inputs').iterdir()) + [p for p in (O/'cases').rglob('*') if p.is_file()]):
    manifest['files'].append(dict(path=str(p),sha256=sha(p)))
(O/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

for name,cpus,mem,limit,extra_sbatch,command in [
    ('preflight',128,'12G','00:20:00','',f'python3 {O}/run_field.py preflight'),
    ('topas',128,'12G','04:00:00','#SBATCH --array=1-4\n',f'python3 {O}/run_field.py topas "$SLURM_ARRAY_TASK_ID"'),
    ('gpu',8,'12G','03:00:00','',f'python3 {O}/run_field.py gpu'),
    ('compare',2,'4G','00:15:00','',f'python3 {O}/analyze_field.py')]:
    script=f'''#!/bin/bash
#SBATCH --job-name=field3cm_{name}
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={limit}
#SBATCH --output={O}/logs/{name}_%A_%a.log
#SBATCH --error={O}/logs/{name}_%A_%a.err
{extra_sbatch}set -eo pipefail
source /software/env_topas.sh
export LD_LIBRARY_PATH=/home/wuwei/sycl_workspace/llvm/build/install/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
{command}
'''
    (O / (name+'.sbatch')).write_text(script)
print(json.dumps(dict(prepared=str(O),histories_per_engine=12800000,source_audit_histories=sum(weights))),flush=True)
