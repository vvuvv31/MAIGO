from pathlib import Path
import csv,json,hashlib,random,yaml
root=Path('/mnt/sdb/wuwei/MAIGO_review_fix_20260923')
out=root/'evidence/review_fix_20260923'
p=root/'evidence/urban_d51e599_20260923/dose_gate.py'
s=p.read_text();old='    if left_below == peak or right_below == peak:'
assert old in s
s=s.replace(old,'    if profile[left_below] >= half or profile[right_below] >= half:\n        return None  # both half-height crossings must be bracketed\n'+old)
p.write_text(s)

N=20000
rng=random.Random(2026092301)
xs=[rng.uniform(-.25,.25) for _ in range(N)]
source=out/'water_beamlet_source'
with source.with_suffix('.csv').open('w') as f:
 w=csv.writer(f)
 w.writerow('spot_id,energy_MeV,x_mm,y_mm,weight,source_x_mm,source_y_mm,source_z_mm,direction_x,direction_y,direction_z,energy_spread_percent,sigma_x_mm,sigma_y_mm,sigma_x_prime,sigma_y_prime,correlation_x,correlation_y'.split(','))
 for i,x in enumerate(xs): w.writerow([i+1,3000,0,0,1,x,0,.0001,0,0,1,0,0,0,0,0,0,0])
with source.with_suffix('.phsp').open('w') as f:
 for x in xs: f.write(f'{x/10:.15g} 6.00001 0 0 1 3000 1 1000060120 0 1\n')
source.with_suffix('.header').write_text(f'''TOPAS ASCII Phase Space

Number of Original Histories: {N}
Number of Original Histories that Reached Phase Space: {N}
Number of Scored Particles: {N}

Columns of data are as follows:
 1: Position X [cm]
 2: Position Y [cm]
 3: Position Z [cm]
 4: Direction Cosine X
 5: Direction Cosine Y
 6: Energy [MeV]
 7: Weight
 8: Particle Type (in PDG Format)
 9: Flag to tell if Third Direction Cosine is Negative (1 means true)
10: Flag to tell if this is the First Scored Particle from this History (1 means true)
''')
base=yaml.safe_load((root/'config/beam_minibeam_water_replay_e250_em12800k_urban_v2_mini.yaml').read_text())
for seed in [1,2,3]:
 cfg=dict(base)
 cfg.update(number_of_histories=N,phantom_length_mm=150.0,maximum_step_mm=.05,
   minibeam_water_primary_urban_max_step_mm=.05,minibeam_water_primary_stopping_power_scale=1.0,
   minibeam_copper_enable_nuclear_attenuation=False,
   minibeam_water_urban_loss_range_file=str(root/'evidence/urban_d51e599_20260923/active_water75_cut005_r1.csv'),
   tps_spots_file=str(source.with_suffix('.csv')),random_seed=2026092300+seed)
 cfg.pop('minibeam_water_primary_plane_depths_mm',None)
 cfg.pop('minibeam_water_primary_plane_output_file',None)
 # Input paths are absolute so the saved config is independently auditable.
 for k,v in list(cfg.items()):
  if isinstance(v,str) and v.startswith('data/'): cfg[k]=str(root/v)
 (root/f'config/review_water_beamlet_s{seed}.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
 run=out/f'topas_water_s{seed}';run.mkdir(exist_ok=True)
 text=f'''i:Ts/Seed = {2026092400+seed}
i:Ts/NumberOfThreads = 16
i:Ts/ShowHistoryCountAtInterval = 5000
b:Ts/PauseBeforeQuit = "False"
b:Gr/Enable = "False"
s:Ge/World/Type = "TsBox"
s:Ge/World/Material = "G4_Galactic"
d:Ge/World/HLX = 200 mm
d:Ge/World/HLY = 300 mm
d:Ge/World/HLZ = 200 mm
b:Ge/World/Invisible = "True"
s:Ge/Water/Parent = "World"
s:Ge/Water/Type = "TsBox"
s:Ge/Water/Material = "Water_75eV"
d:Ge/Water/HLX = 50 mm
d:Ge/Water/HLY = 75 mm
d:Ge/Water/HLZ = 50 mm
d:Ge/Water/TransY = 135 mm
d:Ge/Water/MaxStepSize = 0.05 mm
i:Ge/Water/XBins = 1000
i:Ge/Water/YBins = 600
i:Ge/Water/ZBins = 1
s:So/Carbon/Type = "PhaseSpace"
s:So/Carbon/Component = "World"
s:So/Carbon/PhaseSpaceFileName = "{source}"
b:So/Carbon/PhaseSpaceIgnoreXPos = "False"
b:So/Carbon/PhaseSpaceIgnoreYPos = "False"
b:So/Carbon/PhaseSpaceIgnoreZPos = "False"
b:So/Carbon/PhaseSpaceIncludeEmptyHistories = "False"
i:So/Carbon/PhaseSpaceMultipleUse = 1
s:Ph/ListName = "Default"
s:Ph/Default/Type = "Geant4_Modular"
sv:Ph/Default/Modules = 2 "g4em-standard_opt4" "g4decay"
d:Ph/Default/EMRangeMax = 10 GeV
d:Ph/Default/CutForAllParticles = 0.05 mm
s:Sc/Dose/Quantity = "DoseToMedium"
s:Sc/Dose/Component = "Water"
s:Sc/Dose/OutputType = "Binary"
s:Sc/Dose/OutputFile = "{run}/dose"
s:Sc/Dose/IfOutputFileAlreadyExists = "Exit"
'''
 (run/'input.txt').write_text(text)
meta={'purpose':'first corrected EM-only no-Cu matched-input comparison; not full-chain promotion',
      'histories_per_seed':N,'seeds':3,'particle':'C12','energy_total_MeV':3000,
      'source_width_mm':.5,'source_y_mm':0,'source_depth_mm':.0001,
      'material':'Water_75eV','cut_mm':.05,'step_ceiling_mm':.05,
      'grid_xyz':[1000,1,600],'spacing_xyz_mm':[.1,100,.25],
      'source_csv_sha256':hashlib.sha256(source.with_suffix('.csv').read_bytes()).hexdigest(),
      'source_phsp_sha256':hashlib.sha256(source.with_suffix('.phsp').read_bytes()).hexdigest()}
(out/'comparison_protocol.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(meta,indent=2))
