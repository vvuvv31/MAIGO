from pathlib import Path
import shutil,subprocess,json,time,re
r=Path('/mnt/sdb/wuwei/MAIGO_review_fix_20260923');o=r/'evidence/percent1_20260923';ref=Path('/mnt/sda/wuwei/minibeam_single_center_em_only_water75ev_e250_10m/topas_7378')
for label,n,threads in [('phase_smoke',1000,16),('topas_phase_2m',2000000,128)]:
 d=o/label;d.mkdir(exist_ok=False)
 for f in ['run_energy_scan.txt','run1.txt','aperture.txt','beam_model_single_center_e250.csv']:shutil.copyfile(ref/f,d/f)
 (d/'spots_single_center.csv').write_text(f'spot_id,x,y,energy,weight\n0,0,0,3000,{n}\n')
 s=(ref/'run_single_center_em_only_water75ev_e250_10m.txt').read_text().replace('spots_single_center_e250_10m.csv','spots_single_center.csv').replace('i:Ts/NumberOfThreads = 128',f'i:Ts/NumberOfThreads = {threads}')
 s+='''
i:Ts/Seed = 2026100201
s:Sc/DoseAtPhantomP/IfOutputFileAlreadyExists = "Exit"
s:Sc/AtWaterEntrance/Quantity = "PhaseSpace"
s:Sc/AtWaterEntrance/Surface = "Box/YMinusSurface"
s:Sc/AtWaterEntrance/OnlyIncludeParticlesGoing = "In"
sv:Sc/AtWaterEntrance/OnlyIncludeParticlesNamed = 1 "GenericIon(6,12,*)"
s:Sc/AtWaterEntrance/OutputType = "ASCII"
s:Sc/AtWaterEntrance/OutputFile = "water_entrance"
s:Sc/AtWaterEntrance/IfOutputFileAlreadyExists = "Exit"
i:Sc/AtWaterEntrance/OutputBufferSize = 100000
b:Sc/AtWaterEntrance/IncludeRunID = "True"
b:Sc/AtWaterEntrance/IncludeEventID = "True"
b:Sc/AtWaterEntrance/IncludeTrackID = "True"
b:Sc/AtWaterEntrance/IncludeParentID = "True"
b:Sc/AtWaterEntrance/KillAfterPhaseSpace = "True"
'''
 (d/'run.txt').write_text(s);start=time.monotonic()
 with (d/'run.log').open('x') as f:p=subprocess.run(['bash','-c','source /software/env_topas.sh && exec /software/topas/bin/topas run.txt'],cwd=d,stdout=f,stderr=subprocess.STDOUT)
 if p.returncode:print((d/'run.log').read_text()[-4500:],flush=True);raise RuntimeError(label)
 h=(d/'water_entrance.header').read_text();assert f'Number of Original Histories: {n}\n' in h
 import numpy as np
 a=np.loadtxt(d/'water_entrance.phsp',ndmin=2)
 assert len(a)>0 and a.shape[1]==14 and np.all(abs(a[:,1]*10-60)<.0001) and np.all(a[:,5]>0) and np.all(a[:,13]==0)
 q={'histories':n,'seed':2026100201,'threads':threads,'scored_primary_c12':len(a),'wall_s':time.monotonic()-start,'surface_world_Y_mm':60,'scope':'Entrance only: C12 killed after phase recording; dose maps are NOT full-chain references. No EventID-to-GPU-history matching.'}
 (d/'quality.json').write_text(json.dumps(q,indent=2)+'\n');print(label,json.dumps(q),flush=True)
