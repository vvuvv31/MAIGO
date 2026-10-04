#!/usr/bin/env python3
"""Fresh native two-material interface fixtures, not a GPU continuation model.

All tracks run in Geant4 without killing, RNG use by scoring, or geometry
clipping in analysis. Preserve full families and exact first-step/child states
for subsequent boundary-conditioned model work. This isolated planar fixture
is deliberately not presented as coverage of arbitrary CT corners/edges.
"""
import argparse, concurrent.futures, json, math, os, re, subprocess, sys, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmark/electron_response_model'))
from prototype import sha,save
from macro_model import DTYPE
BINARY=ROOT/'out/electron_standalone_20261003/native_build/topas'
CARDS=ROOT/'out/electron_standalone_20261003/birth_bank64'


def material(node):
    d=CARDS/f'node{node:02d}'
    identity=json.loads((d/'complete.json').read_text())['material_identity']
    text=(d/'topas.txt').read_text()
    name=re.search(r's:Ge/ROI/Material = "([^"]+)"',text)[1]
    return name,'\n'.join(l for l in text.splitlines() if ':Ma/' in l),identity


def audit(d,spec):
    path=d/'steps.phsp';assert path.stat().st_size%312==0
    r=np.memmap(path,DTYPE,mode='r');assert np.all(r['version']==4) and np.isin(r['pdg'],[11,22]).all()
    for key in ['pre','post','direction','post_direction','energy','remaining','dep','length']:
        assert np.isfinite(r[key]).all(),key
    assert np.all(r['dep']>=0) and np.all(r['remaining']>=0)
    order=np.lexsort((r['step'],r['track'],r['event'],r['run']))
    new=np.r_[True,(r['track'][order[1:]]!=r['track'][order[:-1]])|(r['event'][order[1:]]!=r['event'][order[:-1]])|(r['run'][order[1:]]!=r['run'][order[:-1]])]
    starts=np.flatnonzero(new);first=order[starts];last=order[np.r_[starts[1:]-1,len(order)-1]]
    assert np.all(r['step'][first]==1)
    roots=first[r['parent'][first]==0];assert len(roots)==spec['histories']
    assert np.array_equal(np.sort(r['event'][roots]),np.arange(spec['histories']))
    assert np.all(r['pdg'][roots]==11) and np.all(r['energy'][roots].astype(np.float32)==np.float32(spec['energy_MeV']))
    # TOPAS beam phase-space momentum is FP32 before G4 constructs the track;
    # an oblique rotation can shift kinetic energy within that same FP32 bin.
    # Audit the measured monoenergetic source, never replace it by requested KE.
    assert np.all(r['energy'][roots]==r['energy'][roots[0]])
    incident=float(r['energy'][roots[0]])
    expected=[0,0,-spec['distance_mm']]
    assert np.all(r['pre'][roots].astype(np.float32)==np.asarray(expected,dtype=np.float32))
    assert np.max(abs(r['direction'][roots,2]-math.cos(math.radians(spec['angle_deg']))))<2e-7
    # Use actual generating-step keys; ordinal indexing is invalid if a native
    # zero-state navigation step is omitted by the scorer.
    keys={(int(x['run']),int(x['event']),int(x['track']),int(x['step'])):i for i,x in enumerate(r)}
    assert len(keys)==len(r)
    children=first[r['parent'][first]!=0]
    parentrows=np.array([keys[(int(r[i]['run']),int(r[i]['event']),int(r[i]['parent']),int(r[i]['parent_step']))] for i in children],dtype=np.int64)
    assert np.max(abs(r['post'][parentrows]-r['pre'][children]),initial=0)<1e-6
    transfer=np.bincount(parentrows,weights=r['birth'][children],minlength=len(r))
    residual=r['energy']-r['remaining']-r['dep']-transfer
    assert np.all(abs(residual)<=1e-8+1e-10*r['energy'])
    before=order[:-1][~new[1:]];after=order[1:][~new[1:]]
    assert np.max(abs(r['remaining'][before]-r['energy'][after]),initial=0)<1e-10
    assert np.max(abs(r['post'][before]-r['pre'][after]),initial=0)<1e-9
    gaps=r['step'][after]-r['step'][before]
    bad=gaps!=1
    if bad.any():
        b=before[bad];a=after[bad]
        assert np.all(gaps[bad]==2)
        assert np.array_equal(r['post'][b],r['pre'][a]) and np.array_equal(r['post_direction'][b],r['direction'][a]) and np.array_equal(r['remaining'][b],r['energy'][a])
        assert np.all(np.any(np.isclose(abs(r['post'][b]),20,rtol=0,atol=1e-9),axis=1)|(abs(r['post'][b,2])<1e-9))
    # Identify the actual native material numbers by interior step locations.
    interior=np.all(abs(r['pre'])<20-1e-7,axis=1)
    left=interior&(r['pre'][:,2]<-1e-7);right=interior&(r['pre'][:,2]>1e-7)
    lm=np.unique(r['pre_material'][left]);rm=np.unique(r['pre_material'][right]);assert len(lm)==len(rm)==1 and lm[0]!=rm[0]
    inside=np.isin(r['pre_material'],[lm[0],rm[0]])
    maxstep=float(r['length'][inside&(r['pdg']==11)].max())
    assert maxstep<=spec['maximum_step_mm']*(1+1e-6)
    outside_dep=float(r['dep'][~inside].sum());assert outside_dep<1e-12*spec['histories']
    live=last[r['remaining'][last]>0]
    # Every terminal nonzero KE track must physically leave the scored
    # Envelope; no finite source endpoint is treated as a patient escape.
    assert np.all(np.max(abs(r['post'][live]),axis=1)>=100-1e-7)
    deposited=np.bincount(r['event'],weights=r['dep'],minlength=spec['histories'])
    escaped=np.bincount(r['event'][live],weights=r['remaining'][live],minlength=spec['histories'])
    closure=incident-deposited-escaped;assert abs(closure).max()<1e-8
    crossing=inside&np.isin(r['post_material'],[lm[0],rm[0]])&(r['pre_material']!=r['post_material'])
    assert np.max(abs(r['post'][crossing,2]),initial=0)<1e-7
    rootset=set(roots.tolist());child_of_first=np.array([p in rootset for p in parentrows])
    fields={f'first_{k}':np.asarray(r[k][roots]) for k in DTYPE.names}
    fields.update({f'child_{k}':np.asarray(r[k][children[child_of_first]]) for k in DTYPE.names})
    np.savez(d/'first_steps.npz',**fields,child_parent_rows=parentrows[child_of_first],source_rows=roots)
    energy=incident*spec['histories']
    report=dict(status='AUDITED_NATIVE_PLANAR_INTERFACE_DATA_NOT_GPU_ACCEPTANCE',rows=len(r),tracks=len(first),first_steps=len(roots),child_births=len(children),certified_zero_state_gaps=int(bad.sum()),maximum_step_mm=maxstep,max_step_energy_residual_MeV=float(abs(residual).max()),max_history_energy_residual_MeV=float(abs(closure).max()),outside_deposit_MeV=outside_dep,fixture_escape_fraction=float(escaped.sum()/energy),left_deposit_fraction=float(r['dep'][r['pre_material']==lm[0]].sum()/energy),right_deposit_fraction=float(r['dep'][r['pre_material']==rm[0]].sum()/energy),interface_crossings=int(crossing.sum()),first_step_crossing_fraction=float(crossing[roots].mean()),first_step_mean_loss_MeV=float((r['energy'][roots]-r['remaining'][roots]).mean()),first_step_mean_interface_tangential_displacement2_mm2=float(np.mean(np.sum((r['post'][roots,:2]-r['pre'][roots,:2])**2,axis=1))),raw_sha256=sha(path),first_steps_sha256=sha(d/'first_steps.npz'))
    report['actual_source_energy_MeV']=incident
    report['actual_source_position_mm']=r['pre'][roots[0]].tolist()
    report['actual_source_direction']=r['direction'][roots[0]].tolist()
    delta=r['post'][roots]-r['pre'][roots]
    lateral=delta-np.sum(delta*r['direction'][roots],axis=1)[:,None]*r['direction'][roots]
    report['first_step_mean_beam_lateral_displacement2_mm2']=float(np.mean(np.sum(lateral*lateral,axis=1)))
    return report


def one(task):
    d,spec=task;d.mkdir(parents=True,exist_ok=False)
    left,lc,li=material(spec['left_node']);right,rc,ri=material(spec['right_node']);assert left!=right
    text=f'''i:Ts/NumberOfThreads = 1
i:Ts/Seed = {spec['seed']}
b:Ts/PauseBeforeQuit = "False"
sv:Ph/Default/Modules = 1 "g4em-standard_opt4"
d:Ph/Default/CutForAllParticles = 0.05 mm
d:Ph/Default/EMRangeMax = 10 GeV
s:Ge/World/Material = "Vacuum"
d:Ge/World/HLX = 200 mm
d:Ge/World/HLY = 200 mm
d:Ge/World/HLZ = 200 mm
{lc}
{rc}
s:Ge/Envelope/Parent = "World"
s:Ge/Envelope/Type = "TsBox"
s:Ge/Envelope/Material = "Vacuum"
d:Ge/Envelope/HLX = 100 mm
d:Ge/Envelope/HLY = 100 mm
d:Ge/Envelope/HLZ = 100 mm
'''
    for name,mat,z in [('Left',left,-10),('Right',right,10)]:
        text+=f'''s:Ge/{name}/Parent = "Envelope"
s:Ge/{name}/Type = "TsBox"
s:Ge/{name}/Material = "{mat}"
d:Ge/{name}/HLX = 20 mm
d:Ge/{name}/HLY = 20 mm
d:Ge/{name}/HLZ = 10 mm
d:Ge/{name}/TransZ = {z} mm
d:Ge/{name}/MaxStepSize = {spec['maximum_step_mm']:.17g} mm
'''
    text+=f'''s:Ge/BeamPosition/Parent = "World"
s:Ge/BeamPosition/Type = "Group"
d:Ge/BeamPosition/TransX = 0 mm
d:Ge/BeamPosition/TransY = 0 mm
d:Ge/BeamPosition/TransZ = {-spec['distance_mm']:.17g} mm
d:Ge/BeamPosition/RotX = 0 deg
d:Ge/BeamPosition/RotY = {spec['angle_deg']:.17g} deg
d:Ge/BeamPosition/RotZ = 0 deg
s:So/Beam/Type = "Beam"
s:So/Beam/Component = "BeamPosition"
s:So/Beam/BeamParticle = "e-"
d:So/Beam/BeamEnergy = {spec['energy_MeV']:.17g} MeV
u:So/Beam/BeamEnergySpread = 0
s:So/Beam/BeamPositionDistribution = "None"
s:So/Beam/BeamAngularDistribution = "None"
i:So/Beam/NumberOfHistoriesInRun = {spec['histories']}
s:Sc/Steps/Quantity = "PrimaryElectronTransportNtuple"
s:Sc/Steps/Component = "Envelope"
s:Sc/Steps/OutputType = "binary"
s:Sc/Steps/OutputFile = "steps"
s:Sc/Steps/IfOutputFileAlreadyExists = "Exit"
b:Sc/Steps/RecordTransportState = "True"
b:Sc/Steps/PropagateToChildren = "True"
'''
    (d/'topas.txt').write_text(text)
    env=dict(os.environ,TOPAS_G4_DATA_DIR='/software/geant4-11.3.2/share/Geant4/data',LD_LIBRARY_PATH='/software/topas/lib:/software/geant4-11.3.2/lib:'+os.environ.get('LD_LIBRARY_PATH',''))
    start=time.perf_counter()
    with (d/'topas.log').open('x') as log:r=subprocess.run([str(BINARY),'topas.txt'],cwd=d,env=env,stdout=log,stderr=subprocess.STDOUT)
    run=dict(spec,left_material=li,right_material=ri,binary_sha256=sha(BINARY),input_sha256=sha(d/'topas.txt'),exit_code=r.returncode,seconds=time.perf_counter()-start)
    save(d/'run.json',run)
    log=(d/'topas.log').read_text();counts=re.findall(r'Particle source Beam: Total number of histories: (\d+)',log)
    assert r.returncode==0 and counts and int(counts[-1])==spec['histories'],d
    assert sum(map(float,re.findall(r'Total amount of energy not scored for this reason: ([\d.eE+-]+) MeV',log)))==0
    result=dict(run,audit=audit(d,spec));save(d/'report.json',result)
    print(d.name,json.dumps(result['audit']),flush=True);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--histories',type=int,default=1000);p.add_argument('--workers',type=int,default=3);p.add_argument('--max-step',type=float,default=.05);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    assert 0<a.max_step<=.25 and a.histories>0 and a.workers>0
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    cases=[]
    for pair,left,right in [('lung_soft',57,12),('soft_lung',12,57),('soft_bone',12,32),('bone_soft',32,12)]:
        for distance,angle in [(.001,0),(.01,80),(.049,90)]:
            name=f'{pair}_d{distance:g}_a{angle:g}'
            spec=dict(left_node=left,right_node=right,distance_mm=distance,angle_deg=angle,energy_MeV=.4,histories=a.histories,maximum_step_mm=a.max_step,seed=114150001+1009*len(cases))
            cases.append((out/name,spec))
    if a.smoke:cases=cases[:1]
    save(out/'inputs.json',dict(cases=[dict(directory=str(d),**s) for d,s in cases],script_sha256=sha(Path(__file__)),scope='Independent native planar controls; original Geant4 GS, no override; no dose scaling, clipping or child omission'))
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:results=list(pool.map(one,cases))
    save(out/'report.json',dict(status='NATIVE_INTERFACE_SCREEN_ONLY',results=results))
if __name__=='__main__':main()
