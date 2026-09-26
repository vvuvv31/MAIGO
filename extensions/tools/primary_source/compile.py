#!/usr/bin/env python3
"""Compile audited proton raw data into a CINPKG04/SCHNRATE bundle.

No carbon event conversion, endpoint aliasing, or invented nonzero channels.
The coverage report is written before refusing unsupported positive-rate nodes.
"""
import argparse, collections, datetime, hashlib, json, math, struct, subprocess, sys, uuid
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/'extensions/package_tools'))
import cinel02,cinel03
from campaign import sha,dump
TARGETS=[1,6,7,8,12,15,16,17,18,20,11,19,22]
GRID=dict(emin=.1,emax=460.1,step=.5,count=921)

def collect(roots):
    events=[];pins={};models=collections.Counter();nulls=[]
    for root in roots:
        manifest=json.loads((root/'campaign.json').read_text());pins[str(root/'campaign.json')]=sha(root/'campaign.json')
        for t in manifest['tasks']:
            case=root/t['case'];status=json.loads((case/'status.json').read_text())
            if status['returncode'] or sha(case/'run.txt')!=t['input_sha256']:raise ValueError(f'Failed/changed run: {case}')
            pins[str(case/'status.json')]=sha(case/'status.json')
            pins[str(case/'run.txt')]=sha(case/'run.txt')
            for contract in case.rglob('*.contract.json'):
                pins[str(contract)]=sha(contract)
            log=(case/'topas.log').read_text()
            if 'FatalException' in log:raise ValueError(f'Fatal TOPAS log: {case}')
            count=0
            for fn,h in status['raw_pins'].items():
                raw=root/fn
                if sha(raw)!=h:raise ValueError(f'Raw hash changed: {raw}')
                pins[str(raw)]=h
                for r,ps in cinel02.read_raw(raw):
                    if (r['projectile_z'],r['projectile_a'],r['target_z'])!=(1,1,t['target_z']):raise ValueError('Raw projectile/target mismatch')
                    if r['process_name']!='protonInelastic' or r['model_name']!='Binary Cascade':raise ValueError('Wrong proton process/model')
                    cinel02._validate_record(r,ps,single_target=None)
                    vals=[r['collision_energy_MeV'],r['parent_energy_MeV'],r['process_local_deposit_MeV']]+[p['kinetic_energy_MeV'] for p in ps]
                    if not all(math.isfinite(v) and v>=0 for v in vals):raise ValueError('Invalid energy')
                    # Products include the unsupported subtotal: never count it twice.
                    if sum(vals[1:])>vals[0]+max(200,.2*vals[0]):raise ValueError('Raw energy closure failure')
                    events.append((r,ps));models[r['model_name']]+=1;count+=1
            if count==0:nulls.append(dict(target_z=t['target_z'],energy=t['energy_MeV'],histories=t['histories'],case=str(case)))
    return events,pins,models,nulls

def write_package(events,out,pins):
    events.sort(key=lambda it:(it[0]['target_z'],it[0]['collision_energy_MeV_per_u']))
    pkg=cinel03.Cinel03Package();pkg.minimum_energy_MeV_per_u=0.0;pkg.energy_bin_width_MeV_per_u=1.0
    pkg.minimum_events_per_bin=1;pkg.campaign_uuid=str(uuid.uuid5(uuid.NAMESPACE_URL,json.dumps(pins,sort_keys=True)))
    by_target=collections.defaultdict(list);cell_start=0;previous=None;node=None
    for i,(r,ps) in enumerate(events):
        e=float(r['collision_energy_MeV_per_u']);key=(1,1,r['target_z'],int(e));nk=(1,1,r['target_z'],e)
        if key!=previous:
            if previous is not None:pkg.cells[-1]['interaction_count']=i-cell_start
            pkg.cells.append(dict(projectile_z=1,projectile_a=1,target_element_z=key[2],energy_bin=key[3],
                interaction_offset=i,interaction_count=0,energy_lower_MeV_per_u=float(key[3]),energy_upper_MeV_per_u=float(key[3]+1)))
            previous=key;cell_start=i
        if nk!=node:
            pkg.energy_nodes.append(nk);pkg.event_offsets.append(i);by_target[r['target_z']].append(e);node=nk
        pkg.interactions.append(cinel02._pack_fixed(r));pkg.products.extend(cinel02._pack_product(p) for p in ps)
    if not events:raise ValueError('No proton interactions')
    pkg.cells[-1]['interaction_count']=len(events)-cell_start
    pkg.event_offsets.append(len(events));pkg.event_indices=list(range(len(events)))
    path=out/'proton_cinel03.bin';tmp=path.with_suffix('.tmp');pkg.write_binary(tmp)
    reread=cinel03.Cinel03Package.read_binary(tmp)
    if len(reread.interactions)!=len(events):raise ValueError('Package roundtrip failed')
    tmp.replace(path)
    channels=[]
    for z in TARGETS:
        es=by_target[z]
        channels.append(dict(projectile_z=1,projectile_a=1,target_element_z=z,has_support=bool(es),
            energy_min_MeV_per_u=min(es,default=0),energy_max_MeV_per_u=max(es,default=0),
            energy_nodes=len(es),maximum_observed_node_gap_MeV_per_u=max(np.diff(es),default=0),
            maximum_allowed_node_gap_MeV_per_u=5.0,interpolation_policy='stochastic_bracketing',target_alias_allowed=False))
    dump(out/'proton_cinel03.channels.json',dict(channels=channels))
    return path,channels,by_target

def rates(raw,channels,nodes,out,nulls):
    d=json.loads(raw.read_text())
    if (d['projectile_z'],d['projectile_a'])!=(1,1) or d['process_provenance']['process_name']!='protonInelastic':raise ValueError('XS projectile/process mismatch')
    partial=np.zeros((25,13,921),dtype='<f8');grid=.1+np.arange(921)*.5
    if len(d['sections'])!=25:raise ValueError('Missing XS sections')
    for si,sec in enumerate(d['sections']):
        if len(sec['grid'])!=921:raise ValueError('Wrong XS energy grid')
        for j,pt in enumerate(sec['grid']):
            if abs(pt['energy_mevu']-grid[j])>1e-8:raise ValueError('Wrong XS energy')
            found=set()
            for el in pt['elements']:
                z=el['target_z'];found.add(z);v=el['mass_partial_per_mm_at_1g_cm3']
                if not math.isfinite(v) or v<0:raise ValueError('Invalid XS')
                partial[si,TARGETS.index(z),j]=v
    gaps=[];uncovered=[]
    for ti,c in enumerate(channels):
        z=c['target_element_z'];lo=c['energy_min_MeV_per_u'];hi=c['energy_max_MeV_per_u'];es=nodes[z]
        for a,b in zip(es,es[1:]):
            if b-a>5:gaps.append(dict(target_z=z,low=a,high=b))
        for j,e in enumerate(grid):
            if e>250:break
            if np.max(partial[:,ti,j])>0 and (not es or e<lo or e>hi):
                uncovered.append(dict(target_z=z,energy=float(e),max_mass_rate=float(np.max(partial[:,ti,j]))))
    report=dict(gaps=gaps,positive_rate_without_events=uncovered,null_observations=nulls,
                policy='No-event histories alone do not prove zero cross section',pass_coverage=not gaps and not uncovered)
    dump(out/'coverage.json',report)
    if gaps or uncovered:raise ValueError(f'Coverage incomplete: {len(gaps)} gaps; {len(uncovered)} positive-rate nodes outside event domains; see coverage.json')
    total=np.zeros((25,921),dtype='<f8');domains=[]
    for ti,c in enumerate(channels):
        lo=c['energy_min_MeV_per_u'];hi=c['energy_max_MeV_per_u'];has=c['has_support']
        mask=(grid>=lo)&(grid<=hi) if has else np.zeros(921,dtype=bool)
        partial[:,ti,~mask]=0
        total+=partial[:,ti,:]
        domains.append(dict(target_z=TARGETS[ti],energy_min_mevu=lo,energy_max_mevu=hi,has_support=has))
    path=out/'proton_rates.bin'
    with open(path,'wb') as f:
        f.write(struct.pack('<8sIIII3d13i',b'SCHNRATE',3,25,13,921,.1,460.1,.5,*TARGETS))
        f.write(partial.tobytes());f.write(total.tobytes())
        for dom in domains:f.write(struct.pack('<ddB7x',dom['energy_min_mevu'],dom['energy_max_mevu'],dom['has_support']))
    meta=dict(schema_version=1,projectile=dict(z=1,a=1),data_filename=path.name,data_sha256=sha(path),
        binary_magic='SCHNRATE',binary_version=3,target_order=TARGETS,energy_grid=GRID,channel_domains=domains,
        physics_list='g4em-standard_opt4 + QGSP_BIC_HP + g4ion-inclxx',topas_version='4.2.p3',geant4_version='11.3.2',
        schneider_sha256=sha(REPO/'data/HUtoMaterialSchneider.txt'),compiler_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        raw_inputs={str(raw):sha(raw)},source_energy_scope_MeV=[.1,250])
    dump(path.with_suffix('.metadata.json'),meta)
    return path

def bundle(out,primary,rate):
    shared=out/'shared'
    old=json.loads((REPO/'extensions/reference/schneider_physics_bundle_v2_1.json').read_text())
    b=dict(old);b['bundle_name']='proton_therapy_0p1_250_20260926';b['primary_projectile']={'z':1,'a':1}
    b['source_energy_scope_MeV']=[.1,250];b['physics_list_primary']='QGSP_BIC_HP protonInelastic Binary Cascade'
    for role,p in [('primary_rate',rate),('primary_package',primary),('secondary_rate',shared/'secondary_inelastic_rates_v2_1.bin'),
                   ('secondary_package',shared/'cinel03_secondary_targets_v2_1_14p.bin'),('stopping_table',out/'proton_schneider_stopping_v1.bin')]:
        b[role]=dict(file=str(p.resolve()),sha256=sha(p))
        if 'package' in role:
            ch=p.with_suffix('.channels.json');b[role].update(channels_file=str(ch.resolve()),channels_sha256=sha(ch))
    b['generation_timestamp']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    dump(out/'physics_bundle.json',b)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,nargs='+',required=True)
    p.add_argument('--rate-json',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    events,pins,models,nulls=collect(a.campaign);print('Audited interactions',len(events),flush=True)
    primary,channels,nodes=write_package(events,a.output,pins)
    dump(primary.with_suffix('.metadata.json'),dict(schema_version=1,format='CINPKG04',data_filename=primary.name,
         data_sha256=sha(primary),projectile=dict(z=1,a=1),models=dict(models),raw_pins=pins,
         interactions=len(events),source_energy_scope_MeV=[.1,250],compiler_sha256=sha(Path(__file__))))
    rate=rates(a.rate_json,channels,nodes,a.output,nulls);bundle(a.output,primary,rate)
    print('Compiled',a.output/'physics_bundle.json')
if __name__=='__main__':main()
