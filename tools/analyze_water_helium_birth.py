"""Compare primary-produced queued helium KE with first-tracked TOPAS products."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
from run_topas10x_gpu_benchmark import sha

def rows(path):
    with path.open() as f:
        for number,line in enumerate(f,1):
            if not line.strip() or line.lstrip().startswith('#'):continue
            values=[s.strip('"') for s in line.split()]
            if len(values)!=30 or values[0] not in ('interaction','product'):
                raise ValueError(f'Unexpected row {number}: {line[:200]}')
            yield values

def key(r):return tuple(int(r[i]) for i in (1,2,3,4))

def dose_identity(baseline,total):
    if baseline.shape!=total.shape or baseline.size==0:
        raise ValueError('Empty or mismatched dose shape')
    if not np.all(np.isfinite(baseline)) or not np.all(np.isfinite(total)):
        raise ValueError('Nonfinite dose')
    if np.any(baseline<0) or np.any(total<0) or baseline.sum()<=0:
        raise ValueError('Invalid dose')
    return dict(total_unchanged=bool(np.array_equal(baseline,total)),
        changed_voxels=int(np.count_nonzero(baseline!=total)),
        relative_integral_change=float(total.sum()/baseline.sum()-1),
        maximum_absolute_voxel_change=float(np.max(np.abs(total-baseline))))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--job',type=int,required=True)
    a=p.parse_args();root=a.out.resolve();m=json.loads((root/'manifest.json').read_text())
    status=subprocess.check_output(['sacct','-j',str(a.job),'--format=JobIDRaw,State,ExitCode','-n','-P'],text=True)
    if f'{a.job}|COMPLETED|0:0' not in status.splitlines():raise ValueError('Incomplete TOPAS')
    for name,h in m['pins'].items():
        if sha(Path(name))!=h:raise ValueError('Frozen input changed: '+name)
    header=(root/'cascade.header').read_text()
    for field in ['1: Record Kind','8: Parent Track ID','14: Particle Kinetic Energy (MeV)','30: Creator Model ID']:
        if field not in header:raise ValueError('Header layout mismatch')
    baseline=np.fromfile(m['baseline'],'<f8');total=np.fromfile(root/'dose_topas.bin','<f8')
    identity=dose_identity(baseline,total)
    if not identity['total_unchanged']:
        failure=dict(status='REJECTED_NONINVASIVENESS',job=a.job,**identity,
            conclusion='Birth comparison withheld; scorer side effects or run differences require investigation.',
            pins={str(f):sha(f) for f in [Path(m['baseline']),root/'dose_topas.bin',Path(__file__).resolve()]})
        with (root/'noninvasiveness_failure.json').open('x') as f:
            json.dump(failure,f,indent=2,allow_nan=False)
        raise ValueError('Ntuple changed total dose; failure report saved, no birth conclusion')
    interactions={};primary_count=0
    for r in rows(root/'cascade.phsp'):
        if r[0]!='interaction':continue
        k=key(r)
        if k in interactions:raise ValueError('Duplicate interaction key')
        interactions[k]=(int(r[5]),int(r[7])==0,int(r[10]),int(r[11]))
        primary_count+=int(r[7])==0
    energies={};all_energies={};seen=set();orphans=0
    for r in rows(root/'cascade.phsp'):
        if r[0]!='product' or int(r[10])!=2:continue
        k=key(r);identity=k+(int(r[6]),)
        if identity in seen:raise ValueError('Duplicate helium product row')
        seen.add(identity)
        if k not in interactions:orphans+=1;continue
        parent,primary,z,parent_a=interactions[k]
        if int(r[7])!=parent:raise ValueError('Product parent mismatch')
        if float(r[25])!=1:raise ValueError('Unexpected product weight')
        mass=int(r[11]);energy=float(r[13])
        if not np.isfinite(energy) or energy<0:raise ValueError('Invalid birth KE')
        all_energies.setdefault(mass,[]).append(energy)
        if primary:
            if (z,parent_a)!=(6,12):raise ValueError('Unexpected primary ion')
            energies.setdefault(mass,[]).append(energy)
    if orphans:raise ValueError('Orphan helium products')
    ledger_path=Path('/mnt/sda/wuwei/unified_water300_origin_20260907/energy_ledger.json')
    ledger=json.loads(ledger_path.read_text());layout=ledger['cinel02_species_transport_ledger_layout']
    gpu=np.array(ledger['cinel02_species_transport_ledger_MeV']).reshape(layout['shape'])
    n=m['histories'];results={}
    for mass,label in [(3,'3He'),(4,'4He'),(6,'6He')]:
        i=layout['species'].index(label);metrics=layout['metric']
        queued=gpu[i,metrics.index('queued_birth_kinetic')]
        imported=gpu[i,metrics.index('reaction_import_kinetic')]
        t=np.array(energies.get(mass,[]));ta=np.array(all_energies.get(mass,[]))
        results[label]=dict(topas_primary_first_tracked_count=len(t),topas_primary_KE_MeV_per_primary=float(t.sum()/n),
            topas_all_generations_count=len(ta),topas_all_generations_KE_MeV_per_primary=float(ta.sum()/n),
            gpu_primary_queued_KE_MeV_per_primary=float((queued-imported)/n),gpu_all_queued_KE_MeV_per_primary=float(queued/n),
            gpu_primary_over_topas=float((queued-imported)/t.sum()) if t.sum()>0 else None)
    result=dict(status='HELIUM_BIRTH_DIAGNOSTIC_NOT_PRODUCTION',job=a.job,total_unchanged=True,
        primary_interactions=primary_count,interactions=len(interactions),results=results,
        limitations=m['limitations'],pins={str(f):sha(f) for f in [root/'cascade.phsp',root/'dose_topas.bin',ledger_path,Path(__file__).resolve()]})
    with (root/'analysis.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k!='pins'},indent=2))

if __name__=='__main__':main()
