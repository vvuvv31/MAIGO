#!/usr/bin/env python3
"""Unthinned photon collision operator with explicit empty-process coverage.

Uniform native collision samples within each energy/process cell become an
energy-weighted CDF of all positive final branches. Full trajectories are not
part of the runtime artifact. This is a research discretization, not a claim
of exact continuous-energy transport or CT dose acceptance.
"""
import argparse,concurrent.futures,json,struct,sys
from pathlib import Path
import numpy as np
from extract_photon_response import DTYPE,sha,save
OUTCOME=np.dtype([('cdf','<f8'),('fraction','<f8'),('cosine','<f8'),('pdg','<i4'),('pad','<i4')]);assert OUTCOME.itemsize==32


def one(task):
    source,out,bins,rate_source=task;report=json.loads((source/'report.json').read_text());a=report['audit']
    conditional=(source/'conditional.phsp').exists();raw=source/('conditional.phsp' if conditional else 'photons.phsp')
    rates_path=(rate_source if conditional else source)/'photon_rates.csv'
    assert sha(raw)==a['raw_sha256']
    assert sha(rates_path)==(report['rate_source_sha256'] if conditional else a['rates']['sha256'])
    out.mkdir(parents=True,exist_ok=False);r=np.fromfile(raw,DTYPE);c=r[r['kind']==0];child=r[(r['kind']==1)&(r['remaining']>0)]
    # Place cells AROUND native source anchors, not on the anchor energies:
    # FP32 source rounding on a bin edge otherwise creates artificial gaps.
    anchors=np.geomspace(1e-5,.65,bins).astype(np.float32).astype(float)
    edges=np.r_[anchors[0],np.sqrt(anchors[:-1]*anchors[1:]),.65]
    assert np.all(c['energy']>=edges[0]) and np.all(c['energy']<=edges[-1])
    owner={(int(x['run']),int(x['event']),int(x['step'])):i for i,x in enumerate(c)}
    parent=np.array([owner[(int(x['run']),int(x['event']),int(x['step']))] for x in child],dtype=np.int64)
    groups=np.minimum(np.searchsorted(edges,c['energy'],side='right')-1,bins-1)*3+c['process']
    collision_cos=np.sum(c['pre']*c['post'],axis=1);child_cos=np.sum(child['pre']*child['post'],axis=1)
    assert np.max(abs(collision_cos),initial=0)<=1+1e-12 and np.max(abs(child_cos),initial=0)<=1+1e-12
    # Unit-direction roundoff only; no angular/model truncation.
    collision_cos=np.clip(collision_cos,-1,1);child_cos=np.clip(child_cos,-1,1)
    fractions=np.r_[c['deposit']/c['energy'],c['remaining']/c['energy'],child['remaining']/child['energy']]
    cosine=np.r_[np.zeros(len(c)),collision_cos,child_cos]
    pdg=np.r_[np.zeros(len(c),np.int32),np.full(len(c),22,np.int32),child['pdg']]
    cell=np.r_[groups,groups,groups[parent]]
    assert np.isfinite(fractions).all() and np.all(fractions>=0) and np.all(fractions<=1+1e-12)
    positive=fractions>0;fractions=fractions[positive];cosine=cosine[positive];pdg=pdg[positive];cell=cell[positive]
    order=np.argsort(cell,kind='stable');cell=cell[order];fractions=fractions[order];cosine=cosine[order];pdg=pdg[order]
    counts=np.bincount(cell,minlength=3*bins);offsets=np.r_[0,np.cumsum(counts)]
    native_count=np.bincount(groups,minlength=3*bins);weight=np.bincount(cell,weights=fractions,minlength=3*bins)
    assert np.all(abs(weight-native_count)<=1e-9*np.maximum(1,native_count))
    outcomes=np.zeros(len(cell),OUTCOME);outcomes['fraction']=fractions;outcomes['cosine']=cosine;outcomes['pdg']=pdg
    for i in range(3*bins):
        lo,hi=offsets[i:i+2]
        if hi>lo:
            cumulative=np.cumsum(fractions[lo:hi]);cumulative/=cumulative[-1];cumulative[-1]=1;outcomes['cdf'][lo:hi]=cumulative
    rates=np.loadtxt(rates_path,delimiter=',',skiprows=1)[:,:4].astype('<f8')
    channels=np.column_stack([offsets[:-1],counts]).astype('<u8')
    identity=a['material_identity'];rho=identity['runtime_density_g_cm3'];section=identity['section_id']
    path=out/'operator.bin'
    with path.open('xb') as f:
        f.write(struct.pack('<8sIidIIQ',b'MAIPHN01',1,section,rho,len(rates),bins,len(outcomes)))
        for arr in [rates,edges.astype('<f8'),channels,outcomes]:f.write(arr.tobytes())
    assert path.stat().st_size==40+32*len(rates)+8*(bins+1)+16*3*bins+32*len(outcomes)
    centres=np.sqrt(edges[:-1]*edges[1:]);pr=np.stack([np.interp(centres,rates[:,0],rates[:,k]) for k in range(1,4)],axis=1)
    total=pr.sum(axis=1);missing=np.sum(np.where(counts.reshape(-1,3)==0,pr,0),axis=1)/total
    # Certify support over every linear-rate interval split by channel edges.
    # Checking only bin centres misses a threshold inside an otherwise empty
    # process cell. Midpoints plus breakpoints cover positivity on each piece.
    breakpoints=np.unique(np.r_[rates[:,0],edges[(edges>=rates[0,0])&(edges<=rates[-1,0])]])
    probes=np.unique(np.r_[breakpoints,(breakpoints[:-1]+breakpoints[1:])/2])
    process_rates=np.stack([np.interp(probes,rates[:,0],rates[:,k]) for k in range(1,4)],axis=1)
    which=np.minimum(bins-1,np.searchsorted(edges,probes,side='right')-1)
    absent=(process_rates>0)&(counts.reshape(-1,3)[which]==0)
    missing_rows=np.flatnonzero(np.any(absent,axis=1))
    fine_missing=np.sum(np.where(absent,process_rates,0),axis=1)/process_rates.sum(axis=1)
    result=dict(status='RESEARCH_PHOTON_OPERATOR_SUPPORT_REQUIRES_FINE_ENERGY_AUDIT',conditional_native=conditional,node=report['node'],material_identity=identity,bins=bins,outcomes=len(outcomes),collisions=len(c),bytes=path.stat().st_size,sha256=sha(path),source_report_sha256=sha(source/'report.json'),source_raw_sha256=a['raw_sha256'],rates_sha256=sha(rates_path),generator_sha256=sha(Path(__file__)),empty_channels=int(np.count_nonzero(counts==0)),maximum_unrepresented_process_probability=float(missing.max()),missing_probability_by_energy_bin=missing.tolist(),scope='Native rate plus process-conditional energy-weighted final branch; all positive branches retained, no legacy trajectory. Finite energy bins and isotropic azimuth assumption require validation; an empty requested channel is coverage_missing, never deposition/escape.')
    result.update(coverage_complete=not len(missing_rows),support_probe_count=len(probes),fine_maximum_unrepresented_process_probability=float(fine_missing.max()),unsupported_rate_probes=[dict(energy_MeV=float(probes[i]),bin=int(which[i]),missing_processes=np.flatnonzero(absent[i]).tolist()) for i in missing_rows],support_scope='Complete only means no empty requested channel for the declared piecewise-linear attenuation table over10eV–0.65MeV; it does not certify physical interpolation accuracy')
    save(out/'manifest.json',result);print(report['node'],result['bytes'],result['empty_channels'],result['fine_maximum_unrepresented_process_probability'],flush=True);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--rate-source',type=Path);p.add_argument('--nodes',nargs='+',type=int,default=list(range(82)));p.add_argument('--bins',type=int,default=257);p.add_argument('--workers',type=int,default=4);a=p.parse_args();assert a.bins>1 and a.bins<=4096
    a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False)
    tasks=[(a.source.resolve()/f'node{i:02d}',a.out/f'node{i:02d}',a.bins,(a.rate_source or a.source).resolve()/f'node{i:02d}') for i in a.nodes]
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:results=list(pool.map(one,tasks))
    complete=all(x['coverage_complete'] for x in results)
    index=None
    if complete:
        index=a.out/'index.txt'
        index.write_text(f'MAIPHB01 {len(results)}\n'+''.join(f'{x["material_identity"]["section_id"]} {x["material_identity"]["runtime_density_g_cm3"]:.17g} node{x["node"]:02d}/operator.bin {x["sha256"]}\n' for x in results))
    save(a.out/'manifest.json',dict(status='RESEARCH_PHOTON_OPERATOR_BANK_NOT_CT_ACCEPTED',results=results,total_bytes=sum(x['bytes'] for x in results),coverage_complete=complete,index_sha256=sha(index) if index else None))
if __name__=='__main__':main()
