#!/usr/bin/env python3
"""Independent electron material operator; all positive hard branches retained.

This is an explicitly discretized research model, not native continuous-energy
transport equivalence or CT acceptance. No legacy trajectory files are read.
"""
import argparse, concurrent.futures, json, struct
from pathlib import Path
import numpy as np
from extract_photon_response import DTYPE, sha, save
from compile_photon_response import OUTCOME
from electron_hard_rates import witnesses
BASE=Path(__file__).resolve().parents[1]/'out/electron_standalone_20261003'


def one(task):
    node,source,supplement,parameters,elastic,urban,out=task
    name=f'node{node:02d}';par=parameters/name
    meta=json.loads((par/'report.json').read_text());identity=meta['material_identity']
    paramfile=par/'boundary_parameters.csv';assert sha(paramfile)==meta['table_sha256']
    rawpar=np.loadtxt(paramfile,delimiter=',',skiprows=1)
    assert rawpar.shape==(257,12) and np.isfinite(rawpar).all()
    # Signed coarse knots retained for diagnostic comparison; runtime hard
    # rates use the separately reconstructed native cubic intervals below.
    rows=rawpar[:,[0,1,6,7,10,11]].astype('<f8')
    assert np.all(rows[:,:4]>0) and np.all(np.diff(rows[:,0])>0) and np.all(np.diff(rows[:,3])>0)
    bins=len(rows);edges=np.r_[rows[0,0],np.sqrt(rows[:-1,0]*rows[1:,0]),rows[-1,0]]
    fractions=[];cosines=[];pdgs=[];cells=[];native_count=np.zeros(2*bins,dtype=np.int64);inputs=[]
    for root in [source,supplement]:
        if root is None:continue
        d=root/name;report=json.loads((d/'report.json').read_text());audit=report['audit'];raw=d/'conditional.phsp'
        assert sha(raw)==audit['raw_sha256'] and audit['material_identity']==identity
        r=np.fromfile(raw,DTYPE);c=r[r['kind']==0];child=r[(r['kind']==1)&(r['remaining']>0)]
        assert np.all(c['pdg']==11) and np.all((c['process']>=0)&(c['process']<2))
        assert np.all(c['energy']>=edges[0]) and np.all(c['energy']<=edges[-1])
        owner={(int(x['run']),int(x['event']),int(x['step'])):i for i,x in enumerate(c)}
        parent=np.array([owner[(int(x['run']),int(x['event']),int(x['step']))] for x in child],dtype=np.int64)
        groups=np.minimum(np.searchsorted(edges,c['energy'],side='right')-1,bins-1)*2+c['process']
        native_count+=np.bincount(groups,minlength=2*bins)
        f=np.r_[c['deposit']/c['energy'],c['remaining']/c['energy'],child['remaining']/child['energy']]
        cs=np.r_[np.zeros(len(c)),np.sum(c['pre']*c['post'],axis=1),np.sum(child['pre']*child['post'],axis=1)]
        pdg=np.r_[np.zeros(len(c),np.int32),np.full(len(c),11,np.int32),child['pdg']]
        cell=np.r_[groups,groups,groups[parent]];positive=f>0
        assert np.isfinite(f).all() and np.all(f>=0) and np.all(f<=1)
        assert np.isfinite(cs[positive]).all() and np.all(abs(cs[positive])<=1+1e-12)
        fractions.append(f[positive]);cosines.append(np.clip(cs[positive],-1,1));pdgs.append(pdg[positive]);cells.append(cell[positive])
        inputs.append(dict(report_sha256=sha(d/'report.json'),raw_sha256=sha(raw),path=str(raw)))
    cell=np.concatenate(cells);order=np.argsort(cell,kind='stable');cell=cell[order]
    fraction=np.concatenate(fractions)[order];cosine=np.concatenate(cosines)[order];pdg=np.concatenate(pdgs)[order]
    counts=np.bincount(cell,minlength=2*bins);offsets=np.r_[0,np.cumsum(counts)]
    weight=np.bincount(cell,weights=fraction,minlength=2*bins)
    assert np.all(abs(weight-native_count)<=1e-9*np.maximum(1,native_count))
    outcomes=np.zeros(len(cell),OUTCOME);outcomes['fraction']=fraction;outcomes['cosine']=cosine;outcomes['pdg']=pdg
    for i in range(2*bins):
        lo,hi=offsets[i:i+2]
        if hi>lo:
            cdf=np.cumsum(fraction[lo:hi]);cdf/=cdf[-1];cdf[-1]=1;outcomes['cdf'][lo:hi]=cdf
    cubicfile=par/'hard_rate_cubics.csv'
    cubic=np.loadtxt(cubicfile,delimiter=',',skiprows=1)
    assert np.isfinite(cubic).all()
    support=witnesses(cubic,edges)
    gaps=[dict(energy_MeV=float(e),bin=int(i),process=int(p)) for (i,p),e in support.items() if counts[2*i+p]==0]
    pieces=[cubic[cubic[:,0]==p,1:7].astype('<f8') for p in range(2)]
    for part in pieces:
        assert part[0,0]==edges[0] and part[-1,1]==edges[-1]
        assert np.all(part[:,1]>part[:,0]) and np.array_equal(part[1:,0],part[:-1,1])
    elast=elastic/name/'elastic_kernel.bin';urb=urban/name/'urban_material.bin'
    elastic_index=(elastic/'index.txt').read_text().splitlines()[node+1].split()
    assert elastic_index[0]==f'{name}/elastic_kernel.bin' and sha(elast)==elastic_index[1]
    urbreport=json.loads((urban/name/'report.json').read_text());assert sha(urb)==urbreport['table_sha256']
    assert elast.stat().st_size==9464 and urb.stat().st_size==56
    opts=meta['options'];d=out/name;d.mkdir();path=d/'operator.bin'
    # Native rules are pinned by source+unrounded exported values: PIXE=0,
    # lowest electron about100eV, linLossLimit=.01. Fluorescence in discrete collisions remains retained.
    log=(source/name/'topas.log').read_text()
    assert 'PIXE atomic de-excitation enabled                  0' in log
    with path.open('xb') as f:
        f.write(struct.pack('<8sIidIIIIQdddd',b'MAIECH02',2,identity['section_id'],identity['runtime_density_g_cm3'],len(rows),bins,len(pieces[0]),len(pieces[1]),len(outcomes),float(opts['electron_cut_MeV']),float(opts['photon_cut_MeV']),float(opts['lowest_electron_energy_MeV']),float(opts['linear_loss_limit'])))
        f.write(elast.read_bytes());f.write(urb.read_bytes())
        for a in [rows,*pieces,edges.astype('<f8'),np.column_stack([offsets[:-1],counts]).astype('<u8'),outcomes]:f.write(a.tobytes())
    assert path.stat().st_size==9600+48*len(rows)+48*len(cubic)+8*(bins+1)+32*bins+32*len(outcomes)
    result=dict(node=node,material_identity=identity,bytes=path.stat().st_size,sha256=sha(path),collisions=int(native_count.sum()),outcomes=len(outcomes),empty_channels=int(np.count_nonzero(counts==0)),coverage_complete=not gaps,positive_support_cells=len(support),native_cubic_sha256=sha(cubicfile),native_cubic_max_absolute_error=float(cubic[:,-1].max()),unsupported_rate_probes=gaps,inputs=inputs,parameter_sha256=sha(paramfile),elastic_sha256=sha(elast),urban_sha256=sha(urb),generator_sha256=sha(Path(__file__)),scope='All positive parent/local/electron/photon branches, no thinning. Native piecewise cubic rates with max(raw rate,0); piecewise-linear stopping/range and energy-bin conditional fractions with uniform azimuth are explicit new approximations. Fine support is not physical interpolation or dose validation.')
    save(d/'manifest.json',result);print(node,len(outcomes),len(gaps),flush=True);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name,default in [('source','electron_collision_grid82'),('supplement','electron_collision_threshold82_v3'),('parameters','electron_boundary_parameters82_v5'),('elastic','elastic82'),('urban','fluctuation82')]:p.add_argument('--'+name,type=Path,default=BASE/default)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--nodes',type=int,nargs='+',default=list(range(82)));p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    assert len(set(a.nodes))==len(a.nodes) and all(0<=n<82 for n in a.nodes)
    out=a.out.resolve();out.mkdir(exist_ok=False)
    tasks=[(n,a.source.resolve(),a.supplement.resolve(),a.parameters.resolve(),a.elastic.resolve(),a.urban.resolve(),out) for n in a.nodes]
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:results=list(pool.map(one,tasks))
    complete=all(x['coverage_complete'] for x in results);index=None
    if complete:
        index=out/'index.txt';index.write_text(f'MAIECB01 {len(results)}\n'+''.join(f'{x["material_identity"]["section_id"]} {x["material_identity"]["runtime_density_g_cm3"]:.17g} node{x["node"]:02d}/operator.bin {x["sha256"]}\n' for x in results))
    save(out/'manifest.json',dict(status='RESEARCH_CHARGED_BANK_NOT_TRANSPORT_ACCEPTED',coverage_complete=complete,results=results,total_bytes=sum(x['bytes'] for x in results),index_sha256=sha(index) if index else None))
    if not complete:raise SystemExit('Missing positive-rate channels; no runtime index emitted')
if __name__=='__main__':main()
