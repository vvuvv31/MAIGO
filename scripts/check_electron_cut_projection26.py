#!/usr/bin/env python3
"""Test microscopic cut projection before constructing a26 runtime package.

Preserve prompt photon-interaction electrons even below production cuts;
apply cuts only to relaxation children. Charged hard laws are conditioned on
the actual energy transfer. Null fractions are reported separately, never
silently removed from an eventual transport law.
"""
import argparse,json
from pathlib import Path
import numpy as np
from extract_photon_response import DTYPE

def branches(raw,energy,kind,process,ecut,pcut,condition=False):
    events=raw[raw['kind']==0];children=raw[raw['kind']==1]
    null=(events['remaining']==energy)&(events['deposit']==0)&(events['children']==0)&np.all(events['post']==[0,0,1],axis=1)
    selected=~null
    if condition and kind==0:selected &= energy-events['remaining']>= (ecut if process==0 else pcut)
    count=int(selected.sum())
    if not count:return None
    ci=children['event']-events[0]['event'];child_keep=selected[ci]
    prompt=np.zeros(len(children),bool)
    if kind==1 and process<2:
        first=np.r_[True,np.diff(children['event'])!=0] if len(children) else prompt
        prompt=first & (True if process==0 else events['remaining'][ci]>0)
    cut=np.where(children['pdg']==11,ecut,pcut)
    keep=child_keep & (prompt | (children['remaining']>=cut))
    deposit=events['deposit'].copy()
    np.add.at(deposit,ci[child_keep&~keep],children['remaining'][child_keep&~keep])
    ne=np.r_[events['remaining'][selected]/energy,children['remaining'][keep]/energy,np.ones(count)]
    mu=np.r_[events['post'][selected,2],children['post'][keep,2],np.zeros(count)]
    pdg=np.r_[events['pdg'][selected],children['pdg'][keep],np.zeros(count,dtype=int)]
    w=np.r_[events['remaining'][selected],children['remaining'][keep],deposit[selected]]/(energy*count)
    if abs(w.sum()-1)>1e-8:raise ValueError('Projection energy closure')
    active=w>0
    return dict(energy=ne[active],cosine=mu[active],weight=w[active],pdg=pdg[active],events=count,
                null_fraction=float(null.mean()),selected_fraction=count/len(events))

def cdf(b):
    energy=np.unique(np.r_[np.geomspace(1e-5,.1,33),np.linspace(.1,1,33)])
    angle=np.linspace(-1,1,33)
    out=np.zeros((3,len(energy)+1,len(angle)+1))
    category=np.where(b['pdg']==0,0,np.where(b['pdg']==11,1,2))
    np.add.at(out,(category,np.searchsorted(energy,b['energy'],side='left'),np.searchsorted(angle,b['cosine'],side='left')),b['weight'])
    return out.cumsum(1).cumsum(2)[:,:-1,:-1]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--native',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    rows=[]
    for directory in sorted(a.native.glob('composition*')):
        section=int(directory.name[-2:]);audit=np.atleast_1d(np.genfromtxt(directory/'composition_audit.csv',delimiter=',',names=True,dtype=None,encoding='utf8'))
        raw=np.memmap(directory/'conditional.phsp',DTYPE,mode='r')
        _,indices=np.unique(raw['run'],return_index=True);ranges=dict(zip(raw['run'][indices],zip(indices,np.r_[indices[1:],len(raw)])))
        for r in audit:
            kind,proc,energy=int(r['kind']),int(r['process']),float(r['energy_MeV']);ecut,pcut=float(r['electron_cut_MeV']),float(r['photon_cut_MeV'])
            if ecut==.00099 and pcut==.00099:continue
            floor=audit[(audit['kind']==kind)&(audit['process']==proc)&(audit['energy_MeV']==energy)&(audit['electron_cut_MeV']==.00099)&(audit['photon_cut_MeV']==.00099)]
            if len(floor)!=1:raise ValueError('Missing unique floor primitive')
            lo,hi=ranges[int(floor[0]['cell'])];projected=branches(raw[lo:hi],energy,kind,proc,ecut,pcut,True)
            lo,hi=ranges[int(r['cell'])];native=branches(raw[lo:hi],energy,kind,proc,ecut,pcut)
            if not projected or not native:continue
            moments={}
            for field in ['energy','cosine']:
                av=[float(b['weight']@b[field]) for b in [projected,native]]
                # Conservative branch-weight variance diagnostic; not a full
                # clustered-event confidence certificate for promotion.
                variance=sum(float(np.sum(b['weight']**2*(b[field]-mean)**2)) for b,mean in zip([projected,native],av))
                moments[field]=dict(projected=av[0],native=av[1],difference=av[0]-av[1],z=(av[0]-av[1])/np.sqrt(variance) if variance else 0.)
            rows.append(dict(section=section,kind=kind,process=proc,energy=energy,ecut=ecut,pcut=pcut,
                             selected_events=projected['events'],native_events=native['events'],
                             floor_null_fraction=projected['null_fraction'],native_null_fraction=native['null_fraction'],
                             moments=moments,joint_cdf_difference=float(abs(cdf(projected)-cdf(native)).max())))
        print(json.dumps(dict(section=section,completed_conditions=len([r for r in rows if r['section']==section]))),flush=True)
    result=dict(status='MICROSCOPIC_CUT_PROJECTION_DIAGNOSTIC_NOT_PACKAGE_OR_NULL_LAW_ACCEPTANCE',cases=len(rows),
                maximum_moment_z=max(abs(r['moments'][f]['z']) for r in rows for f in ['energy','cosine']),
                maximum_joint_cdf=max(r['joint_cdf_difference'] for r in rows),rows=rows)
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
