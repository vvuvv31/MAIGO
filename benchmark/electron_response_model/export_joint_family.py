#!/usr/bin/env python3
"""Export the isolated double-arithmetic operator; its GPU scorer is FP32."""
import argparse
from pathlib import Path
import struct
import numpy as np
from joint_family import load_model, attach_entry
from prototype import save, sha

STEP=np.dtype([('cdf','<f8'),('delta','<f8',(3,)),('post','<f8',(3,)),('source','<f8',(3,)),
               ('energy','<f8'),('length','<f8'),('fraction','<f8',(4,)),('child','<i4',(2,))])
GRAPH=np.dtype([('delta','<f8',(3,)),('length','<f8'),('fraction','<f8',(4,)),('postdir','<f8',(3,)),
                ('next','<i4'),('child','<i4',(2,)),('pdg','<i4'),('inside','<i4'),('pad','<i4')])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--entry-from',type=Path)
    p.add_argument('--incident',type=float,help='Research cross-energy test: draw the birth from the frozen continuation table')
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert STEP.itemsize==136 and GRAPH.itemsize==112
    models,report=load_model(a.model);incident=report['continuation_audit']['actual_energy_MeV']
    cross=None
    if a.entry_from:incident,cross=attach_entry(models,report,a.entry_from)
    assert not(a.entry_from and a.incident is not None)
    if a.incident is not None:
        incident=float(np.float32(a.incident))
        assert models['continuation']['edges'][0]<=incident<=models['continuation']['edges'][-1]
    a.out.mkdir(parents=True,exist_ok=False)
    exported=[]
    for name in ['continuation','entry']:
        m=models[name];r=np.zeros(len(m['cdf']),STEP)
        assert m['fraction'].shape[1]==4 and m['child'].shape[1]==2
        for key in ['cdf','delta','length','fraction','child']:r[key]=m[key]
        r['post']=m['postdir'];r['source']=m['source_direction'];r['energy']=m['source_energy']
        exported.append(r)
    if a.incident is not None:exported[1]=np.empty(0,STEP)
    g=models['graph'];gr=np.zeros(len(g['pdg']),GRAPH)
    for key in g:
        if key!='energy':gr[key]=g[key]
    m=models['continuation'];bins=len(m['nearest'])
    assert max(len(r) for r in exported+[gr])<2**31
    path=a.out/'operator.bin'
    with path.open('xb') as f:
        f.write(struct.pack('<8sdIIII',b'JTFAM002',incident,len(exported[0]),len(exported[1]),len(gr),bins))
        for array in [m['edges'].astype('<f8'),m['offsets'].astype('<u4'),m['nearest'].astype('<u4'),*exported,gr]:
            f.write(array.tobytes())
    save(a.out/'manifest.json',dict(status='RESEARCH_OPERATOR_NOT_CT_ACCEPTED',model_report_sha256=sha(a.model/'report.json'),
         independent_entry=cross,payload_sha256=sha(path),payload_bytes=path.stat().st_size,
         birth_from_continuation=a.incident is not None,
         incident_energy_MeV=incident,continuation=len(exported[0]),entry=len(exported[1]),graph=len(gr),bins=bins,
         scoring='FP32 required',script_sha256=sha(Path(__file__))))


if __name__=='__main__':main()
