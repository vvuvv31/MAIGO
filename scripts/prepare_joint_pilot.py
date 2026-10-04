#!/usr/bin/env python3
"""Package the selected lung joint operator for the opt-in CT integration."""
import argparse,json,struct,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmark/electron_response_model'))
from joint_family import load_model
from macro_model import DTYPE
from prototype import sha,save

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ['model','full-model','operator','out']:p.add_argument('--'+arg,type=Path,required=True)
    a=p.parse_args();m,report=load_model(a.model);full,fr=load_model(a.full_model)
    assert report.get('secondary_continuation_resampling') and not report.get('resample_photon_electrons',False)
    assert 'reduction' not in report and 'cutoff_endpoint' in report
    assert report['source']['sha256']==fr['source']['sha256']
    operator=json.loads((a.operator/'manifest.json').read_text())
    assert operator['model_report_sha256']==sha(a.model/'report.json')
    assert operator['payload_sha256']==sha(a.operator/'operator.bin') and operator['birth_from_continuation']
    raw_path=Path(report['source']['path']);assert sha(raw_path)==report['source']['sha256']
    raw=np.memmap(raw_path,DTYPE,mode='r');native=np.flatnonzero(raw['parent']!=0);g=full['graph']
    assert len(native)==len(g['pdg'])
    photon=np.zeros(len(native),bool);active=np.flatnonzero(g['pdg']==22)
    while len(active):
        photon[active]=True
        following=np.unique(np.r_[g['next'][active],g['child'][active].ravel()]);following=following[following>=0]
        active=following[~photon[following]]
    g['next'][(~photon)&g['inside']&(g['pdg']==11)&(g['fraction'][:,1]>0)]=-2
    reached=np.zeros(len(native),bool)
    active=np.unique(np.r_[full['continuation']['child'].ravel(),full['entry']['child'].ravel()]);active=active[active>=0]
    while len(active):
        reached[active]=True
        following=np.unique(np.r_[g['next'][active],g['child'][active].ravel()]);following=following[following>=0]
        active=following[~reached[following]]
    ids=np.flatnonzero(reached);remap=np.full(len(native),-1,np.int32);remap[ids]=np.arange(len(ids))
    for key,array in g.items():
        candidate=array[ids].copy()
        if key in ['next','child']:candidate=np.where(candidate>=0,remap[np.maximum(candidate,0)],candidate)
        assert np.array_equal(candidate,m['graph'][key]),key
    pre=raw['direction'][native[ids]].copy()
    assert np.array_equal(raw['post_direction'][native[ids]],m['graph']['postdir'])
    assert np.max(abs(np.linalg.norm(pre,axis=1)-1))<1e-7
    root=Path(__file__).resolve().parents[1]
    index=root/'data/ct_electron/proton/index.json';ci=root/'data/ct_electron/carbon/index.json'
    def identity(path):
        return next(t['material_identity'] for t in json.loads(path.read_text())['tables'] if t['path'].split('/')[0]==report['source']['table'])
    pi,carbon=identity(index),identity(ci)
    assert pi['section_id']==carbon['section_id']==1
    assert pi['runtime_density_g_cm3']==carbon['runtime_density_g_cm3']==report['source']['density_g_cm3']
    for key in ['mean_excitation_energy_eV','element_weights']:assert pi['composition'][key]==carbon['composition'][key]
    a.out.mkdir(parents=True,exist_ok=False);path=a.out/'joint_pilot.bin'
    with path.open('xb') as f:
        f.write(struct.pack('<8sIdd',b'MAIJNT01',1,report['source']['density_g_cm3'],.05))
        f.write(sha(index).encode());f.write(sha(a.model/'report.json').encode())
        with (a.operator/'operator.bin').open('rb') as src:
            while chunk:=src.read(1<<20):f.write(chunk)
        f.write(pre.astype('<f8').tobytes())
    save(a.out/'manifest.json',dict(status='EXPERIMENTAL_CT_HYBRID_NOT_ACCEPTED',file_sha256=sha(path),bytes=path.stat().st_size,
        model_report_sha256=sha(a.model/'report.json'),operator_manifest_sha256=sha(a.operator/'manifest.json'),
        index_sha256=sha(index),material_identity=pi,graph_rows=len(ids),
        scope='Selected unthinned joint table, exact photons and photon descendants; legacy continuation outside certified homogeneous support',
        script_sha256=sha(Path(__file__))))
    print(json.dumps(dict(file=str(path),sha256=sha(path),bytes=path.stat().st_size)),flush=True)

if __name__=='__main__':main()
