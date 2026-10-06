#!/usr/bin/env python3
"""Direct native composition/cut probes; no runtime material-density nodes.

The rates phase measures explicit-cut microscopic model rates. The outcomes
phase only samples supported conditions, preserving every correlated branch.
These are research inputs, not a qualified production package.
"""
import argparse,concurrent.futures,json,os,re,struct,subprocess,time
from pathlib import Path
import numpy as np
from build_electron_response26 import collision_nodes
from extract_water_electron_primitives import CARD
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--phase',choices=['rates','outcomes'],default='rates')
    p.add_argument('--rates',type=Path);p.add_argument('--samples',type=int,default=20000)
    p.add_argument('--sections',type=int,nargs='+',default=list(range(26)))
    p.add_argument('--models',nargs='+',help='Restrict sampled native models; rates remain unfiltered')
    a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False);a.binary=a.binary.resolve()
    nodes=collision_nodes(ROOT/'data/archive/electron_density_nodes_20261006/shared/charged','charged');cards={}
    for card in (ROOT/'out/electron_standalone_20261003/photon_conditional_bank512').glob('node*/topas.txt'):
        text=card.read_text();m=re.search(r'd:Ma/[^/]+/Density = ([\d.eE+-]+) g/cm3',text)
        if m:cards[float(m[1])]=text
    for n in nodes:
        index=ROOT/'data/archive/electron_density_nodes_20261006/shared/charged'
        file=next(line.split()[2] for line in (index/'index.txt').read_text().splitlines()[1:]
                  if (int(line.split()[0]),float(line.split()[1]))==(n['section'],n['density']))
        with (index/file).open('rb') as f:n['cut']=struct.unpack('<8sIid4IQ4d',f.read(80))[9:11]
    tasks=[]
    for section in a.sections:
        group=sorted((n for n in nodes if n['section']==section),key=lambda n:n['density'])
        ref=group[len(group)//2];d=a.out/f'composition{section:02d}';d.mkdir()
        text=CARD.format(quantity='ElectronCompositionProbe') if section==25 else cards[ref['density']].replace('PhotonConditionalDump','ElectronCompositionProbe')
        (d/'topas.txt').write_text(text)
        if a.phase=='rates':
            cuts=sorted(set([tuple(group[i]['cut']) for i in [0,len(group)//2,len(group)-1]]+[(.00099,.00099)]))
            rows=[]
            for ecut,pcut in cuts:
                for kind,processes,energies in [(0,2,[.003,.01,.04,.08,.1,.100001,.2,.4,.6]),(1,3,[.003,.01,.08,.2,.4,.6])]:
                    for energy in energies:
                        for process in range(processes):rows.append([len(rows),kind,process,energy,ecut,pcut,0])
        else:
            if not a.rates:raise ValueError('--rates is required for outcomes')
            rates=np.atleast_1d(np.genfromtxt(a.rates/f'composition{section:02d}'/'composition_audit.csv',delimiter=',',names=True,dtype=None,encoding='utf8'))
            rows=[[int(r['cell']),int(r['kind']),int(r['process']),float(r['energy_MeV']),float(r['electron_cut_MeV']),float(r['photon_cut_MeV']),a.samples]
                  for r in rates if r['table_rate_per_mm']>0 and (not a.models or r['model'] in a.models)]
        (d/'requests.txt').write_text(''.join(' '.join(f'{v:.17g}' if isinstance(v,float) else str(v) for v in row)+'\n' for row in rows))
        tasks.append(dict(directory=str(d),section=section,density=ref['density'],requests=len(rows)))
    env=dict(os.environ,TOPAS_G4_DATA_DIR='/software/geant4-11.3.2/share/Geant4/data',LD_LIBRARY_PATH='/software/topas/lib:/software/geant4-11.3.2/lib')
    def run(task):
        d=Path(task['directory']);start=time.monotonic()
        with (d/'topas.log').open('x') as log:r=subprocess.run([str(a.binary),'topas.txt'],cwd=d,env=env,stdout=log,stderr=subprocess.STDOUT)
        if r.returncode or 'ELECTRON_COMPOSITION_PROBE=PASS' not in (d/'topas.log').read_text():raise RuntimeError(str(d))
        audit=np.atleast_1d(np.genfromtxt(d/'composition_audit.csv',delimiter=',',names=True,dtype=None,encoding='utf8'))
        if len(audit)!=task['requests'] or np.any(audit['requested']!=audit['accepted']) or np.max(audit['max_energy_residual_MeV'])>1e-9:raise ValueError('native audit')
        out=dict(task,seconds=time.monotonic()-start,status='NATIVE_MODEL_AUDIT_PASS_NOT_RUNTIME_PACKAGE')
        (d/'report.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True);return out
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:reports=list(pool.map(run,tasks))
    (a.out/'report.json').write_text(json.dumps(dict(phase=a.phase,status='COMPONENT_INPUTS_NOT_GPU_OR_DOSE_ACCEPTANCE',compositions=len(reports),reports=reports),indent=2)+'\n')
if __name__=='__main__':main()
