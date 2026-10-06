#!/usr/bin/env python3
"""Independent fixed-event native births for26 compositions and cut endpoints.

Native calls do not slow an ion through a5MeV bin. Each accepted collision
has the specified parent energy, native production cut and recorded children.
"""
import argparse,concurrent.futures,json,os,re,subprocess,time
from pathlib import Path
import numpy as np
from build_electron_response26 import collision_nodes
from probe_electron_birth_analytic26 import ME
from extract_photon_response import DTYPE
ROOT=Path(__file__).resolve().parents[1]
BAD={0,1,15,16,17,18,19,23}
def maximum(e,a):
    mass=938.272013 if a==1 else 11174.863387984389/12
    tau=e/mass;r=ME/(a*mass)
    return 2*ME*tau*(tau+2)/(1+2*(tau+1)*r+r*r)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--samples',type=int,default=100000);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--sections',type=int,nargs='+',default=list(range(26)))
    p.add_argument('--offgrid-boundary-dir',type=Path,
                   help='Use the middle independent boundary-probe density for each requested section')
    a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False);a.binary=a.binary.resolve()
    if a.samples<2 or a.workers<1:raise ValueError('samples/workers')
    nodes=collision_nodes(ROOT/'data/archive/electron_density_nodes_20261006/shared/charged','charged')
    cards={}
    for card in (ROOT/'out/electron_standalone_20261003/photon_conditional_bank512').glob('node*/topas.txt'):
        text=card.read_text();m=re.search(r'd:Ma/[^/]+/Density = ([\d.eE+-]+) g/cm3',text)
        if m:cards[float(m[1])]=text
    tasks=[]
    for s in a.sections:
        group=sorted((n for n in nodes if n['section']==s),key=lambda n:n['density'])
        selected=[group[0],group[-1]] if s in BAD else [group[len(group)//2]]
        offgrid=None
        if a.offgrid_boundary_dir:
            reports=[json.loads(f.read_text()) for f in a.offgrid_boundary_dir.glob('c*_off*/report.json')]
            reports=sorted((r for r in reports if r['section']==s),key=lambda r:r['density'])
            if not reports:raise ValueError(f'No independent cut probe for section{s}')
            offgrid=reports[len(reports)//2]
            selected=[dict(section=s,density=offgrid['density'])]
        for j,n in enumerate(selected):
            if s==25:
                from extract_water_electron_primitives import CARD
                text=CARD.format(quantity='IonBirthConditionalDump')
            else:
                base=min(group,key=lambda row:abs(row['density']-n['density']))
                text=cards[base['density']].replace('PhotonConditionalDump','IonBirthConditionalDump')
                if offgrid:
                    text=re.sub(r'(d:Ma/[^/]+/Density = )([\d.eE+-]+)( g/cm3)',
                                lambda m:m[1]+f'{n["density"]:.17g}'+m[3],text)
            # The source file is explicitly a native probe input, not a runtime
            # density registry. Actual cut is independently written by Geant4.
            path=ROOT/'data/archive/electron_density_nodes_20261006/shared/charged'
            import struct
            lines=(path/'index.txt').read_text().splitlines()[1:]
            if offgrid:cut=offgrid['cut'][0]
            else:
                file=next(line.split()[2] for line in lines if (int(line.split()[0]),float(line.split()[1]))==(s,n['density']))
                with (path/file).open('rb') as f:cut=struct.unpack('<8sIid4IQ4d',f.read(80))[9]
            for species in [1,12]:
                effective=max(cut,.00025) if species==1 else cut
                left,right=0.,450.
                for _ in range(64):
                    middle=(left+right)/2
                    if maximum(middle,species)>effective:right=middle
                    else:left=middle
                energies=sorted(set([right*1.05]+[e for e in [2.,5.,20.,50.,100.,200.,250.] if maximum(e,species)>effective]))
                d=a.out/f'c{s:02d}_cut{j}_a{species}';d.mkdir()
                (d/'topas.txt').write_text(text+f'\ni:Sc/Steps/ProjectileA = {species}\n')
                (d/'requests.txt').write_text(''.join(f'{i} 0 {e*species:.17g} {a.samples}\n' for i,e in enumerate(energies)))
                tasks.append(dict(directory=str(d),section=s,density=n['density'],projectile_a=species,expected_cut=cut,energies=energies))
    (a.out/'inputs.json').write_text(json.dumps(tasks,indent=2)+'\n')
    env=dict(os.environ,TOPAS_G4_DATA_DIR='/software/geant4-11.3.2/share/Geant4/data',LD_LIBRARY_PATH='/software/topas/lib:/software/geant4-11.3.2/lib')
    def run(spec):
        d=Path(spec['directory']);start=time.monotonic()
        with (d/'topas.log').open('x') as f:r=subprocess.run([str(a.binary),'topas.txt'],cwd=d,env=env,stdout=f,stderr=subprocess.STDOUT)
        if r.returncode or 'ION_BIRTH_CONDITIONAL_DUMP=PASS' not in (d/'topas.log').read_text():raise RuntimeError(str(d))
        native=np.fromfile(d/'conditional.phsp',DTYPE);events=native[native['kind']==0];child=native[native['kind']==1]
        if len(events)!=a.samples*len(spec['energies']) or not np.isfinite(child['remaining']).all():raise ValueError('count/energy')
        if np.max(abs(events['energy']-events['remaining']-events['deposit']-events['children']))>1e-8:raise ValueError('energy closure')
        if np.any(child['pdg']!=11) or np.any(child['remaining']<=0):raise ValueError('non-electron birth')
        meta=(d/'ion_models.txt').read_text();actual=float(re.search(r'electron_cut_MeV=([\d.eE+-]+)',meta)[1])
        if abs(actual/spec['expected_cut']-1)>1e-12:raise ValueError('native cut differs')
        summary=dict(spec,samples_per_energy=a.samples,seconds=time.monotonic()-start,events=len(events),children=len(child),actual_cut=actual,status='NATIVE_FIXED_EVENT_ENERGY_AUDIT_PASS')
        (d/'report.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True);return summary
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:results=list(pool.map(run,tasks))
    (a.out/'report.json').write_text(json.dumps(dict(status='NATIVE_COMPONENT_DATA_NOT_GPU_DOSE_ACCEPTANCE',results=results),indent=2)+'\n')
if __name__=='__main__':main()
