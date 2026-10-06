#!/usr/bin/env python3
"""Native rate/scalar probes at densities absent from the source bank."""
import argparse,concurrent.futures,json,os,re,struct,subprocess,time
from pathlib import Path
import numpy as np
from build_electron_response26 import collision_nodes,raw_rate
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False)
    data=np.load(a.model/'rates26.npz');nodes=collision_nodes(ROOT/'data/archive/electron_density_nodes_20261006/shared/charged','charged');cards={};tasks=[]
    for card in (ROOT/'out/electron_standalone_20261003/photon_conditional_bank512').glob('node*/topas.txt'):
        text=card.read_text();m=re.search(r'd:Ma/[^/]+/Density = ([\d.eE+-]+) g/cm3',text)
        if m:cards[float(m[1])]=text
    for section in [0,1,15,16,17,18,19,23]:
        group=sorted((n for n in nodes if n['section']==section),key=lambda n:n['density'])
        for j,(left,right) in enumerate(zip(group,group[1:])):
            rho=float(np.float32(np.sqrt(left['density']*right['density'])))
            text=cards[left['density']].replace('PhotonConditionalDump','ElectronBoundaryProbe')
            text=re.sub(r'(d:Ma/[^/]+/Density = )([\d.eE+-]+)( g/cm3)',lambda m:m[1]+f'{rho:.17g}'+m[3],text)
            d=a.out/f'c{section:02d}_off{j:02d}';d.mkdir();(d/'topas.txt').write_text(text);tasks.append((d,section,rho))
    env=dict(os.environ,TOPAS_G4_DATA_DIR='/software/geant4-11.3.2/share/Geant4/data',LD_LIBRARY_PATH='/software/topas/lib:/software/geant4-11.3.2/lib')
    binary=ROOT/'out/electron_standalone_20261003/electron_boundary_native_v4/topas'
    def run(task):
        d,section,rho=task;start=time.monotonic()
        with (d/'topas.log').open('x') as f:p=subprocess.run([str(binary),'topas.txt'],cwd=d,env=env,stdout=f,stderr=subprocess.STDOUT)
        if p.returncode or 'ELECTRON_BOUNDARY_PROBE=PASS' not in (d/'topas.log').read_text():raise RuntimeError(str(d))
        options=dict(line.split('=',1) for line in (d/'boundary_options.txt').read_text().splitlines())
        actual=float(options['density_g_cm3'])
        if np.float32(actual)!=np.float32(rho) or abs(actual/rho-1)>1e-12:raise ValueError('density identity')
        r=dict(directory=str(d),section=section,density=rho,cut=[float(options['electron_cut_MeV']),float(options['photon_cut_MeV'])],seconds=time.monotonic()-start)
        (d/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True);return r
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:reports=list(pool.map(run,tasks))
    groups=[];references=[];offset=0
    with (a.out/'fixture.bin').open('xb') as f:
        f.write(struct.pack('<I',len(reports)*2))
        for report in reports:
            native=np.loadtxt(Path(report['directory'])/'hard_rate_cubics.csv',delimiter=',',skiprows=1)
            for process in [0,1]:
                prefix=f'c{report["section"]:02d}_p{process}_';edges=data[prefix+'edges'];bounds=data[prefix+'cut_piece_bounds'];coeff=data[prefix+'coeff'];slope=data[prefix+'density_slope'];low,high=data[prefix+'density_domain'];majorant=data[prefix+'coefficient_majorant'][0]
                rows=native[native[:,0]==process,1:7]
                energy=np.geomspace(1e-5,.65,1024).astype('<f4');energy[0]=edges[0];energy[-1]=edges[-1]
                query=np.column_stack([np.full(len(energy),report['density']),np.full(len(energy),report['cut'][process]),energy]).astype('<f4')
                f.write(struct.pack('<4I3f',len(edges)-1,len(bounds)-1,coeff.shape[1],len(query),low,high,majorant))
                for x in [edges,bounds,coeff,slope,query]:f.write(x.astype('<f4').tobytes())
                expected=np.maximum(0.,raw_rate(rows,energy.astype(float)))
                groups.append(dict(**report,process=process,offset=offset,count=len(energy),energy=energy));references.extend(expected);offset+=len(energy)
    binary=a.out/'check';subprocess.run(['g++','-O3','-std=c++20','-I'+str(ROOT/'include'),str(ROOT/'scripts/check_electron_cut_rates26.cpp'),'-o',str(binary)],check=True)
    result=subprocess.run([str(binary),str(a.out/'fixture.bin'),str(a.out/'actual.bin')],capture_output=True,text=True,check=True);(a.out/'runtime.json').write_text(result.stdout)
    actual=np.fromfile(a.out/'actual.bin','<f4').astype(float);reference=np.array(references);rows=[]
    for g in groups:
        sl=slice(g['offset'],g['offset']+g['count']);expected=reference[sl];error=abs(actual[sl]-expected);peak=max(expected.max(),1e-30);active=expected>peak*1e-4
        i=int(np.argmax(error/np.maximum(expected,peak*1e-4)))
        row=dict(g,peak_normalized_error=float(error.max()/peak),relative_active=float(np.max(error[active]/expected[active])) if active.any() else 0.,hotspot_energy=float(g['energy'][i]),hotspot_native=float(expected[i]),hotspot_candidate=float(actual[sl][i]));row.pop('energy');rows.append(row);print(json.dumps(row),flush=True)
    report=dict(status='INDEPENDENT_OFFGRID_NATIVE_RATE_COMPARISON_NOT_GPU_DOSE',runtime=json.loads(result.stdout),maximum_peak_normalized_error=max(r['peak_normalized_error'] for r in rows),maximum_active_relative=max(r['relative_active'] for r in rows),component_pass=all(r['peak_normalized_error']<.0025 and r['relative_active']<.01 for r in rows),rows=rows)
    (a.out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
    if not report['component_pass']:raise SystemExit(2)
if __name__=='__main__':main()
