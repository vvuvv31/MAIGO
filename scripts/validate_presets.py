#!/usr/bin/env python3
"""Matched two-batch regression of the eight runtime presets; no throughput claim."""
import argparse,json,os,re,subprocess
from pathlib import Path
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[1]
CASES=[f'{s}_{m}_{b}' for s in ['proton','carbon'] for m in ['water','ct'] for b in ['broadbeam','minibeam']]
AUDITS=['unified-em-audit','all-ion-elastic','elastic-failure-sites','urban-v2-steps','minibeam-copper-cascade']

def run(case,arm,binary,input_root,out,histories):
    directory=out/case/arm;directory.mkdir(parents=True)
    cfg=yaml.safe_load((input_root/'config'/f'{case}.yaml').read_text())
    for key,value in cfg.items():
        if isinstance(value,str) and value.startswith(('data/','inputs/')):cfg[key]=str(input_root/value)
    cfg['random_seed']=1001904402
    file=directory/'case.yaml';file.write_text(yaml.safe_dump(cfg,sort_keys=False))
    env={k:v for k,v in os.environ.items() if not k.startswith('CARBON_')}
    env['LD_LIBRARY_PATH']='/home/wuwei/sycl_workspace/llvm/build/install/lib'
    command=[str(binary),'--config',str(file),'--batch-count','2','--histories-per-batch',str(histories),'--batch-output',str(directory/'batches')]
    (directory/'command.json').write_text(json.dumps(command,indent=2)+'\n')
    with (directory/'gpu.log').open('w') as log:r=subprocess.run(command,cwd=input_root,env=env,stdout=log,stderr=subprocess.STDOUT)
    text=(directory/'gpu.log').read_text()
    assert r.returncode==0,(case,arm,text[-4000:])
    for i in range(2):
        q=json.loads((directory/'batches'/f'batch{i:06d}'/'quality_report.json').read_text())
        assert q['accepted'] and not q['failures'] and q['queue_overflow_count']==0 and q['physical_relative_energy_residual']<1e-4,q
        dose=np.fromfile(directory/'batches'/f'batch{i:06d}'/'dose.raw','<f4')
        assert np.isfinite(dose).all() and (dose>=0).all() and dose.max()>0
    for line in text.splitlines():
        if line.startswith(tuple('['+key+']' for key in AUDITS[1:])):
            assert all(int(n)==0 for n in re.findall(r'(?:coverage_missing|invalid)=(\d+)',line)),line
    if arm=='candidate':
        selected=re.findall(r'^\[transport-pipeline\] name=(\S+)',text,re.M)
        assert selected==[cfg.get('transport_pipeline','generic')]*2,(case,selected)
    rows=re.findall(r'^\[unified-em-audit\] (.*)$',text,re.M)
    assert len(rows)==2 and all(int(row.split()[0])==0 for row in rows)
    return directory,text

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control',type=Path,required=True);p.add_argument('--control-root',type=Path,required=True)
    p.add_argument('--candidate',type=Path,default=ROOT/'build/carbon_mc');p.add_argument('--candidate-root',type=Path,default=ROOT);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--histories',type=int,default=10000)
    p.add_argument('--cases',nargs='+',choices=CASES,default=CASES)
    a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False)
    reports=[]
    for case in a.cases:
        c,ct=run(case,'control',a.control.resolve(),a.control_root.resolve(),a.out,a.histories)
        n,nt=run(case,'candidate',a.candidate.resolve(),a.candidate_root.resolve(),a.out,a.histories)
        if case.endswith('_minibeam'):
            assert len(re.findall(r'^\[primary-compact\] enabled=1 ',nt,re.M))==2,(case,'compact pipeline not used')
            assert len(re.findall(r'^\[primary-split-time\] ',nt,re.M))==2,(case,'split kernels not launched')
        counts={}
        for key in AUDITS:
            rows=[re.findall(r'^\['+key+r'\] (.*)$',t,re.M) for t in [ct,nt]]
            assert rows[0]==rows[1],(case,key,rows)
            counts[key]=len(rows[0])
        l1=[]
        for i in range(2):
            name=Path('batches')/f'batch{i:06d}'
            x=np.fromfile(c/name/'dose.raw','<f4').astype(float);y=np.fromfile(n/name/'dose.raw','<f4').astype(float)
            assert x.shape==y.shape
            difference=float(np.abs(x-y).sum()/max(np.abs(x).sum(),1e-30));assert difference<1e-5,(case,difference);l1.append(difference)
            # Largest-remainder spot allocation and every source field/seed stay identical.
            assert (c/name/'primary_batches.csv').read_bytes()==(n/name/'primary_batches.csv').read_bytes()
            xl=json.loads((c/name/'energy_ledger.json').read_text());yl=json.loads((n/name/'energy_ledger.json').read_text())
            for key in ['histories','nuclear_interactions','all_ion_primary_elastic_interactions','all_ion_secondary_elastic_interactions','elastic_post_em_null_collisions']:
                assert xl[key]==yl[key],(case,key,xl[key],yl[key])
            # The release layout keeps live ion audits and removes retired zero ledgers.
            if 'ion_species_transport_ledger_MeV' in yl:
                assert not any(key.startswith('cinel02_') for key in yl),(case,'retired ledger emitted')
                assert np.allclose(xl['ion_species_transport_ledger_MeV'],yl['ion_species_transport_ledger_MeV'],rtol=1e-5,atol=1e-4),(case,'ion energy ledger')
                assert xl['ion_species_terminal_reason_counts']==yl['ion_species_terminal_reason_counts'],(case,'ion terminal counts')
            assert 'energy_balance_error' not in yl,(case,'duplicate energy output')
            retired_neutral_outputs = {'E_neutron_kinetic_gain_MeV', 'E_neutron_deposited_water_MeV',
                'E_neutral_overflow_MeV', 'E_neutral_unsupported_product_MeV',
                'E_neutral_package_closure_residual_MeV', 'elastic_queued_neutral_MeV', 'neutral_queue_overflow'}
            assert not retired_neutral_outputs.intersection(yl),(case,'retired neutral output')
            assert not list((n/name).glob('neutron*')),(case,'neutron transport output emitted')
            xq=json.loads((c/name/'quality_report.json').read_text());yq=json.loads((n/name/'quality_report.json').read_text())
            assert 'absolute_accounting_energy_residual_MeV' not in yq and 'accounting_relative_energy_residual' not in yq,(case,'duplicate accounting output')
            for key in ['voxel_scored_energy_MeV','voxel_grid_closure_residual_MeV']:
                if key in xq:assert np.isclose(xq[key],yq[key],rtol=1e-5,atol=1e-4)
        report=dict(case=case,histories_per_arm=2*a.histories,relative_L1_by_batch=l1,discrete_audits_match=True,source_batches_match=True,counters=counts)
        reports.append(report);print(json.dumps(report),flush=True)
    (a.out/'summary.json').write_text(json.dumps(dict(passed=True,reports=reports,throughput_claim=False,new_TOPAS_gamma=False),indent=2)+'\n')
if __name__=='__main__':main()
