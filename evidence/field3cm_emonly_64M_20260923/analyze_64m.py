from pathlib import Path
import csv,hashlib,json,os,subprocess,sys,tarfile,datetime
import numpy as np
O=Path('/mnt/sdb/wuwei/MAIGO_review_fix_20260923/evidence/field3cm_emonly_64M_20260923')
OLD=O.parent/'field3cm_emonly_20260923'
validation='--validate-baseline' in sys.argv
env=dict(os.environ)
if validation:
    env.update(MAIGO_ANALYSIS_INPUT=str(OLD),MAIGO_ANALYSIS_OUTPUT=str(O/'results/baseline_validation'))
for script in ['analyze_field_64m.py','analyze_uncertainty_64m.py','gamma_noise_64m.py']:
    subprocess.run([sys.executable,str(O/script)],env=env,check=True)
if validation:
    D=O/'results/baseline_validation'
    old=json.loads((OLD/'results/comparison.json').read_text());new=json.loads((D/'comparison.json').read_text())
    assert old['histories_per_engine']==new['histories_per_engine']==12800000
    assert old['rows']==new['rows']
    a=list(csv.DictReader((OLD/'results/depth_curves.csv').open()));b=list(csv.DictReader((D/'depth_curves.csv').open()))
    assert len(a)==len(b)
    for aa,bb in zip(a,b):
        assert aa['ROI']==bb['ROI']
        for k in aa:
            if k!='ROI':np.testing.assert_allclose(float(aa[k]),float(bb[k]),rtol=1e-12,atol=1e-12,equal_nan=True)
    g=json.loads((D/'gamma_noise_summary.json').read_text())
    for k,v in g['results'].items():assert v['pass_rate_percent']==g['baseline_12p8M_gamma_pass_rates'][k]
    result=dict(status='PASSED',checked_at=datetime.datetime.now().astimezone().isoformat(),
        exact_baseline_ROI_results=True,all_depth_curves_match=True,all_four_gamma_pass_rates_match=True)
    (O/'analysis_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
else:
    assert json.loads((O/'results/comparison.json').read_text())['histories_per_engine']==64000000
    def sha(p):
        h=hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
        return h.hexdigest()
    files=[p for p in (O/'results').iterdir() if p.is_file()]+[p for p in O.iterdir() if p.is_file() and p.suffix in ['.py','.sh','.json','.tsv'] and p.name not in ['final_file_checksums.json','serial_status.json']]
    files+=sorted(p for p in (O/'cases').rglob('*') if p.is_file() and p.name not in ['dose.bin','dose.binheader'])
    checks=[dict(path=str(p.relative_to(O)),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
    (O/'final_file_checksums.json').write_text(json.dumps(checks,indent=2)+'\n')
    files.append(O/'final_file_checksums.json')
    b=O/'final_64m_results_bundle.tar.gz'
    with tarfile.open(b,'w:gz') as tar:
        for p in files:tar.add(p,arcname=str(p.relative_to(O)))
    b.with_suffix(b.suffix+'.sha256').write_text(sha(b)+'  '+b.name+'\n')
    print('FINAL_BUNDLE',str(b),flush=True)
