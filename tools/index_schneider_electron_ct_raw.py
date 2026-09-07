"""Index completed CT response data, refusing partial jobs and identity drift."""
import argparse,json,re,subprocess
from pathlib import Path
from analyze_longitudinal_holdout import sha

def index(root,out):
    manifest=root/'manifest.json';m=json.loads(manifest.read_text())
    for path,pin in m['inputs'].items():
        if sha(Path(path))!=pin:raise ValueError('Changed input: '+path)
    sources=[]
    for section,c in sorted(m['cases'].items(),key=lambda kv:int(kv[0])):
        raw=Path(c['path']);logs=list(raw.parent.glob('job_*.log'))
        if len(logs)!=1:raise ValueError('Missing/ambiguous job log: '+section)
        jid=re.fullmatch(r'job_(\d+)\.log',logs[0].name)[1]
        rows=subprocess.check_output(['sacct','-j',jid,'--format=JobIDRaw,State,ExitCode','--noheader','--parsable2'],text=True).splitlines()
        if not any(row.split('|')[:3]==[jid,'COMPLETED','0:0'] for row in rows):raise ValueError('Job not successfully completed: '+jid)
        if not raw.is_file() or raw.stat().st_size==0 or not raw.with_suffix('.header').is_file():raise ValueError('Missing raw')
        sources.append(dict(path=str(raw),sha256=sha(raw),hu=c['hu'],section_id=c['section_id'],
            density_g_cm3=c['density_g_cm3'],parent_inset_mm=c['parent_inset_mm'],job_id=jid))
    if {s['section_id'] for s in sources}!=set(range(25)):raise ValueError('Incomplete material coverage')
    with out.open('x') as f:json.dump(dict(energy_max_MeVu=500,status='UNVALIDATED_CT_RAW',sources=sources,
        campaign_manifest_sha256=sha(manifest),indexer_sha256=sha(Path(__file__))),f,indent=2)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args();index(a.root.resolve(),a.output.resolve())
