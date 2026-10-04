#!/usr/bin/env python3
"""Run an isolated carbon CT reference and keep strict completed-history checks."""
from pathlib import Path
import argparse,hashlib,json,os,re,subprocess,time

ROOT=Path(__file__).resolve().parents[3]
BINARY=Path('/mnt/sda/wuwei/real_ct_electron_diagnostic_20260928/build/topas')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('case');a=p.parse_args()
    d=ROOT/'out/carbon_ct_minibeam_20261002'/('topas_'+a.case)
    config=(d/'topas.txt').read_text()
    n=int(re.search(r'i:So/Beam/NumberOfHistoriesInRun = (\d+)',config)[1])
    assert '"PrimaryIonizationResponsePhysics"' not in config
    assert 'd:Ph/Default/EMRangeMax = 10 GeV' in config
    env=os.environ.copy();env.update(TOPAS_G4_DATA_DIR='/software/geant4-11.3.2/share/Geant4/data',
        LD_LIBRARY_PATH='/software/topas/lib:/software/geant4-11.3.2/lib:'+env.get('LD_LIBRARY_PATH',''))
    start=time.monotonic()
    with (d/'topas.log').open('x') as log:
        code=subprocess.run([str(BINARY),'topas.txt'],cwd=d,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    log=(d/'topas.log').read_text();counts=re.findall(r'Particle source Beam: Total number of histories: (\d+)',log)
    unscored=sum(float(v) for v in re.findall(r'Total amount of energy not scored for this reason: ([\d.eE+-]+) MeV',log))
    ok=code==0 and counts and int(counts[-1])==n and (d/'edep.bin').exists() and unscored/(3000*n)<1e-6
    status=dict(exit_code=code,accepted=bool(ok),histories=n,actual_histories=int(counts[-1]) if counts else None,
        seconds=time.monotonic()-start,binary=str(BINARY),binary_sha256=sha(BINARY),input_sha256=sha(d/'topas.txt'),
        unscored_energy_MeV=unscored,projectile=[6,12],energy_MeVu=250)
    (d/'status.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status),flush=True)
    if not ok:raise RuntimeError('TOPAS reference did not pass completion checks')
if __name__=='__main__':main()
