#!/usr/bin/env python3
"""Check explicit primary channel support."""
import argparse
import copy
import json
import os
import subprocess
import tempfile
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,default=ROOT/'build/carbon_mc')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    reports=[]
    channels=ROOT/'data/nuclear/schneider/carbon/final_states.channels.json'
    carbon=json.loads(channels.read_text())
    assert len(carbon['channels'])==13 and all(c['has_support'] is True for c in carbon['channels'])
    proton=json.loads((ROOT/'data/nuclear/schneider/proton/final_states.channels.json').read_text())
    assert all(type(c['has_support']) is bool for c in proton['channels'])
    assert any(c['has_support'] is False for c in proton['channels'])
    reports.append(dict(check='explicit_carbon_and_proton_support',passed=True))
    env={k:v for k,v in os.environ.items() if not k.startswith('CARBON_')}
    env['LD_LIBRARY_PATH']='/home/wuwei/sycl_workspace/llvm/build/install/lib'
    with tempfile.TemporaryDirectory(prefix='maigo_package_contracts_') as temporary:
        temporary=Path(temporary)
        cfg=yaml.safe_load((ROOT/'config/carbon_water_broadbeam.yaml').read_text())
        for key,value in cfg.items():
            if isinstance(value,str) and value.startswith(('data/','inputs/')):cfg[key]=str(ROOT/value)
        package=Path(cfg['ct_schneider_primary_cinel03_file'])
        local_package=temporary/package.name;local_package.symlink_to(package)
        cfg['ct_schneider_primary_cinel03_file']=str(local_package)
        bundle=json.loads(Path(cfg['ct_schneider_physics_bundle_file']).read_text())
        bundle['primary_package']['channels_file']=channels.name
        local_bundle=temporary/'bundle.json';local_bundle.write_text(json.dumps(bundle))
        cfg['ct_schneider_physics_bundle_file']=str(local_bundle)
        file=temporary/'case.yaml';file.write_text(yaml.safe_dump(cfg,sort_keys=False))
        for name,expected in [('missing_support','requires explicit boolean has_support'),
                ('nonboolean_support','requires explicit boolean has_support'),
                ('false_support','primary rate/package domain mismatch')]:
            payload=copy.deepcopy(carbon)
            if name=='missing_support':del payload['channels'][0]['has_support']
            if name=='nonboolean_support':payload['channels'][0]['has_support']='true'
            if name=='false_support':payload['channels'][0]['has_support']=False
            (temporary/channels.name).write_text(json.dumps(payload))
            result=subprocess.run([str(args.binary.resolve()),'--config',str(file),'--histories','1'],cwd=ROOT,
                env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            assert result.returncode!=0 and expected in result.stdout and '[unified-em-audit]' not in result.stdout,(name,result.stdout)
            reports.append(dict(check=name,passed=True))
    args.output.write_text(json.dumps(dict(passed=True,checks=reports),indent=2)+'\n')
    print('Passed',len(reports),'package contract checks')
if __name__=='__main__':main()
