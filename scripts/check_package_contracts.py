#!/usr/bin/env python3
"""Check explicit primary channel support and delta-sidecar validation."""
import argparse
import copy
import json
import os
import struct
import subprocess
import tempfile
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,required=True)
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
    env['CUDA_DISABLE_PTX_JIT']='1'
    with tempfile.TemporaryDirectory(prefix='maigo_package_contracts_') as temporary:
        temporary=Path(temporary)
        cfg=yaml.safe_load((ROOT/'config/carbon_water_broadbeam.yaml').read_text())
        for key,value in cfg.items():
            if isinstance(value,str) and value.startswith(('data/','inputs/')):cfg[key]=str(ROOT/value)
        package=Path(cfg['ct_schneider_primary_cinel03_file'])
        local_package=temporary/package.name;local_package.symlink_to(package)
        cfg['ct_schneider_primary_cinel03_file']=str(local_package)
        original_bundle=cfg['ct_schneider_physics_bundle_file']
        bundle=json.loads(Path(original_bundle).read_text())
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
        (temporary/channels.name).write_text(json.dumps(carbon))
        cfg['ct_schneider_primary_cinel03_file']=str(package)
        cfg['ct_schneider_physics_bundle_file']=original_bundle
        delta=bytearray(Path(cfg['em_delta_moments_file']).read_bytes())
        assert delta[:8]==b'EMDMOMT2' and len(delta)>16+4096*4
        local_delta=temporary/'delta.bin'
        cfg['em_delta_moments_file']=str(local_delta)
        file.write_text(yaml.safe_dump(cfg,sort_keys=False))
        for name,offset,replacement,expected in [
                ('delta_magic',0,b'INVALID!','Delta moments header mismatch'),
                ('delta_version',8,struct.pack('<I',3),'Delta moments header mismatch'),
                ('delta_nodes',12,struct.pack('<I',0),'Delta moments header mismatch'),
                ('delta_negative_tail',len(delta)-4,struct.pack('<f',-1),'Delta moments contain invalid values'),
                ('delta_nan_tail',len(delta)-4,struct.pack('<f',float('nan')),'Delta moments contain invalid values'),
                ('delta_inf_tail',len(delta)-4,struct.pack('<f',float('inf')),'Delta moments contain invalid values'),
                ('delta_truncated',len(delta)-4,None,'Delta moments header mismatch'),
                ('delta_missing',0,None,'No such file or directory')]:
            if name=='delta_missing':
                local_delta.unlink()
            elif replacement is None:
                local_delta.write_bytes(delta[:offset])
            else:
                payload=delta.copy();payload[offset:offset+len(replacement)]=replacement
                local_delta.write_bytes(payload)
            result=subprocess.run([str(args.binary.resolve()),'--config',str(file),'--histories','1'],cwd=ROOT,
                env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            assert result.returncode!=0 and expected in result.stdout and '[unified-em-audit]' not in result.stdout,(name,result.stdout)
            reports.append(dict(check=name,passed=True))
    args.output.write_text(json.dumps(dict(passed=True,checks=reports),indent=2)+'\n')
    print('Passed',len(reports),'package contract checks')
if __name__=='__main__':main()
