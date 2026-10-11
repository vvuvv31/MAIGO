#!/usr/bin/env python3
"""Check installed strict26 packages and explicitly enabled preset paths.

No digests. GPU loaders additionally check numerical support and majorants.
"""
import argparse,json,math,os,struct
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
def require(ok,message):
    if not ok:raise ValueError(message)
def verify(root):
    registry=json.loads((root/'docs/electron_packages_manifest.json').read_text())
    require(registry['schema']==3,'Registry schema')
    require(json.loads((root/'data/em/electrons/manifest.json').read_text())==registry,'Installed registry mismatch')
    packages=registry['packages'];require(set(packages)=={'shared','proton','carbon'},'Three packages required')
    report={}
    for name,entry in packages.items():
        directory=root/entry['root'];spec=json.loads((directory/'manifest.json').read_text())
        require(spec==entry['manifest'],f'{name}: registry mismatch')
        require(spec['schema']==3 and spec['compositions']==26 and spec['runtime_density_nodes']==0,'Composition representation')
        for file,item in spec['files'].items():
            path=directory/file;require(path.resolve().is_relative_to(directory.resolve()),'Member outside package')
            require(path.is_file() and path.stat().st_size==item['bytes'],f'{name}: size mismatch {file}')
        total=sum(item['bytes'] for item in spec['files'].values());require(total==spec['total_bytes'],'Package size')
        if name!='shared':
            with (directory/'parameters.bin').open('rb') as f:magic,version,z,a=struct.unpack('<8s3I',f.read(20))
            require((magic,version)==(b'MAIEPR26',1) and [z,a]==spec['projectile'],'Projectile schema/identity')
            require((directory/'parameters.bin').stat().st_size==44,'Projectile layout')
            require(spec['requires']==[dict(package_id=packages['shared']['manifest']['package_id'])],'Shared dependency')
        else:
            for file,target in spec['runtime_aliases'].items():
                require(target==f'../{file[:-6]}/parameters.bin','Unexpected runtime alias')
                require(os.path.samefile(directory/file,directory/target),'Projectile alias must share the same payload')
            for file,magic,version in [('birth26.bin',b'MAIBCA26',1),('scalar26.bin',b'MAIESC26',1),('penelope26.bin',b'MAIPEN26',1),('brems26.bin',b'MAISB026',2),('rates26.bin',b'MAIERG26',1)]:
                with (directory/file).open('rb') as f:h=struct.unpack('<8s2I',f.read(16))
                require(h==(magic,version,26),f'{file}: schema/composition count')
            with (directory/'joint.bin').open('rb') as f:h=struct.unpack('<8sI',f.read(12))
            require(h==(b'MAIJNB02',25),'Joint Schneider composition coverage')
            for component,magic in [('charged','MAIEC026'),('photon','MAIPH026')]:
                lines=(directory/f'{component}.index').read_text().splitlines();require(lines[0].split()==[magic,'26'] and len(lines)==27,'Index count/schema')
                for expected,row in enumerate(lines[1:]):
                    section,density,file=row.split();require(int(section)==expected and math.isfinite(float(density)) and float(density)>0,'Composition identity')
                    require(file in spec['files'],'Unregistered operator')
                    with (directory/file).open('rb') as f:
                        if component=='charged':h=struct.unpack('<8sIIf',f.read(20));require(h[:3]==(b'MAIEL026',1,expected) and h[3]==float(density),'Angular composition identity')
                        else:h=struct.unpack('<8sIid',f.read(24));require(h==(b'MAIPHF01',1,expected,float(density)),'Photon composition identity')
        report[name]=dict(files=len(spec['files']),bytes=total,compositions=26,density_nodes=0,verified=True)
    require(sum(x['bytes'] for x in report.values())==registry['total_unique_bytes'],'Unique size')
    expected=dict(material_electron_birth_file='data/em/electrons/shared/birth26.bin',material_electron_joint_pilot_file='data/em/electrons/shared/joint.bin',material_electron_charged_index_file='data/em/electrons/shared/charged.index',material_photon_response_index_file='data/em/electrons/shared/photon.index')
    presets={}
    for species in ['proton','carbon']:
        for geometry in ['ct','water']:
            for beam in ['broadbeam','minibeam']:
                cfg=yaml.safe_load((root/f'config/{species}_{geometry}_{beam}.yaml').read_text())
                enabled=cfg.get('enable_electron_transport',False)
                require(type(enabled) is bool,'Electron transport switch must be boolean')
                if enabled:
                    require(all(cfg.get(k)==v for k,v in expected.items()),'Enabled preset does not select strict26')
                presets[f'{species}_{geometry}_{beam}']=dict(enabled=enabled,paths_checked=enabled)
    report['presets']=presets
    return report
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output',type=Path);a=p.parse_args()
    text=json.dumps(verify(a.root.resolve()),indent=2)+'\n'
    if a.output:a.output.write_text(text)
    print(text,end='')
