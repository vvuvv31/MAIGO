"""Prepare a TOPAS partition exposing primary/electron definition mismatch."""
import argparse,json
from pathlib import Path
import water_neutral_lineage as lineage
from run_topas10x_gpu_benchmark import sha

GROUPS={
 'total':[],
 'primary':['s:Sc/{name}/OnlyIncludeParticlesOfGeneration = "Primary"'],
 'neutral_lineage':['sv:Sc/{name}/OnlyIncludeIfParticleOrAncestorNamed = 2 "neutron" "gamma"'],
 'electrons_no_neutral':[
     'sv:Sc/{name}/OnlyIncludeParticlesNamed = 2 "e-" "e+"',
     'sv:Sc/{name}/OnlyIncludeIfParticleOrAncestorNotNamed = 2 "neutron" "gamma"'],
 'other_secondary_no_neutral':[
     's:Sc/{name}/OnlyIncludeParticlesOfGeneration = "Secondary"',
     'sv:Sc/{name}/OnlyIncludeParticlesNotNamed = 2 "e-" "e+"',
     'sv:Sc/{name}/OnlyIncludeIfParticleOrAncestorNotNamed = 2 "neutron" "gamma"']}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    root=p.parse_args().out.resolve()
    lineage.GROUPS=GROUPS
    lineage.prepare(root)
    path=root/'manifest.json';m=json.loads(path.read_text())
    m['pins'][str(Path(__file__).resolve())]=sha(Path(__file__).resolve())
    m['partition']='C12 parentID0; neutron/gamma ancestry; non-neutral-lineage electrons; other non-neutral-lineage secondaries'
    m['limitations']=['TOPAS primary tracks use restricted stopping; GPU primary includes locally assigned electronic loss.',
        'All non-neutral electrons give a conservative upper bound, not exact primary-electron ancestry.',
        'Other secondary group includes non-electron particles; do not label it primary or pure ion without checking.']
    path.write_text(json.dumps(m,indent=2))

if __name__=='__main__':main()
