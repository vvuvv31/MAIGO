"""Complete the existing origin partition with non-neutron/gamma neutral cache."""
import json,sys
from pathlib import Path
import prepare_water_parent_origin as prep
from run_topas10x_gpu_benchmark import sha

if __name__=='__main__':
    prep.MAPPING['neutral_origin']='other_neutral'
    prep.main()
    root=Path(sys.argv[sys.argv.index('--out')+1]).resolve()
    path=root/'manifest.json';m=json.loads(path.read_text())
    m['pins'][str(Path(__file__).resolve())]=sha(Path(__file__).resolve())
    m['limitations'].append('neutral_origin bucket captures neutral cache excluding neutron/gamma ancestry; it is not assigned to GPU charged categories.')
    path.write_text(json.dumps(m,indent=2))
