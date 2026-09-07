"""Read-only parent survival audit using the repository packed wire schema."""
import argparse,json,re
from pathlib import Path
import numpy as np
from run_topas10x_gpu_benchmark import sha

def dtype_for(source,name):
    body=source.split('struct '+name+' {',1)[1].split('\n};',1)[0]
    types={'std::uint64_t':'<u8','std::uint32_t':'<u4','std::int32_t':'<i4',
           'std::uint16_t':'<u2','std::int16_t':'<i2','std::uint8_t':'u1','std::int8_t':'i1','float':'<f4','char':'S1'}
    fields=[]
    for typ,key,count in re.findall(r'\b(std::\w+|float|char)\s+(\w+)(?:\[(\d+)\])?\s*\{',body):
        fields.append((key,types[typ],(int(count),)) if count else (key,types[typ]))
    return np.dtype(fields,align=False)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();schema=Path('include/carbon/inelastic_package_v3.hpp');text=schema.read_text()
    hd=dtype_for(text,'Cinel03PackageHeader');ev=dtype_for(text,'Cinel03InteractionRecord');results={}
    for filename in ['cinel03_c12_targets_v2_1.bin','cinel03_secondary_targets_v2_1_14p.bin']:
        path=Path('data/schneider')/filename;h=np.fromfile(path,dtype=hd,count=1)[0]
        if bytes(h['magic'])!=b'CINPKG04' or h['version']!=4 or h['header_size']!=hd.itemsize or h['interaction_record_size']!=ev.itemsize:
            raise ValueError('Wire schema mismatch')
        offset=int(h['header_size'])+int(h['cell_count'])*int(h['index_record_size'])
        data=np.memmap(path,mode='r',dtype=ev,offset=offset,shape=(int(h['interaction_count']),))
        water=np.isin(data['target_element_z'],[1,8]);alive=(data['parent_status']==0)&(data['parent_energy_MeV']>1e-4)
        results[filename]=dict(sha256=sha(path),events=len(data),positive_surviving_parent_events=int(alive.sum()),
            water_target_events=int(water.sum()),water_positive_surviving_parent_events=int((alive&water).sum()),
            maximum_parent_energy_MeV=float(data['parent_energy_MeV'].max()))
    with a.out.open('x') as f:json.dump(dict(schema_sha256=sha(schema),results=results),f,indent=2)
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
