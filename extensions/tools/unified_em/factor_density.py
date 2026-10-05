"""Factor existing native ion EM data into 26 composition bases and sparse density corrections.

No new physical samples, dose scaling, or hidden native-package fallback. Native
cut-sensitive splines/record scalars and discrete model switches remain explicit.
The current loader reconstructs the existing device layout for dose validation.
"""
from pathlib import Path
import argparse, json, struct, time
import numpy as np

EXPONENTS = np.array([1, 1, 1, -1, 1, 0, 1, 1, 1, 0], dtype=np.int32)

def runs(values):
    values = np.asarray(values, dtype='<u4')
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
    return np.column_stack([starts, values[starts]]).astype('<u4')

def write_runs(stream, values):
    a = runs(values)
    stream.write(struct.pack('<I', len(a)))
    stream.write(a.tobytes())

def sparse_correction(base, target, tolerance):
    # Fit in the original knot ordinal; reconstruction preserves the original
    # energy grid and subsequent runtime energy interpolation.
    base, target = base.astype(np.float64), target.astype(np.float64)
    # Ratios avoid catastrophic cancellation where density/cut shifts make
    # a target much smaller than the scaled composition base. Zero-base
    # regions encode the absolute target instead.
    residual = np.divide(target, base, out=target.copy(), where=base != 0).astype('<f4')
    if not np.isfinite(residual).all():
        raise ValueError('Nonfinite density correction coefficient')
    denominator = np.maximum(np.abs(target), 1e-30)
    selected = {0, len(target)-1}
    stack = [(0, len(target)-1)]
    while stack:
        lo, hi = stack.pop()
        if hi <= lo+1:
            continue
        idx = np.arange(lo+1, hi)
        w = (idx-lo)/(hi-lo)
        coefficient = residual[lo] + w*(float(residual[hi])-float(residual[lo]))
        pred = np.where(base[idx] != 0, base[idx]*coefficient, coefficient).astype('<f4')
        pred[target[idx] == 0] = 0  # exact zero support is encoded separately
        errors = abs(pred.astype(float)-target[idx])/denominator[idx]
        worst = int(np.argmax(errors))
        if errors[worst] > tolerance:
            k = int(idx[worst]); selected.add(k); stack += [(lo,k),(k,hi)]
    indices = np.array(sorted(selected), dtype='<u4')
    pairs = np.empty(len(indices), dtype=np.dtype([('index','<u4'),('delta','<f4')]))
    pairs['index'] = indices; pairs['delta'] = residual[indices]
    idx = np.arange(len(target))
    correction = np.interp(idx, indices, residual[indices])
    reconstructed = np.where(base != 0, base*correction, correction).astype('<f4')
    reconstructed[target == 0] = 0
    nonnegative = bool(np.all(target >= 0))
    if nonnegative:
        reconstructed = np.maximum(reconstructed, 0)
    error = float(np.max(abs(reconstructed.astype(float)-target)/denominator))
    # Allow a small FP32 interpolation rounding margin at knot endpoints.
    if error > max(2*tolerance, 2e-6):
        raise ValueError(f'Correction reconstruction exceeds budget: {error}')
    return pairs, error

def write_correction(stream, base, target, tolerance):
    pairs, error = sparse_correction(base, target, tolerance)
    stream.write(struct.pack('<I',len(pairs))); stream.write(pairs.tobytes())
    write_runs(stream, target == 0)
    return len(pairs), error

def factor(source, delta_source, output, delta_output, tolerance):
    start = time.monotonic()
    with source.open('rb') as f:
        magic, version, nm, ns, nr, nn, nv = struct.unpack('<8s6I',f.read(32))
    assert magic == b'EMJOINT1' and version == 1 and ns == 18 and nr == nm*ns
    material_offset = 32
    mats = np.memmap(source,mode='r',offset=material_offset,shape=(nm,),dtype=[('section','<i4'),('density','<f4')])
    species_offset = material_offset+8*nm
    species = np.memmap(source,mode='r',offset=species_offset,shape=(ns,2),dtype='<u4')
    record_offset = species_offset+8*ns
    records = np.memmap(source,mode='r',offset=record_offset,shape=(nr,),dtype=[('f','<f4',(14,)),('u','<u4',(12,))])
    node_offset = record_offset+104*nr
    nodes = np.memmap(source,mode='r',offset=node_offset,shape=(nn,13),dtype='<f4')
    vectors = np.memmap(source,mode='r',offset=node_offset+52*nn,shape=(nv,6),dtype='<f4')
    with delta_source.open('rb') as f:
        dm,dv,dn=struct.unpack('<8s2I',f.read(16))
    assert dm==b'EMDMOMT2' and dv==2 and dn==nn
    delta=np.memmap(delta_source,mode='r',offset=16,shape=(nn,2),dtype='<f4')
    bases={}
    for section in sorted(set(mats['section'])):
        ids=np.flatnonzero(mats['section']==section)
        bases[int(section)]=int(ids[len(ids)//2])
    assert len(bases)==26
    output.parent.mkdir(parents=True,exist_ok=True);delta_output.parent.mkdir(parents=True,exist_ok=True)
    metrics={'max_node_relative_error':0.,'max_delta_relative_error':0.,'node_correction_knots':0,'delta_correction_knots':0}
    with output.open('xb') as out,delta_output.open('xb') as dout:
        out.write(struct.pack('<8s6I',b'EMFAC001',1,nm,ns,nr,nn,nv))
        out.write(mats.tobytes());out.write(species.tobytes());out.write(records.tobytes())
        # Native spline intervals are a small, cut-sensitive part of the package;
        # keep them exact in this first independent dose-validation prototype.
        out.write(vectors.tobytes())
        dout.write(struct.pack('<8s3I',b'EMDFAC01',1,nr,nn))
        for record_id in range(nr):
            material_id,ion=divmod(record_id,ns)
            base_material=bases[int(mats[material_id]['section'])];base_id=base_material*ns+ion
            offset,count=map(int,records[record_id]['u'][:2]);bo,bc=map(int,records[base_id]['u'][:2])
            assert count==bc and np.array_equal(nodes[offset:offset+count,0],nodes[bo:bo+bc,0])
            ratio=np.float32(mats[material_id]['density']/mats[base_material]['density'])
            out.write(struct.pack('<If',base_id,ratio))
            dout.write(struct.pack('<IIIIf',offset,count,record_id,base_id,ratio))
            if base_id==record_id:
                out.write(nodes[offset:offset+count].tobytes());dout.write(delta[offset:offset+count].tobytes())
            else:
                for column,power in enumerate(EXPONENTS,1):
                    scale=np.float32(ratio if power==1 else (1/ratio if power==-1 else 1))
                    baseline=(nodes[bo:bo+count,column]*scale).astype('<f4')
                    k,e=write_correction(out,baseline,nodes[offset:offset+count,column],tolerance)
                    metrics['node_correction_knots']+=k;metrics['max_node_relative_error']=max(metrics['max_node_relative_error'],e)
                for column in (11,12):
                    write_runs(out,nodes[offset:offset+count,column].astype('<u4'))
                for column in range(2):
                    baseline=(delta[bo:bo+count,column]*ratio).astype('<f4')
                    k,e=write_correction(dout,baseline,delta[offset:offset+count,column],tolerance)
                    metrics['delta_correction_knots']+=k;metrics['max_delta_relative_error']=max(metrics['max_delta_relative_error'],e)
            if record_id%180==0:
                print(f'{record_id}/{nr} records, elapsed {time.monotonic()-start:.1f}s',flush=True)
    report=dict(schema='EMFAC001 + EMDFAC01',scope='26 native composition bases + sparse density corrections; host reconstruction for unchanged GPU layout',reference_source=str(source),reference_delta_source=str(delta_source),materials=nm,composition_bases=len(bases),base_material_indices=bases,ions=ns,records=nr,nodes=nn,tolerance=tolerance,native_record_scalars_and_splines='retained exactly',discrete_switches_and_zero_support='retained exactly',fresh_native_extraction=False,reference_bytes=source.stat().st_size+delta_source.stat().st_size,compact_bytes=output.stat().st_size+delta_output.stat().st_size,em_bytes=output.stat().st_size,delta_bytes=delta_output.stat().st_size,seconds=time.monotonic()-start,validation_status='candidate; component/full-plan dose acceptance pending',**metrics)
    report['size_reduction_percent']=100*(1-report['compact_bytes']/report['reference_bytes'])
    output.with_suffix('.factorization.json').write_text(json.dumps(report,indent=2)+'\n')
    for original,destination,schema,size_key in [(source,output,'EMFAC001','package_bytes'),(delta_source,delta_output,'EMDFAC01','bytes')]:
        metadata=json.loads(original.with_suffix('.json').read_text())
        metadata.update(schema=schema,version=1,encoding='26 composition bases + sparse density corrections; dense host reconstruction',composition_bases=26,reconstruction_tolerance=tolerance,native_source=str(original),validation_status=report['validation_status'])
        metadata[size_key]=destination.stat().st_size
        if schema=='EMDFAC01':metadata['layout']='EMDFAC01 header, record directories, base values and sparse density corrections'
        destination.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('data/archive/ion_em_native_20261005/unified_em_v1.bin'))
    p.add_argument('--delta-source',type=Path,default=Path('data/archive/ion_em_native_20261005/unified_em_delta_moments_v2.bin'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--delta-output',type=Path,required=True)
    p.add_argument('--tolerance',type=float,default=5e-5)
    a=p.parse_args();assert 0<a.tolerance<.01
    factor(a.source,a.delta_source,a.output,a.delta_output,a.tolerance)
