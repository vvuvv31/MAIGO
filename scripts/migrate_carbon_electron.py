#!/usr/bin/env python3
"""Losslessly migrate the independently validated C12 v3 layout to v4 storage.

Only the audited a12195b4 compiler identity is supported. Every input block
is SHA-checked; no graph or physics field is regenerated or approximated.
The full source-based checker must validate migrated artifacts before use.
"""
import argparse,ctypes,ctypes.util,hashlib,json,os,struct,sys,time
from pathlib import Path
from prepare_material_electron import fingerprint
OLD_VERSION='a12195b49ad9dfdd227a8803f0e637d19b93fb7bd37aec915bad968f6fd7f6aa'
class Buffer(ctypes.Structure):
    _fields_=[('data',ctypes.c_void_p),('size',ctypes.c_size_t),('pos',ctypes.c_size_t)]
class Encoder:
    def __init__(self):
        name=ctypes.util.find_library('zstd')
        if not name:raise RuntimeError('libzstd is required')
        z=self.z=ctypes.CDLL(name)
        declarations={'ZSTD_createCStream':([],ctypes.c_void_p),'ZSTD_freeCStream':([ctypes.c_void_p],ctypes.c_size_t),'ZSTD_initCStream':([ctypes.c_void_p,ctypes.c_int],ctypes.c_size_t),'ZSTD_CCtx_setPledgedSrcSize':([ctypes.c_void_p,ctypes.c_ulonglong],ctypes.c_size_t),'ZSTD_CCtx_setParameter':([ctypes.c_void_p,ctypes.c_int,ctypes.c_int],ctypes.c_size_t),'ZSTD_compressStream2':([ctypes.c_void_p,ctypes.POINTER(Buffer),ctypes.POINTER(Buffer),ctypes.c_int],ctypes.c_size_t),'ZSTD_isError':([ctypes.c_size_t],ctypes.c_uint),'ZSTD_getErrorName':([ctypes.c_size_t],ctypes.c_char_p)}
        for n,(a,r) in declarations.items():getattr(z,n).argtypes=a;getattr(z,n).restype=r
    def checked(self,n):
        if self.z.ZSTD_isError(n):raise RuntimeError(self.z.ZSTD_getErrorName(n).decode())
        return n
    def copy(self,source,dest,expected_bytes):
        z=self.z;ctx=z.ZSTD_createCStream()
        if not ctx:raise MemoryError('zstd context')
        digest=hashlib.sha256();copied=0;output=ctypes.create_string_buffer(1024*1024)
        try:
            self.checked(z.ZSTD_initCStream(ctx,1));self.checked(z.ZSTD_CCtx_setPledgedSrcSize(ctx,expected_bytes))
            self.checked(z.ZSTD_CCtx_setParameter(ctx,201,1)) # ZSTD_c_checksumFlag
            with source.open('rb') as f,dest.open('xb') as g:
                while True:
                    chunk=f.read(8*1024*1024);digest.update(chunk);copied+=len(chunk)
                    storage=ctypes.create_string_buffer(chunk) if chunk else None
                    incoming=Buffer(ctypes.cast(storage,ctypes.c_void_p) if storage else None,len(chunk),0)
                    ending=not chunk
                    while True:
                        outgoing=Buffer(ctypes.cast(output,ctypes.c_void_p),len(output),0)
                        remaining=self.checked(z.ZSTD_compressStream2(ctx,ctypes.byref(outgoing),ctypes.byref(incoming),2 if ending else 0))
                        g.write(memoryview(output).cast('B')[:outgoing.pos])
                        if (ending and remaining==0) or (not ending and incoming.pos==incoming.size):break
                    if ending:break
                g.flush();os.fsync(g.fileno())
            if copied!=expected_bytes:raise RuntimeError('Source changed during encoding')
            return digest.hexdigest()
        finally:z.ZSTD_freeCStream(ctx)

def key(table,version,format_number):
    pins=[table[k] for k in ['data_sha256','metadata_sha256','state_reference_sha256','state_reference_metadata_sha256']]
    if any(len(p)!=64 or any(c not in '0123456789abcdef' for c in p) for p in pins):raise ValueError('Invalid source pins')
    # Fixed audited x86_64 little-endian ABI, checked below and by the C++ reader.
    text=' '.join([f'material-prepared-carbon-em-v{format_number}',version,'6','12',*pins,'232','8','24','40','40','8','1'])
    return hashlib.sha256(text.encode()).hexdigest()

def main():
    raise RuntimeError("Legacy v3-to-v4 migration retired; use migrate_carbon_bitmap.py for v4-to-v5")
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--index',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--codec',choices=['raw','zstd'],default='zstd');p.add_argument('--hardlink',action='store_true');p.add_argument('--begin',type=int,default=0);p.add_argument('--end',type=int);a=p.parse_args()
    if sys.byteorder!='little' or struct.calcsize('P')!=8:raise RuntimeError('Migration requires the audited little-endian 64-bit ABI')
    if a.hardlink and a.codec!='raw':raise ValueError('Hardlinks require raw codec')
    index=json.loads(a.index.read_text())
    if index['projectile']!=[6,12]:raise ValueError('Migration requires a C12 index')
    tables=sorted(index['tables'],key=lambda t:(t['material_identity']['section_id'],t['material_identity']['runtime_density_g_cm3']))
    version=fingerprint();a.cache.mkdir(parents=True,exist_ok=True);encoder=Encoder() if a.codec=='zstd' else None
    report=[];begin=time.perf_counter()
    for i in range(a.begin,min(a.end if a.end is not None else len(tables),len(tables))):
        t=tables[i];old=key(t,OLD_VERSION,3);new=key(t,version,4);source=a.source/old;dest=a.cache/new
        if dest.exists():raise FileExistsError(f'Refuse to overwrite migrated artifact: {dest}')
        raw=(source/'manifest').read_bytes();pin=(source/'manifest.sha256').read_text().strip()
        if hashlib.sha256(raw).hexdigest()!=pin:raise RuntimeError('Old manifest SHA mismatch')
        lines=raw.decode().splitlines();head=lines[0].split()
        if len(head)!=7 or head[:2]!=['MAIGO_CARBON_EM_3',old]:raise RuntimeError('Old artifact identity mismatch')
        if int(head[2])!=t['material_identity']['section_id'] or float(head[3])!=t['material_identity']['runtime_density_g_cm3']:raise RuntimeError('Old material identity mismatch')
        sources=int(head[6]);nodes=int(head[5])
        if sources<1 or sources>1024 or nodes<1:raise RuntimeError('Old artifact count invalid')
        pattern=[('channels',40)]+[('compact_records',144),('compact_tracks',64),('compact_media',32),('next',8),('child_offsets',8),('children',8),('certified_child_energy',8),('edges',8),('offsets',8),('shared_starts',8),('logical_crossing_count',8),('crossings32',4)]*sources+[('birth_channels',40),('birth_samples',24)]
        if len(lines)!=1+len(pattern):raise RuntimeError('Old descriptor count mismatch')
        staging=a.cache/(new+'.partial.'+str(os.getpid())+'.'+str(time.time_ns()));staging.mkdir();total=stored=0
        for b,(line,expected) in enumerate(zip(lines[1:],pattern)):
            name,n,w,digest=line.split();n=int(n);w=int(w)
            if (name,w)!=expected or n<0 or n>sys.maxsize//w:raise RuntimeError('Old descriptor mismatch')
            src=source/f'{b}.bin';size=n*w
            if src.stat().st_size!=size:raise RuntimeError('Old block size mismatch')
            dst=staging/(f'{b}.bin.zst' if encoder else f'{b}.bin')
            if encoder:actual=encoder.copy(src,dst,size)
            else:
                h=hashlib.sha256()
                with src.open('rb') as f:
                    if a.hardlink:
                        while chunk:=f.read(8*1024*1024):h.update(chunk)
                    else:
                        with dst.open('xb') as g:
                            while chunk:=f.read(8*1024*1024):h.update(chunk);g.write(chunk)
                            g.flush();os.fsync(g.fileno())
                actual=h.hexdigest()
                if a.hardlink:os.link(src,dst)
            if actual!=digest:raise RuntimeError(f'Old block SHA mismatch: {src}')
            total+=size;stored+=dst.stat().st_size
        head[:2]=['MAIGO_CARBON_EM_4',new];manifest=(' '.join(head)+'\n'+'\n'.join(lines[1:])+'\n').encode()
        (staging/'manifest').write_bytes(manifest);(staging/'manifest.sha256').write_text(hashlib.sha256(manifest).hexdigest()+'\n');staging.rename(dest)
        report.append(dict(table=i,source_key=old,key=new,raw_bytes=total,stored_bytes=stored));print(json.dumps(report[-1]),flush=True)
    (a.cache/f'migration_{a.begin}_{a.end}.json').write_text(json.dumps(dict(source_version=OLD_VERSION,target_version=version,index_sha256=hashlib.sha256(a.index.read_bytes()).hexdigest(),codec=a.codec,seconds=time.perf_counter()-begin,entries=report,note='Every source block checked; require full source-based checker and GPU validation for promotion'),indent=2)+'\n')
if __name__=='__main__':main()
