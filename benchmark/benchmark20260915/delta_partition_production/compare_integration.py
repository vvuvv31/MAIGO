"""Compare production replay against the accepted candidate (large local doses required)."""
from pathlib import Path
import numpy as np,json
R=Path(__file__).resolve().parent;old=R.with_name('delta_partition_review');out=[]
for case,*_ in json.loads((R/'benchmarks/cases.json').read_text()):
 a=np.fromfile(old/'benchmarks'/case/'dose_gpu.raw','<f4');b=np.fromfile(R/'benchmarks'/case/'dose_gpu.raw','<f4')
 q=json.loads((R/'benchmarks'/case/'gpu_status.json').read_text());assert q['complete'] and q['quality']['queue_overflow_count']==0
 error=float(abs(a-b).max()/a.max());assert error<1e-5
 out.append(dict(case=case,max_difference_over_peak=error,relative_integral_difference=float(b.sum(dtype='f8')/a.sum(dtype='f8')-1)))
a=np.fromfile(old/'RT07575_delta_partition_merged.raw','<f4');b=np.fromfile(R/'RT07575_delta_partition_merged.raw','<f4');error=float(abs(a-b).max()/a.max());assert error<1e-5
out.append(dict(case='RT07575',max_difference_over_peak=error,relative_integral_difference=float(b.sum(dtype='f8')/a.sum(dtype='f8')-1)))
(R/'integration_comparison.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
