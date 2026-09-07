"""Verify source-scoring-only transport invariance against geometry candidate."""
import json
from pathlib import Path
from run_topas10x_gpu_benchmark import sha


def main():
    data=Path('/mnt/sda/wuwei')
    old=data/'ct_secondary_exact_faces_seed9196301_20260907/on'
    new=data/'ct_secondary_source_score_20260907/on'
    a=json.loads((old/'energy_ledger.json').read_text())
    b=json.loads((new/'energy_ledger.json').read_text())
    keys=[k for k,v in a['schneider_diagnostics'].items() if type(v)==int and 'MeV' not in k and 'mismatch' not in k]
    for k in keys:
        if a['schneider_diagnostics'][k]!=b['schneider_diagnostics'][k]:raise ValueError('Transport count changed: '+k)
    for k in ('queries','ordered_path_replays'):
        if a['electron_joint_response'][k]!=b['electron_joint_response'][k]:raise ValueError('Electron count changed')
    q=json.loads((new/'quality_report.json').read_text())
    if [f['code'] for f in q['failures']]!=['unvalidated_electron_joint_response'] or q['queue_overflow_count']:
        raise ValueError('Quality regression')
    for k in ('voxel_to_ingrid_ratio','grid_split_closure_ratio'):
        if abs(q[k]-1)>1e-6:raise ValueError('Closure regression: '+k)
    result=dict(status='FOCUSED_CHECKS_PASS_NOT_GAMMA_ACCEPTANCE',unchanged_count_keys=keys,
        voxel_to_ingrid_ratio=q['voxel_to_ingrid_ratio'],grid_split_closure_ratio=q['grid_split_closure_ratio'],
        pins={str(p):sha(p) for p in (old/'energy_ledger.json',new/'energy_ledger.json',new/'quality_report.json')},
        limitation='Float energy sums may differ by atomic addition order; this is not a full trajectory trace proof.')
    with (new.parent/'scoring_invariance.json').open('x') as f:json.dump(result,f,indent=2)
    print('Focused transport-count/closure checks PASS')


if __name__=='__main__':main()
