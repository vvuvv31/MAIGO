#!/usr/bin/env python3
"""Publish frozen independent26 CT dose results and link any later default migration."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
    root=ROOT/'out/electron_complete26_20261006';campaign=root/'gpu_complete26_v2'
    head=json.loads((campaign/'full_0p1_analysis/comparison.json').read_text());assert not head['preliminary']
    broad={s:json.loads((campaign/f'{s}_broadbeam_analysis/comparison.json').read_text()) for s in ['proton','carbon']}
    migration=ROOT/'docs/electron26_default_promotion_20261006.json'
    promoted=migration.exists() and json.loads(migration.read_text()).get('passed',False)
    r=dict(status='DOSE_FIRST_COMPLETE26_CT_RESULTS',default_promoted=promoted,runtime_density_nodes=0,old_bank_reads=0,
           shared_bytes=json.loads((root/'gpu_package_v1/manifest.json').read_text())['shared_bytes'],
           head_0p1=head,broadbeam=broad,carbon_minibeam='runtime smoke passed; dose accuracy deferred',
           acceptance_met=bool(head['relative_gamma_accepted'] and all(b['accuracy_accepted'] for b in broad.values())),
           approximations=json.loads((root/'gpu_package_v1/manifest.json').read_text())['approximations'])
    (root/'dose_summary_v2.json').write_text(json.dumps(r,indent=2)+'\n')
    p=ROOT/'docs/electron_complete26_20261006.json';doc=json.loads(p.read_text());doc['status']='COMPLETE26_CT_DOSE_FIRST_ACCEPTED' if r['acceptance_met'] else 'COMPLETE26_CT_DOSE_FIRST_REJECTED';doc['gpu_dose']['head_0p1']=head;doc['dose_first_acceptance_met']=r['acceptance_met'];doc['remaining']=[] if promoted else ['default migration smoke pending'];doc['default_promoted']=promoted;doc['default_migration_report']='docs/electron26_default_promotion_20261006.json';doc['further_physics_refinement_required_by_user_gamma_gate']=not r['acceptance_met'];p.write_text(json.dumps(doc,indent=2)+'\n')
    old=head['control_TOPAS'];new=head['candidate_TOPAS']
    text=f'''# Independent26 electron CT dose-first validation

## Scope and runtime model

This report records frozen independent26 dose-first validation. The later
default migration is documented in [the migration report](electron26_default_promotion_20261006.md). RT07575 is a head case. Proton/carbon CT broadbeam and minibeam
use one shared charged/photon/spatial bank and separate44-byte projectile
birth parameter files. The candidate opens zero old82/83-node bank files.
MAIEC026/MAIPH026 indices have exactly26 composition entries. Density
response uses continuous scalar fits and mass scaling, without density pairs.
The complete shared package is1,746,666,256 bytes (about1.75GB), dominated
by spatial joint response. Collision components alone are not the full package.

Charged hard outcomes use low-energy Penelope, high-energy Moller, complete
atomic cascades and elemental Seltzer-Berger spectra with native2BS angular
sampling. The existing composition-based spatial joint response preserves
lateral spreading. GPU dose scoring, electron geometry and loss gates remain
FP32. Group32, electron maximum step0.05mm, joint maximum0.25mm and ion
maximum steps0.1mm minibeam/0.5mm broadbeam are unchanged; ion
fluctuations remain ON. Water material is Water_75eV.

First-dose approximations: actual-cut hard-rate mass scaling, Penelope laws
at each composition calibration density, and calibrated photon conditional
terminals with continuous mass-scaled flight rates. These are not claims of
independently exact continuous-density native terminal/rate agreement.

## CT broadbeam dose results

Both use TOPAS local1%/1mm>99%, search0.1mm, full-grid reference>=10%max.
Normalization is absolute; no fitting or registration.

| Species | Primaries | Old83 TOPAS | Independent26 TOPAS | Change in dose sum versus old GPU |
|---|---:|---:|---:|---:|
| Proton |25,063,000|{broad['proton']['control_TOPAS']['pass_percent']:.6f}%|{broad['proton']['candidate_TOPAS']['pass_percent']:.6f}%|{broad['proton']['dose_sum_change_percent']:+.6f}%|
| Carbon |12,963,818|{broad['carbon']['control_TOPAS']['pass_percent']:.6f}%|{broad['carbon']['candidate_TOPAS']['pass_percent']:.6f}%|{broad['carbon']['dose_sum_change_percent']:+.6f}%|

## RT07575 head0.1% dose-first comparison\n\nAll119 batches /236,086,480 primaries finish with matched seeds, spot\nallocations and all487 optimized spots. Runtime energy/EM/coverage/completion\naudits pass with zero queue overflow. TOPAS BODY including GTV, reference\n>=10% reference maximum, local3%/0.2mm, search0.02mm:\n\n- Original83: {old['pass_percent']:.8f}%; independent26: {new['pass_percent']:.8f}%.\n- Change: {head['TOPAS_pass_change_percentage_points']:+.8f} percentage points.\n- Matched GPU gamma: {head['against_control_pass_percent']:.8f}%.\n- RelativeL1: {100*head['relative_L1']:.8f}%; dose-sum change: {head['dose_sum_change_percent']:+.8f}%.\n- Local absolute relative-difference percentiles: {head['local_percent_quantiles']}.\n- Resident throughput: old {head['timing']['control']['resident_primary_s']:.1f}/s,\n  candidate {head['timing']['candidate']['resident_primary_s']:.1f}/s. Controls\n  were reused; this is not an interleaved timing comparison or a speedup claim.\n\nThe user's relative minibeam gate passes: {head['relative_gamma_accepted']}.\nAbsolute TOPAS>99% passes: {new['passed']}; report this independently.\nTogether with both broadbeam>99% gates, dose-first acceptance is\n{r['acceptance_met']}. Retained density/conditional-law approximations are\nlisted in the candidate manifest; the user's instruction does not require\nmore component refinement when these gamma changes are small. The dose experiment precedes the separate default migration and archival;\nsee the migration report for installed package paths and eight-preset smokes.\nEvidence: out/electron_complete26_20261006/gpu_complete26_v2/full_0p1_analysis/.

## Runtime and component evidence

All four CT scenarios pass2x50,000-primary runtime smokes. Carbon minibeam
uses the separate lung CT benchmark; accuracy remains deferred. File-open
tracing verifies no old density-bank reads. The first GPU smoke's34 coverage
misses were caused by CPU/device log rounding at3/52 physical endpoints;
a two-ulp logdomain guard fixes these without changing physical density bounds.
The failure and independent endpoint replay are preserved in failed.md.

Native charged full-event checks pass141 conditions across26 compositions;
atomic relaxation passes96 cases; repaired SB/2BS passes738 conditions.
Continuous Penelope density fits and exact native Kvasov rate-ratio correction
also pass their independent component checks, but are not used in this first
GPU candidate. Gamma acceptance does not retroactively certify every component
approximation. No more component refinement is required by the user's current
small-gamma-change instruction when the dose gates above pass.

Reproduction: scripts/build_electron_complete26_candidate.py,
scripts/validate_electron_complete26_fullplan.py,
scripts/validate_electron_complete26_ct.py and
scripts/report_electron_complete26.py. The source/candidate frozen binary is
v2. A later v4 rebuild changes only source-byte diagnostic double counting;
its closeout smoke is recorded separately. This frozen dose experiment precedes the default migration. See the migration
report for installed paths, archived references and the water mapping repair.
'''
    p=ROOT/'docs/electron_complete26_20261006.md';p.write_text(text)
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    a=np.load(campaign/'full_0p1_analysis/dose_arrays.npz');o=a['old'];n=a['new'];mask=a['mask'];y=int(np.argmax(mask.sum(axis=(1,2))))
    local=np.full_like(o[y],np.nan);np.divide(100*(n[y]-o[y]),o[y],out=local,where=mask[y]&(o[y]>0))
    fig,axes=plt.subplots(1,3,figsize=(14,5),constrained_layout=True)
    vmax=max(o[y].max(),n[y].max())
    for ax,data,title in zip(axes,[o[y].T,n[y].T,local.T],['Original83 dose (Gy)','Independent26 dose (Gy)','Local dose change (%)']):
        im=ax.imshow(data,origin='lower',cmap='RdBu_r' if ax is axes[2] else 'inferno',vmin=-1 if ax is axes[2] else 0,vmax=1 if ax is axes[2] else vmax);ax.set_title(title);ax.set_xlabel('Reference grid x index');ax.set_ylabel('Reference grid z index');fig.colorbar(im,ax=ax)
    fig.suptitle(f'RT07575 head0.1% full plan; reference slice{y}; difference shown in gamma mask')
    fig.savefig(campaign/'full_0p1_analysis/dose_plane_comparison.png',dpi=160);plt.close(fig)
    print(json.dumps({k:r[k] for k in ['status','acceptance_met','runtime_density_nodes','old_bank_reads','shared_bytes']},indent=2))
if __name__=='__main__':main()
