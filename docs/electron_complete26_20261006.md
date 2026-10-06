# Independent26 electron CT dose-first validation

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
| Proton |25,063,000|99.883408%|99.865655%|-0.002115%|
| Carbon |12,963,818|99.794424%|99.790508%|-0.012787%|

## RT07575 head0.1% dose-first comparison

All119 batches /236,086,480 primaries finish with matched seeds, spot
allocations and all487 optimized spots. Runtime energy/EM/coverage/completion
audits pass with zero queue overflow. TOPAS BODY including GTV, reference
>=10% reference maximum, local3%/0.2mm, search0.02mm:

- Original83: 98.90331377%; independent26: 98.90008030%.
- Change: -0.00323347 percentage points.
- Matched GPU gamma: 100.00000000%.
- RelativeL1: 0.03505711%; dose-sum change: +0.00023755%.
- Local absolute relative-difference percentiles: {'50': 0.021694094949466978, '95': 0.11974207653964573, '99': 0.19871121464256608, '99.9': 0.3496046459578124, '100': 1.574920814992798}.
- Resident throughput: old 78671.2/s,
  candidate 64565.7/s. Controls
  were reused; this is not an interleaved timing comparison or a speedup claim.

The user's relative minibeam gate passes: True.
Absolute TOPAS>99% passes: False; report this independently.
Together with both broadbeam>99% gates, dose-first acceptance is
True. Retained density/conditional-law approximations are
listed in the candidate manifest; the user's instruction does not require
more component refinement when these gamma changes are small. This dose experiment precedes the separate default migration and archival;
see the migration report for installed package paths and eight-preset smokes.
Evidence: out/electron_complete26_20261006/gpu_complete26_v2/full_0p1_analysis/.

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
