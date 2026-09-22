# C4 transport contract (fix C4) — anchor 0f2c0ca + C4 work

## 1. Physical geometry (TOPAS reference vs MAIGO config)

TOPAS single-center (benchmark/carbonminibeam/run1.txt + aperture.txt,
beam along +Y mapped to MAIGO +Z):
- World: air, ±2 m.
- Snout: air cylinder R = 60 mm, HL = 30 mm, TransY = −30 mm.
- Aperture: copper, HL = 30 mm (60 mm thick); 0.5 mm slits at ±3.6 mm pitch
  (ctc), slit holes HLX = 0.25 mm each side; TransZ = 0 within snout.
- Water phantom Box: Water_75eV, 100×100×100 mm³ at TransY = 110 mm
  (Y ∈ [60, 160] mm), MaxStepSize 0.05 mm. Scorer 1000×1000×1 (0.1 mm).
- Water replay: entrance phase space at the water surface (Y = 60 mm);
  MAIGO `minibeam_water_entrance_world_y_mm = 60.0` matches.

MAIGO replay config: phantom_length 250 mm (longer than TOPAS 100 mm —
histories stop/cooldown inside; exit handling verified by survival counts),
voxel scorer 1000×1 (analysis projection of the TOPAS bins; scorer-only
difference, transport-identical).

## 2. Unified accepted step (contract holder: urban_v2_propose_and_sample)

Order (fixed, tested): external TRUE ceiling → fMinimal limit (R = MSC
mirror, §4) → true→geom → geometry truncation (min with boundary) →
geom→final-true inversion (branch-consistent delta) → DoIt gate →
endpoint postSafety → angle + displacement sampling → contract fields.
True length (loss/fluctuation/nuclear paths) and geometric length
(position advance) are never mixed: result carries proposed_t, final_t,
final_g, stable_delta separately; legacy consumers ignore the extras.

-State lifecycle: primary water — one state per history, boundary on first
  segment ever (single water entry, persistent tlimit ✓ G4); Cu — per
  traversal, re-entry-aware boundary ✓; secondary C12 — per-track
  persistent state + born flag, saved/restored across chunk suspend/resume
  (SecondaryResumeState.sec_urban_tlimit_mm/born), boundary on first Urban
  segment (G4 newborn/firstStep semantics ✓). first_step flag is written
  but intentionally unread (no reference behavior keys on it on fMinimal).
- Determinism: every draw is a pure function of
  (seed, history, outer, segment, dim); call order, batching, chunking, and
  replay transforms cannot change any address (tested: C4 determinism).
- Old macro-overlay (FE-tail/Highland) kept as explicit diagnostic
  baselines, byte-identical (legacy consumers untouched).

## 3. MSC currentRange mirror (executed reference)

G4UrbanMscModel caps the true step at exactly GetRange (16/16 over-range
probes, t_lim/R = 1.0000), cuts-independent, exactly linear:
water R = E/7.2, Cu R = E/64.5 (0.12–4800 MeV probed). Neither CSDA (3.3x
smaller at 250 MeV/u) nor restricted residual (66x smaller at 60 MeV/u).
Old restricted range clamped production steps below ~35 MeV and killed
scatter at end-of-range; the mirror restores reference behavior. Loss side
keeps the restricted table (current_range_mm has no readers — verified).
C12 only; unknown materials fail the limiter fast.

## 4. Remaining approximations (bounded follow-ups, NOT silent PASS)

A1. Material safety box == scorer extents (voxel_min/max + phantom_length
    passed as the Urban safety box). Numerically shared today; semantically
    must split (scorer res change must not move safety). Bound: box faces
    are ≥ 50 mm from the beam core; safety only matters within tlimit of a
    face. Follow-up: separate material box params + no-op test.
A2. boundary_path = subdivision remainder, not re-queried after direction
    change. Inside homogeneous water/Cu the true boundary is the outer box
    (reached only at faces); slit-edge Cu/air crossings inside the block
    use exact material boundaries (copper path). Bound: angular change per
    0.05 mm step ~mrad → boundary error O(1e-4 mm). Follow-up: per-segment
    box-exit re-query + dose re-validation.
A3. Loss allocated per macro step (deposited_MeV, straggling) while MSC
    subdivides; loss uses accepted t_final per segment where it matters
    (predictor), macro envelope otherwise. Unchanged by this round.
A4. Nuclear optical depth consumed over the macro path_step, not the MSC
    true length (rate·0.25 mm ≪ 1 per step; second-order).
A5. Lateral displacement is relocation only (position += R; never added to
    loss/nuclear lengths — verified by inspection).
A6. Curved-path interior: discrete condensed-history convention (endpoint
    displacement + chord advance), convergence vs step size covered by the
    maxstep scan matrix (C5/C6).
