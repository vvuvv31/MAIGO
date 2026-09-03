# Step 17 Evidence: Generate C12 Correlated Events for All 13 Target Elements

## Summary
- **Status**: PASSED (All 5 Gates 100% Pass)
- **Target Elements**: 13 Schneider target elements (H, C, N, O, Na, Mg, P, S, Cl, Ar, K, Ca, Ti)
- **Campaign Execution**: Local SLURM `sbatch` on partition `compute` across node `ps` (13 jobs, 156 threads $\le 192$, 104 GiB $\le 160$ GiB)
- **Raw Campaign Output**: 1,560 worker binary files under `/mnt/sda/wuwei/cinel03-campaigns/production/` containing 602,277 events
- **Compiled Package**: `data/schneider/cinel03_c12_targets.bin` (46,208,476 bytes, SHA256 `cc9a328a76e18f9e866e9dc4296a69483656ed06fa35e668de0fd7f6aae5a036`)
- **Metadata**: `data/schneider/cinel03_c12_targets.metadata.json`
- **Rejection Audit**: `data/schneider/cinel03_c12_targets.rejection_audit.json` (27 physical INCL++ closure outliers recorded with full kinematics ledger)

---

## Gate Results

| Gate | Description | Metric / Criterion | Result | Status |
|---|---|---|---|---|
| **Gate 1** | Elemental Target Coverage & Aliasing | 13/13 Schneider elements present, 0 aliasing errors | 13/13 covered, 0 aliasing errors, 0 non-C12 projectiles | **PASS** |
| **Gate 2** | Energy Domain & Statistics | Range [1.0, 430.0] MeV/u, $\ge 1,000$ events/target, rejection $< 0.1\%$ | $\ge 1,694$ events/target, 30,113 total, rejection 0.0896% (27 events) | **PASS** |
| **Gate 3** | Kinematics & Product Roles | Unit direction norms, non-negative KE, valid roles, 0 closure violations | 30,113 events, 53,533 products checked: 0 norm violations, 0 role violations, 0 closure violations | **PASS** |
| **Gate 4** | Binary Packaging & Provenance | Magic `CINPKG04`, CRC32 verified, SHA256 match, complete metadata | Magic `CINPKG04`, SHA256 matches declared, 0 missing keys | **PASS** |
| **Gate 5** | C++ Table & GPU Compatibility | `InelasticPackageV3Table::from_binary` and `make_device_tables()` | Loads cleanly, all 13 element lookups succeed, compact tables generated | **PASS** |

---

## Elemental Distribution in Package

| Target Symbol | $Z$ | Event Count in Package | Total Raw Events Scanned | Natural Isotopes Observed |
|---|---|---|---|---|
| **H** | 1 | 2,591 | 51,828 | A=1 (99.99%), A=2 (0.01%) |
| **C** | 6 | 2,944 | 58,890 | A=12 (98.9%), A=13 (1.1%) |
| **N** | 7 | 2,499 | 49,972 | A=14 (99.7%), A=15 (0.3%) |
| **O** | 8 | 2,350 | 47,003 | A=16 (99.7%), A=17, A=18 |
| **Na** | 11 | 2,072 | 41,450 | A=23 (100.0%) |
| **Mg** | 12 | 2,461 | 49,221 | A=24 (79.0%), A=25 (9.9%), A=26 (11.1%) |
| **P** | 15 | 2,360 | 47,198 | A=31 (100.0%) |
| **S** | 16 | 2,333 | 46,661 | A=32 (95.0%), A=33, A=34 (4.3%), A=36 |
| **Cl** | 17 | 2,218 | 44,368 | A=35 (75.6%), A=37 (24.4%) |
| **Ar** | 18 | 2,148 | 42,946 | A=36, A=38, A=40 (99.6%) |
| **K** | 19 | 1,694 | 33,872 | A=39 (93.2%), A=40, A=41 (6.8%) |
| **Ca** | 20 | 2,126 | 42,508 | A=40 (96.9%), A=42, A=43, A=44 (2.1%), A=48 |
| **Ti** | 22 | 2,317 | 46,333 | A=46 (8.2%), A=47 (7.7%), A=48 (73.7%), A=49 (5.3%), A=50 (5.1%) |
