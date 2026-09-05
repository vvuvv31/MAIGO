# RT06423 section-0 delta-tail strict Gamma

A TOPAS-derived transverse delta-electron tail was applied only to primary C12 electronic dose in Schneider section 0. The correction conserves energy, sends endpoints outside the aligned 3D scorer to the outside-grid ledger, and leaves water/package/stopping/MCS/global scale unchanged.

Twenty exact-split local RTX 2080 Ti shards completed: 20/20 accepted, zero overflow, 151,087,660 histories. With the frozen 50k seed-42, 10%-threshold, 0.5-mm trilinear search method, nominal metrics are Global/Local 1%/1mm = 98.426%/87.786% and Global/Local 3%/0mm = 99.804%/93.422%.

The independent homogeneous HU -1000 3D check reproduces TOPAS tail fraction (10.147% vs 10.243%) and tail median radius (9.485 vs 9.484 mm). This remains a declared research-scope approximation, not general electron transport. See `strict-gamma-summary.json` for hashes and limitations.
