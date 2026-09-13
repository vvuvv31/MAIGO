# Joint EM integration, 2026-09-13

Research-only primary C12 in homogeneous unit-density water; default TOPAS unchanged.
Six 50k GPU runs: three full-physics (0.5 mm depth) and three pure-EM (0.1 mm depth).
Integration audit counts match the isolated candidate; zero overflow and sampling failure.
`regression.json` compares integrated vs isolated dose, not GPU vs TOPAS.
`full_physics_results.json` and `fine_em_results.json` compare the isolated candidate to TOPAS.
`component_analysis.json` partitions the full-physics integrated-dose deficit by origin;
this is not a causal attribution to a particular missing process.

`config_guards.txt` records rejected configuration tests. The existing CT/elastic guards
reject some combinations before the new model-specific guard is reached.
Three component CTest cases passed. High-statistics validation and default-path regression
remain pending. Raw 3D output and binaries remain in the local scratch directories.
No production, CT or minibeam acceptance is implied.
