# Three-case BODY gamma search-step comparison

All values are pass percentages. Evaluation: BODY reference voxels with TOPAS dose >=10% of full-volume maximum. Global normalization uses that maximum; local uses the reference dose at each query. Search the full unmodified GPU dose with trilinear interpolation and no extrapolation. No dose rescaling.

20022516 and RT07575 use the final material-stopping runs; RT06423 uses its latest available complete exact-faces run (not the final material-stopping configuration).

## 20022516

Evaluated voxels: 806092; CT material alignment: 100.00000000%.

| Search step (mm) | G33 | L33 | G22 | L22 | G11 | L11 | G30 | L30 |
|---|---|---|---|---|---|---|---|---|
| 1.0 | 100.0000 | 99.7922 | 99.9955 | 96.4784 | 94.9720 | 62.0914 | 99.9968 | 93.8317 |
| 0.5 | 100.0000 | 99.9986 | 100.0000 | 99.8588 | 99.7000 | 88.1481 | 99.9968 | 93.8317 |
| 0.25 | 100.0000 | 99.9995 | 100.0000 | 99.9799 | 99.8585 | 98.9258 | 99.9968 | 93.8317 |
| 0.125 | 100.0000 | 99.9995 | 100.0000 | 99.9811 | 99.8630 | 99.5278 | 99.9968 | 93.8317 |

## RT06423

Evaluated voxels: 369266; CT material alignment: 100.00000000%.

| Search step (mm) | G33 | L33 | G22 | L22 | G11 | L11 | G30 | L30 |
|---|---|---|---|---|---|---|---|---|
| 1.0 | 100.0000 | 99.8719 | 99.9965 | 97.7805 | 93.6311 | 65.3069 | 99.9989 | 97.0414 |
| 0.5 | 100.0000 | 99.9986 | 100.0000 | 99.9190 | 99.7389 | 92.9544 | 99.9989 | 97.0414 |
| 0.25 | 100.0000 | 100.0000 | 100.0000 | 99.9986 | 99.9163 | 99.1805 | 99.9989 | 97.0414 |
| 0.125 | 100.0000 | 100.0000 | 100.0000 | 99.9997 | 99.9234 | 99.7568 | 99.9989 | 97.0414 |

## RT07575

Evaluated voxels: 350401; CT material alignment: 100.00000000%.

| Search step (mm) | G33 | L33 | G22 | L22 | G11 | L11 | G30 | L30 |
|---|---|---|---|---|---|---|---|---|
| 1.0 | 100.0000 | 99.8619 | 99.9940 | 97.4492 | 94.8642 | 66.1887 | 99.9951 | 97.1769 |
| 0.5 | 100.0000 | 99.9997 | 100.0000 | 99.9075 | 99.7634 | 90.6844 | 99.9951 | 97.1769 |
| 0.25 | 100.0000 | 100.0000 | 100.0000 | 99.9949 | 99.9249 | 99.1444 | 99.9951 | 97.1769 |
| 0.125 | 100.0000 | 100.0000 | 100.0000 | 99.9997 | 99.9292 | 99.7469 | 99.9951 | 97.1769 |

Checks: dose SHA256 and complete history counts verified; accepted shard overflow counters zero; DICOM/packed CT material alignment >99.9%; RT07575 BODY mask and overlapping prior gamma results reproduced exactly. 3%/0mm values are identical at every search step.

These are finite-lattice pass rates, not exact continuous gamma minima. BODY masks use DICOM voxel-centre polygon rasterization with signed-distance interpolation between contour planes. Full input hashes and configuration identifiers are in the case JSON files.
