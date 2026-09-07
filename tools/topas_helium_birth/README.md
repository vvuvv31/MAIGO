# Conditional helium birth diagnostic

`CarbonBirthOnlyNtuple` is a diagnostic clone of the installed
`CarbonCascadeNtuple` (original `.cc` SHA256
`4ebb0ac6ca06fa54a718932316363df7967408b421e1faf61d1083f1821c577b`).
Only the class name and extra hadronic cross-section query differ. Column 17
is deliberately zero/unmeasured; never use this output to compile rate tables.
Original extensions and physics packages are not modified.

Important: a scorer attached to Water with PropagateToChildren creates an
additional `Water_1x1x1` parallel scoring world when the ROI already has a
dose scorer. Jobs 2577/2578 give identical dose despite removal of the query,
but both differ from the no-scorer control. Thus the query hypothesis is
excluded for this observed difference. `prepare_water_birth_roi.py` instead
shares the existing `ROI_64x64x800` scoring copy. Verify dose identity before
interpreting the records; do not relax the identity check.

The records are **first-tracked conditional products**, not a complete vertex
census. First-step reactions, parents outside ROI, and overwritten parent
contexts can omit or misassociate records. ROI-only births are not full-water
births. GPU queued birth minus reaction import measures primary-queued energy,
not products discarded below cutoff. No statistical significance or package
defect follows from a single energy-sum comparison.

Build uses local `/home/wuwei/topas/extensions` and `topas-build`. The pre-build
binary is preserved as `/home/wuwei/topas/topas-frozen-d50f90504eafe2b2` (full SHA
`d50f90504eafe2b207c0cac309d14c61ad9a99de97321f4844d5388c9bd59eee`).
New executable SHA is
`1d7ba0a56d3841f2413c7e2adfd7f07445b42f842d777f3abb8c7ad91e3b3338`.
Old manifests still name the original executable location; use the preserved
binary to audit historical content, never rewrite historical manifests.

Run only through local Slurm, with all active work totaling at most 192 CPUs /
160 GiB. Preparation tools refuse existing directories. Analysis requires
completed jobs, frozen SHA checks and unchanged 3D total dose. Outputs are
diagnostic only, not production qualification.
