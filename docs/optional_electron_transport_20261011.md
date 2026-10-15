# 历史记录：可选电子输运

此接口已于 2026-10-15 删除，以下为历史实现记录。当前实现见 [移除说明](no_electron_transport_20261015.md)。

# Explicit optional electron transport

User-authorized on 2026-10-11: electron transport is optional for every preset and custom configuration. `enable_electron_transport` defaults to `false` in the configuration struct and YAML parser; all eight YAML presets explicitly set it to false. Merely supplying electron package paths does not enable transport.

When disabled, charged/birth/photon/spatial-joint paths are consumed as supported YAML inputs but are not resolved or loaded. Electron cache, broadbeam selection and electron variance reduction are inactive. Ionization-electron energy uses the existing local-deposition path. Ion fluctuations, nuclear charged-secondary transport, Cu fluctuations, and explicit untracked neutral energy sinks are unchanged.

When enabled, specify `enable_electron_transport: true` and the existing complete projectile-matched strict26 charged/birth/photon/spatial-joint paths. CT broadbeam also requires `broadbeam_electron_tracking: true` and its existing region-selection settings. Existing package/schema/coverage/geometry checks apply; there is no substitution for absent data. Enabled electron transport includes electromagnetic photons. Water broadbeam has no electron region/data in its reference preset; enabling a master flag alone does not invent an electron package or a supported water selection.

Preset dependency inventories distinguish inactive optional electron inputs from required inputs. The installed data manifest still includes the electron assets, ready for explicitly enabled configurations. Runtime metadata records the master switch.

Validation: proton-minibeam CUDA build; host configuration loading of all eight defaults (all pipeline names enabled only in the host checker); missing master flag defaults off; missing electron paths ignored when off; explicit on preserves complete paths and rejects an empty mandatory charged index; omitted CT minibeam step resolves to 0.5 mm; 87 retired-interface checks; full installed asset hashes and preset dependencies regenerated. This is configuration validation, not eight-preset GPU/TOPAS physics acceptance. Evidence: `/mnt/sdb/wuwei/topas_full_plan/RT06423/proton_minibeam/validation_20261011/{check_opt_in.log,electron_switch_validation.json,interface_check.json,data_manifest_check.json}`.

RT06423 proton-minibeam validation runs 116,368,247 incident histories with electrons off, patient cap 0.5 mm and Cu cap 0.05 mm. The user subsequently requested the 0.5 mm CT minibeam default for both proton and carbon. Completed 0.1 mm batches are retained separately in `patient_step0p1_partial`; the complete 0.5 mm campaign restarts with the same plan allocation/seeds. No mixed-step doses are merged. When electron transport is explicitly enabled, existing spatial-joint step restrictions still apply. Completed case: BODY local3%/0.2mm gamma 98.8072% above prescription threshold 0.4Gy; GTV 99.2065%. Mean campaign throughput 236294/s on RTX2080Ti, wall 492.473s; GPU-only voxel SE RMS 1.8902%. Reference is the same weighted TOPAS DIJ used for optimization, not independent replay. All59 audits pass, zero overflow. The within-run 0.1mm partial campaign is not a matched full-dose acceptance baseline.
