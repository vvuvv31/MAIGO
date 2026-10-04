# MAIGO native physics extraction kit

The current runtime has eight presets: proton/carbon × CT/water × broadbeam/minibeam.
Their input inventory is [../docs/package_manifest.json](../docs/package_manifest.json);
[package_manifest.json](package_manifest.json) is the installed-file size catalog used by
`verify_kit.py`. Large packages and native raw histories are not stored in Git.

The extraction reference is TOPAS 4.2.p3 with Geant4 11.3.2. Water stopping,
fluctuation, electron births and electron transport use TOPAS **Water_75eV**.
The G4_WATER material probe supplies composition/radiation length only.
A changed native version, material, production cut or physics list needs its own
extraction and dose validation. File presence/size checks do not establish physical validity.

| Runtime data | Extraction/build tools |
|---|---|
| Unified EM and delta moments | `tools/unified_em/` |
| Schneider primary/secondary CINEL03 and reaction rates | `tools/schneider/`, `package_tools/cinel02.py` |
| All-ion elastic samples and recoil stopping | `tools/general_ion_elastic/` |
| Copper primary/fragment nuclear and stopping data | `tools/minibeam_copper/` |
| Native neutron channels and recoil stopping | `tools/neutron/` |
| Global all-ion Urban tables | `tools/urban/` |
| Projectile electron births | `../scripts/build_ion_electron_birth_native.py`, `../scripts/extract_ion_electron_birth.py`, `../scripts/assemble_electron_birth_bank.py` |
| Shared charged/photon operators | `../scripts/compile_electron_response.py`, `../scripts/compile_photon_response.py` |
| Shared electron joint response | `../scripts/electron_joint_compiler.py`, `../scripts/build_schneider_joint.py` |

The electron registry has 82 Schneider density nodes and one Water_75eV node.
Proton and carbon have separate birth banks; charged/photon and Schneider joint
responses are shared. Water currently has no joint-response acceleration node.
Extraction native sources live in `../benchmark/electron_response_model/native/`.
Retained native raw histories are provenance inputs; old benchmark drivers are
historical evidence, not runtime compatibility interfaces.

Build extension-enabled TOPAS binaries in separate directories:

```bash
extensions/build_topas_extensions.sh \
  --topas-source /path/to/OpenTOPAS-4.2.3 \
  --build-dir /work/topas-em \
  --profile unified-em \
  --geant4-dir /path/to/geant4/lib/cmake/Geant4
```

Use the matching `nuclear-copper` or `all-ion-elastic` profile for those exporters.
The common and Unified EM exporters define different implementations of
`CarbonStoppingPowerNtuple`; they must be built separately. The frozen elastic
source snapshots preserve native reference provenance. Runtime tables are checked
for schema, dimensions, finite values, projectile/material coverage and the
applicable physics audits. No content-digest checker or digest manifest is required.

```bash
python3 extensions/verify_kit.py --require-packages
python3 extensions/tools/unified_em/verify_unified_em_data.py
python3 extensions/tools/general_ion_elastic/verify_all_ion_elastic.py
python3 extensions/tools/schneider/verify_schneider_v2_1_data.py
python3 scripts/check_electron_packages.py
```

`mm.md` and `mm_zh.md` describe the current runtime, batch caches, package sizes and
acceptance status. Successful extraction, compilation or smoke tests alone do not
establish patient-dose acceptance.
