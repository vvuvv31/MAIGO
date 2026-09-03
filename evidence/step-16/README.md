# Step 16: Versioned Elemental-Target Event Package (CINEL03)

## Provenance
- **Validated Commit C**: `f14e573921a9b2f29b87de8f901f9952b3f567de`
- **Package Magic**: `CINPKG04`
- **Package Version**: `4`
- **Little-Endian Marker**: `0x01020304`
- **Header Size**: 136 bytes
- **Index Record Size**: 36 bytes
- **Interaction Record Size**: 476 bytes
- **Product Record Size**: 72 bytes
- **Energy Node Size**: 12 bytes
- **Checksum**: CRC32 over payload tables (`FLAG_CRC32_CHECKSUM = 1 << 5`)

---

## 1. Schema Specification & Decoupled Contract

1. **Key Contract**:
   ```text
   event key = projectile_Z, projectile_A, target_element_Z, energy_node
   material section is NOT part of the event key
   ```
   - In CINEL02, target isotopes were water-specific ($H=1, A=1$ and $O=8, A=16$) and material sections were entangled.
   - In CINEL03, the event key is strictly parameterized by `(projectile_z, projectile_a, target_element_z, energy_node)`.
   - Material section and density are completely absent from the event key and payload lookup.

2. **Payload Completeness**:
   - Retains one complete correlated interaction:
     - Parent final status, final energy, and direction vector.
     - All direct charged daughters with $Z$, $A$, kinetic energy, and direction.
     - Neutron and gamma secondary records.
     - Process local deposit ($E_{\text{local}}$) and non-ionizing deposit.
     - Actual target isotope $A$ when available recorded in payload (`target_a`).
     - Event provenance (run ID, thread ID, event ID, material name, process name, model name).

3. **Fail-Closed Target Contract**:
   - If a target element $Z$ is requested that does not exist in the package:
     - **Production Mode** (`audit_mode = false`): Immediately throws a hard `std::runtime_error` fail-closed exception.
     - **Audit Mode** (`audit_mode = true`): Increments a named counter and returns `UINT64_MAX` without throwing.

---

## 2. Acceptance Verification Results ([`tools/verify_step16_cinel03_schema.py`](file:///mnt/sdb/wuwei/MAIGO/tools/verify_step16_cinel03_schema.py))

```
================================================================================
Step 16 Acceptance Verification: CINEL03 Elemental-Target Schema
Mode: Read-Only Verification
================================================================================
[Gate 1] Validating CINEL03 / CINPKG04 Schema Specification & Record Contract...
  -> Header size = 136 bytes, Index = 36 bytes, Interaction = 476 bytes, Product = 72 bytes.
  -> Event key verified: projectile_Z, projectile_A, target_element_Z, energy_node.
  -> Gate 1 PASSED: Schema contract satisfies Step 16 specification.
[Gate 2] Testing Deterministic Serialization & C++/Python Interoperability...
  -> Python and C++ structures match bitwise. Serialization is 100% deterministic.
  -> Gate 2 PASSED.
[Gate 3] Running CTest Suite & Fail-Closed Guard Validations...
  -> test_step16 passed all assertions: missing target fail-closed, audit counter, corrupt offsets, bad CRC32, unsupported magic.
  -> Gate 3 PASSED.
[Gate 4] Checking Synthetic Correlated Replay on CPU and GPU...
  -> Correlated event replay verified on both CPU and SYCL GPU device.
  -> Gate 4 PASSED.
[Gate 5] Verifying Step 13, 14, and 15 Regressions...
  -> Step 13, 14, and 15 verification gates remain 100% passed.
  -> Gate 5 PASSED.
================================================================================
Overall Step 16 Acceptance: PASS
```
