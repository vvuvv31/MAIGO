# Step 18 Evidence: Material Target Runtime Selection & CINEL03 GPU Replay

## Summary
- **Status**: PASSED (All 5 Gates 100% Pass)
- **Objective**: Use validated element partial rates to sample the interacting target element in Schneider CT media, then replay the matching C12 correlated fragmentation event on GPU.
- **Header & Implementation**:
  - [`include/carbon/schneider_target_sampler.hpp`](file:///mnt/sdb/wuwei/MAIGO/include/carbon/schneider_target_sampler.hpp)
  - [`src/schneider_target_sampler.cpp`](file:///mnt/sdb/wuwei/MAIGO/src/schneider_target_sampler.cpp)
  - [`include/carbon/inelastic_package_v3.hpp`](file:///mnt/sdb/wuwei/MAIGO/include/carbon/inelastic_package_v3.hpp) (`cinel03_find_event_device`)
- **Evidence Summary**: [`evidence/step-18/step18_target_runtime_summary.json`](file:///mnt/sdb/wuwei/MAIGO/evidence/step-18/step18_target_runtime_summary.json)

---

## Gate Results

| Gate | Description | Metric / Criterion | Result | Status |
|---|---|---|---|---|
| **Gate 1** | Categorical Target Distribution & Goodness-of-Fit | Pearson $\chi^2 < \chi^2_{\text{crit}}$ across soft tissue and bone sections | Section 8 $\chi^2 = 0.0066$ ($\text{df}=6$), Section 20 $\chi^2 = 0.00038$ ($\text{df}=8$) | **PASS** |
| **Gate 2** | Density Independence | Density $\rho$ cancels from target fractions, mean deviation $< 10^{-6}$ | Absolute difference $< 10^{-6}$ across $\rho \in [0.2, 2.5]\text{ g/cm}^3$ | **PASS** |
| **Gate 3** | Universal Element Library Reuse | Target $Z$ queries map to identical event blocks regardless of host medium | Oxygen ($Z=8$) in section 5, 8, 20 queries identical event indices | **PASS** |
| **Gate 4** | Fail-Closed Target Policy | Unsupported/missing targets throw in production; increment counter in audit mode | Production throws `std::runtime_error`; audit mode records missing target | **PASS** |
| **Gate 5** | CPU/GPU Bitwise Equivalence & Diagnostics | 1,000 parallel device queries produce bitwise identical target and event IDs | 1,000 / 1,000 matches, 0 lookup failures, atomic diagnostics valid | **PASS** |
| **Regression** | Previous Step Regression Suite | P1–P3 and Steps 00–17 tests pass 100% | All unit tests pass with zero regressions | **PASS** |

---

## Mathematical and Device Layout Architecture
1. **Host-Side Precomputation**:
   - `SchneiderTargetSampler`: Constructs a precompiled CDF table of dimension $25 \times 860 \times 13$, where $C_{12} = 1.0$ within exact floating-point tolerance.
   - Precomputes unscaled mass total rates $\Sigma_{\text{tot}}(E) = \sum_k \Sigma_k(E)$.
2. **Density Cancellation**:
   - Macroscopic hazard $\mu_{\text{tot}}(E) = \rho \cdot \Sigma_{\text{tot}}(E)$ governs step length and nuclear optical depth.
   - Fractional target probability $P(\text{target}=k) = \frac{\rho \Sigma_k(E)}{\rho \Sigma_{\text{tot}}(E)} = \frac{\Sigma_k(E)}{\Sigma_{\text{tot}}(E)}$ is strictly density-invariant.
3. **SYCL Device Function**:
   - `sample_schneider_target_device` evaluates the uniform grid node in $O(1)$ and performs a 13-element linear comparison loop.
   - `cinel03_find_event_device` performs binary search over `energy_nodes` in $O(\log N)$ on GPU device, retrieving the sampled event index.
   - Atomic diagnostic tracking in `SchneiderTargetDiagnostics` logs all section, target Z, and energy distributions.
