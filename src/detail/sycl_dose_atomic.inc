// Shared SYCL transport helpers included by transport_sycl_legacy.cpp and
// transport_sycl.cpp. Do not compile this file as a standalone translation unit.
//
// Include context (required):
//   - inside `namespace carbon` (NOT the anonymous namespace)
//   - SYCL headers and carbon/* headers already included by the parent .cpp

// Global dose scorer scalar for device atomics / USM dose buffers.
// CARBON_DOSE_FP32=1 (CMake) uses float atomics — much faster on consumer NVIDIA
// GPUs (Turing FP64 is ~1/32 rate). Host TransportResult remains double.
#if defined(CARBON_DOSE_FP32)
using DoseAtomicT = float;
inline constexpr bool k_dose_atomic_fp32 = true;
#else
using DoseAtomicT = double;
inline constexpr bool k_dose_atomic_fp32 = false;
#endif

// 1-D depth tallies (depth dose, in-FOV depth dose) always accumulate in
// double: cross-bin redistribution models (e.g. the longitudinal delta tail)
// add many sub-ulp(float) shares to large per-bin totals, which float atomics
// silently drop. The 3-D voxel array above keeps the fast float path; 1-D
// arrays are tiny (hundreds of bins) so the extra traffic is negligible.
using DepthAtomicT = double;

// LET moment accumulation follows the same consumer-GPU-friendly default.
// Results are promoted to double on the host. Explicit validation builds can
// retain FP64 accumulation with -DCARBON_LET_FP64=ON.
#if defined(CARBON_LET_FP64)
using LetAtomicT = double;
inline constexpr bool k_let_atomic_fp32 = false;
#else
using LetAtomicT = float;
inline constexpr bool k_let_atomic_fp32 = true;
#endif
