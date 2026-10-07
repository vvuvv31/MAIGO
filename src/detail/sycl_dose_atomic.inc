// SYCL transport helpers for Unified EM and independent beamline/recoil modules. Do not compile this file as a standalone translation unit.
//
// Include context (required):
//   - inside `namespace carbon` (NOT the anonymous namespace)
//   - SYCL headers and carbon/* headers already included by the parent .cpp

// Global dose scorer scalar for device atomics / USM dose buffers.
// CARBON_DOSE_FP32=1 (CMake) uses float atomics — much faster on consumer NVIDIA
// GPUs (Turing FP64 is ~1/32 rate). Host TransportResult remains double.
using DoseAtomicT = float;
inline constexpr bool k_dose_atomic_fp32 = true;

// Preserve the pre-candidate scalar when the optional model is off.
// Longitudinal diagnostics already require the explicit FP64 build.
using DepthAtomicT = DoseAtomicT;

