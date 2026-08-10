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
