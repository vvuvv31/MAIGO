// SYCL transport helpers for Unified EM and independent beamline/recoil modules. Do not compile this file as a standalone translation unit.
//
// Include context (required):
//   - inside `namespace carbon` (NOT the anonymous namespace)
//   - SYCL headers and carbon/* headers already included by the parent .cpp

// Device dose buffers and atomics use FP32. Host TransportResult remains double.
using DoseAtomicT = float;
using DepthAtomicT = DoseAtomicT;
