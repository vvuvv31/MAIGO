// SYCL transport helpers for Unified EM and independent beamline/recoil modules. Do not compile this file as a standalone translation unit.
//
// Include context (required):
//   - inside `namespace carbon` (NOT the anonymous namespace)
//   - SYCL headers and carbon/* headers already included by the parent .cpp

// Device dose buffers and atomics use FP32. Host TransportResult remains double.
using DoseAtomicT = float;
using DepthAtomicT = DoseAtomicT;

// LET is a separate, optional tally: unrestricted electronic stopping only,
// all transported charged ions, no nuclear vertex/cutoff/copper deposits.
// The track-length estimator weights stopping by expected electronic loss;
// it does not assign a divergent dE/dl to zero-length local deposits. Numerator
// and denominator remain additive across spots, fields, devices and batches.
inline void score_electronic_let(float* numerator, float* denominator,
                                 std::size_t voxel, float stopping_MeV_per_mm,
                                 float true_path_mm) {
    if (numerator == nullptr || !(stopping_MeV_per_mm > 0.0F) ||
        !(true_path_mm > 0.0F)) return;
    const float energy = stopping_MeV_per_mm * true_path_mm;
    using Atomic = sycl::atomic_ref<float, sycl::memory_order::relaxed,
        sycl::memory_scope::device, sycl::access::address_space::global_space>;
    Atomic(numerator[voxel]).fetch_add(energy * stopping_MeV_per_mm);
    Atomic(denominator[voxel]).fetch_add(energy);
}
