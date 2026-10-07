// SYCL transport helpers for Unified EM and independent beamline/recoil modules. Do not compile this file as a standalone translation unit.
//
// Include context (required):
//   - inside `namespace carbon` (NOT the anonymous namespace)
//   - SYCL headers and carbon/* headers already included by the parent .cpp

// Device-side profile increment. Compiles to nothing unless
// CARBON_TRANSPORT_PROFILE is defined at build time.
inline void profile_add(std::uint64_t* counters,
                        const TransportProfileSlot slot,
                        const std::uint64_t n = 1) {
    (void)counters;
    (void)slot;
    (void)n;
}

inline void profile_face(std::uint64_t* counters,
                         const bool is_primary,
                         const CtClampPath path) {
    (void)counters;
    (void)is_primary;
    (void)path;
}
