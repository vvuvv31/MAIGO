// Shared SYCL transport helpers included by transport_sycl_legacy.cpp and
// transport_sycl.cpp. Do not compile this file as a standalone translation unit.
//
// Include context (required):
//   - inside `namespace carbon` (NOT the anonymous namespace)
//   - SYCL headers and carbon/* headers already included by the parent .cpp

// Device-side profile increment. Compiles to nothing unless
// CARBON_TRANSPORT_PROFILE is defined at build time.
inline void profile_add(std::uint64_t* counters,
                        const TransportProfileSlot slot,
                        const std::uint64_t n = 1) {
#ifdef CARBON_TRANSPORT_PROFILE
    if (counters == nullptr || n == 0) {
        return;
    }
    sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        atomic_counter(counters[static_cast<std::size_t>(slot)]);
    atomic_counter.fetch_add(n);
#else
    (void)counters;
    (void)slot;
    (void)n;
#endif
}

inline void profile_face(std::uint64_t* counters,
                         const bool is_primary,
                         const CtClampPath path) {
#ifdef CARBON_TRANSPORT_PROFILE
    profile_add(counters, is_primary ? TransportProfileSlot::primary_face_clamp_calls
                                     : TransportProfileSlot::secondary_face_clamp_calls);
    profile_add(counters, is_primary ? primary_face_slot(path) : secondary_face_slot(path));
#else
    (void)counters;
    (void)is_primary;
    (void)path;
#endif
}
