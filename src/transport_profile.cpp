#include "carbon/transport_profile.hpp"

#include <sstream>

namespace carbon {

std::string TransportProfile::summary() const {
    if (!enabled) {
        return "Transport profile: disabled (rebuild with -DCARBON_ENABLE_TRANSPORT_PROFILE=ON)\n";
    }
    std::ostringstream out;
    out << "Transport profile counters:\n";
    const auto p = [&](const char* name, TransportProfileSlot slot) {
        out << "  " << name << ": " << get(slot) << '\n';
    };
    p("primary_steps", TransportProfileSlot::primary_steps);
    p("primary_ct_samples", TransportProfileSlot::primary_ct_samples);
    p("primary_mass_sp_lookups", TransportProfileSlot::primary_mass_sp_lookups);
    p("primary_sp_table_lookups", TransportProfileSlot::primary_sp_table_lookups);
    p("primary_face_clamp_calls", TransportProfileSlot::primary_face_clamp_calls);
    p("primary_face_three_axis", TransportProfileSlot::primary_face_three_axis);
    p("primary_face_short_step_skip", TransportProfileSlot::primary_face_short_step_skip);
    p("primary_face_homogeneous_skip", TransportProfileSlot::primary_face_homogeneous_skip);
    p("primary_face_near_z_face", TransportProfileSlot::primary_face_near_z_face);
    p("primary_face_near_z_homogeneous",
      TransportProfileSlot::primary_face_near_z_homogeneous);
    p("primary_dose_depth_atomics", TransportProfileSlot::primary_dose_depth_atomics);
    p("primary_dose_voxel_atomics", TransportProfileSlot::primary_dose_voxel_atomics);
    p("primary_straggling", TransportProfileSlot::primary_straggling);
    p("primary_mcs", TransportProfileSlot::primary_mcs);
    p("primary_nuclear", TransportProfileSlot::primary_nuclear);
    p("primary_boundary_nudge_continues",
      TransportProfileSlot::primary_boundary_nudge_continues);

    p("secondary_steps", TransportProfileSlot::secondary_steps);
    p("secondary_ct_samples", TransportProfileSlot::secondary_ct_samples);
    p("secondary_mass_sp_lookups", TransportProfileSlot::secondary_mass_sp_lookups);
    p("secondary_sp_table_lookups", TransportProfileSlot::secondary_sp_table_lookups);
    p("secondary_face_clamp_calls", TransportProfileSlot::secondary_face_clamp_calls);
    p("secondary_face_three_axis", TransportProfileSlot::secondary_face_three_axis);
    p("secondary_face_short_step_skip",
      TransportProfileSlot::secondary_face_short_step_skip);
    p("secondary_face_homogeneous_skip",
      TransportProfileSlot::secondary_face_homogeneous_skip);
    p("secondary_dose_atomics", TransportProfileSlot::secondary_dose_atomics);
    p("secondary_straggling", TransportProfileSlot::secondary_straggling);
    p("secondary_mcs", TransportProfileSlot::secondary_mcs);
    p("secondary_cascade", TransportProfileSlot::secondary_cascade);
    p("secondary_boundary_nudge_continues",
      TransportProfileSlot::secondary_boundary_nudge_continues);
    p("secondary_energy_nonprogress_steps",
      TransportProfileSlot::secondary_energy_nonprogress_steps);
    p("secondary_position_nonprogress_steps",
      TransportProfileSlot::secondary_position_nonprogress_steps);
    p("secondary_forced_progress_nudges",
      TransportProfileSlot::secondary_forced_progress_nudges);
    p("secondary_step_cap_hits", TransportProfileSlot::secondary_step_cap_hits);

    out << "  secondary_track_step_hist (bucket b = [2^b, 2^(b+1)) steps):\n";
    const auto base =
        static_cast<std::size_t>(TransportProfileSlot::secondary_track_hist_base);
    for (std::size_t b = 0; b < secondary_track_hist_buckets(); ++b) {
        const auto lo = b == 0 ? 1ULL : (1ULL << b);
        const auto hi = 1ULL << (b + 1);
        out << "    [" << lo << "," << hi << "): " << counters[base + b] << '\n';
    }
    return out.str();
}

}  // namespace carbon
