#pragma once

namespace carbon::pipelines::primary_stage {

// Compile-time kernel identities, not YAML options or physical process IDs.
// Keep the integer values stable: PrimaryTransportKernel uses them as template
// arguments. A stage selects both its input state and its compile-time physics
// policy; changing only the launch site can therefore select the wrong policy.
inline constexpr int beamline = 3;
inline constexpr int medium = 4;
inline constexpr int full_transport = 6;
inline constexpr int full_transport_nonion = 7;
inline constexpr int beamline_queued_children = 9;

// CT broadbeam specializations. Shared-cell paths require the host to prove
// exact equality of the material and dose grids before reusing a cell lookup.
inline constexpr int proton_ct = 15;
inline constexpr int carbon_ct = 17;
inline constexpr int carbon_ct_shared_cell = 18;

// Begin consumes source/entry state; resume consumes a continuation checkpoint.
// Each pair must retain the same physics policy and particle/RNG identity.
inline constexpr int minibeam_ct_begin = 19;
inline constexpr int minibeam_ct_resume = 20;
inline constexpr int proton_ct_shared_cell_begin = 21;
inline constexpr int proton_ct_shared_cell_resume = 22;

// Stage predicates describe launch/checkpoint ownership. Keep these mappings
// here so the transport body does not reconstruct them from overlapping flags.
constexpr bool is_proton_ct(int stage) noexcept {
    return stage == proton_ct || stage == proton_ct_shared_cell_begin ||
           stage == proton_ct_shared_cell_resume;
}
constexpr bool is_carbon_ct(int stage) noexcept {
    return stage == carbon_ct || stage == carbon_ct_shared_cell;
}
constexpr bool uses_shared_ct_cell(int stage) noexcept {
    return stage == carbon_ct_shared_cell || stage == proton_ct_shared_cell_begin ||
           stage == proton_ct_shared_cell_resume;
}
constexpr bool is_continuation(int stage) noexcept {
    return stage == minibeam_ct_begin || stage == minibeam_ct_resume ||
           stage == proton_ct_shared_cell_begin || stage == proton_ct_shared_cell_resume;
}
constexpr bool is_resume(int stage) noexcept {
    return stage == minibeam_ct_resume || stage == proton_ct_shared_cell_resume;
}
constexpr int transport_body(int stage, bool broadbeam) noexcept {
    if (is_continuation(stage) && !broadbeam) return medium;
    if (stage == beamline_queued_children) return beamline;
    if (is_proton_ct(stage) || is_carbon_ct(stage)) return full_transport;
    return stage;
}

} // namespace carbon::pipelines::primary_stage
