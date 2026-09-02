#pragma once

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <vector>

namespace carbon {

class CrossSectionTable {
public:
    CrossSectionTable() = default;
    CrossSectionTable(std::vector<double> energies_MeVu,
                      std::vector<double> macroscopic_cross_sections_per_mm);

    static CrossSectionTable from_csv(const std::filesystem::path& path);
    // Paper C-C fit + Kox(C,O)/Kox(C,C) + ICRU-H macroscopic water table.
    static CrossSectionTable from_fred_paper_water(double density_g_per_cm3 = 1.0);
    static std::vector<CrossSectionTable> from_schneider_csv(
        const std::filesystem::path& path);

    [[nodiscard]] double interpolate(double energy_MeVu) const noexcept;
    [[nodiscard]] double interpolate_target_h_fraction(double energy_MeVu) const noexcept;
    [[nodiscard]] const std::vector<double>& energies() const noexcept;
    [[nodiscard]] const std::vector<double>& values() const noexcept;
    [[nodiscard]] const std::vector<double>& target_h_fractions() const noexcept;
    [[nodiscard]] const std::vector<double>& macro_h_per_mm() const noexcept;
    [[nodiscard]] const std::vector<double>& macro_o_per_mm() const noexcept;
    void set_target_h_fractions(std::vector<double> fractions);
    void set_partial_macros(std::vector<double> macro_h, std::vector<double> macro_o);

private:
    std::vector<double> energies_MeVu_;
    std::vector<double> macroscopic_cross_sections_per_mm_;
    std::vector<double> target_h_fractions_;
    std::vector<double> macro_h_per_mm_;
    std::vector<double> macro_o_per_mm_;
};

struct ResampledCrossSectionGrid {
    std::vector<float> macroscopic_per_mm;
    std::vector<float> target_h_fraction;
};

// Resample inelastic data onto the transport grid used by device kernels.
[[nodiscard]] ResampledCrossSectionGrid resample_cross_section_grid(
    const CrossSectionTable& cross_section,
    const std::vector<double>& transport_energies_MeVu);

struct SchneiderResampledCrossSectionGrid {
    static constexpr std::size_t kExpectedSections = 25;
    std::vector<double> transport_energies_MeVu;
    // Flattened table: [section_id * energy_nodes + node_idx]
    // Unit: mm^-1 at 1 g/cm3 (mass cross section; density is NOT multiplied here)
    std::vector<float> mass_xs_per_mm_at_1g_cm3;

    [[nodiscard]] std::size_t energy_nodes() const noexcept {
        return transport_energies_MeVu.size();
    }

    [[nodiscard]] float at(std::size_t section_id, std::size_t energy_node) const {
        if (section_id >= kExpectedSections) {
            throw std::out_of_range(
                "Schneider section ID " + std::to_string(section_id) +
                " out of range (expected 0.." + std::to_string(kExpectedSections - 1) + ")");
        }
        if (energy_node >= energy_nodes()) {
            throw std::out_of_range(
                "Energy node index " + std::to_string(energy_node) +
                " out of range (max " + std::to_string(energy_nodes()) + ")");
        }
        return mass_xs_per_mm_at_1g_cm3[section_id * energy_nodes() + energy_node];
    }
};

// Resample 25-section Schneider cross-section tables onto the transport energy grid.
// Validates identical nodes across sections, strict energy coverage (fail-fast without clamping),
// and stores mass rates in units of mm^-1 at 1 g/cm3 (without density multiplication).
[[nodiscard]] SchneiderResampledCrossSectionGrid resample_schneider_cross_section_grid(
    const std::vector<CrossSectionTable>& tables,
    const std::vector<double>& transport_energies_MeVu);

struct TransportConfig;

// Prepares and resamples the 25-section Schneider cross-section table for primary transport on the host.
// Reads config.ct_schneider_cross_section_file, validates exactly 25 sections, and resamples onto transport_energies_MeVu.
[[nodiscard]] SchneiderResampledCrossSectionGrid prepare_schneider_primary_xs(
    const TransportConfig& config,
    const std::vector<double>& transport_energies_MeVu);

// Single canonical layout indexing helper shared by host and GPU device code.
// Contiguous layout: [section][energy]
// index = section_id * energy_nodes + energy_index
inline constexpr std::size_t schneider_cross_section_index(
    std::size_t section_id,
    std::size_t energy_index,
    std::size_t energy_nodes) noexcept {
    return section_id * energy_nodes + energy_index;
}

// Evaluates mass cross section rate (mm^-1 / (g/cm^3)) via linear interpolation on host or device.
inline float schneider_primary_mass_xs(
    const float* __restrict table,
    std::uint32_t section_count,
    std::uint32_t energy_nodes,
    float e_min,
    float inv_dE,
    std::uint32_t section_id,
    float energy_mevu) noexcept {
    if (table == nullptr || section_id >= section_count || energy_nodes < 2) {
        return 0.0F;
    }
    const float f_node = (energy_mevu - e_min) * inv_dE;
    const auto max_node = static_cast<std::uint32_t>(energy_nodes - 1);
    if (f_node <= 0.0F) {
        return table[schneider_cross_section_index(section_id, 0, energy_nodes)];
    }
    if (f_node >= static_cast<float>(max_node)) {
        return table[schneider_cross_section_index(section_id, max_node, energy_nodes)];
    }
    const auto idx0 = static_cast<std::uint32_t>(f_node);
    const auto idx1 = idx0 + 1U;
    const float frac = f_node - static_cast<float>(idx0);
    const float v0 = table[schneider_cross_section_index(section_id, idx0, energy_nodes)];
    const float v1 = table[schneider_cross_section_index(section_id, idx1, energy_nodes)];
    return v0 + frac * (v1 - v0);
}

// Evaluates macroscopic cross section Sigma (mm^-1) = density_g_cm3 * mass_rate on host or device.
// Density multiplication occurs exactly once at runtime; no reference-density division is performed.
inline float schneider_primary_macroscopic_xs(
    const float* __restrict table,
    std::uint32_t section_count,
    std::uint32_t energy_nodes,
    float e_min,
    float inv_dE,
    std::uint32_t section_id,
    float energy_mevu,
    float density_g_cm3) noexcept {
    return density_g_cm3 * schneider_primary_mass_xs(
        table, section_count, energy_nodes, e_min, inv_dE, section_id, energy_mevu);
}

// Consumes optical depth over a segment of length step_mm in a medium with macro_xs (mm^-1).
// If collision occurs within step_mm: returns true, clamps step_mm to collision distance, resets tau_active.
// Otherwise: returns false, subtracts macro_xs * step_mm from tau_remaining, keeps tau_active true.
inline bool consume_schneider_optical_depth_segment(
    float& tau_remaining,
    bool& tau_active,
    float& step_mm,
    float macro_xs,
    float rng_u) noexcept {
    if (!tau_active) {
        tau_remaining = -std::log(std::max(rng_u, 1.0e-12F));
        tau_active = true;
    }
    const float delta_tau = macro_xs * step_mm;
    if (macro_xs > 0.0F && delta_tau >= tau_remaining) {
        const float collision_s = tau_remaining / macro_xs;
        step_mm = std::min(step_mm, collision_s);
        tau_remaining = 0.0F;
        tau_active = false;
        return true;
    }
    tau_remaining -= delta_tau;
    return false;
}

#ifdef CARBON_HAS_SYCL
// GPU batch evaluation helper for testing Schneider primary cross-section table lookup & density scaling.
[[nodiscard]] std::vector<float> test_schneider_device_lookup_batch(
    const std::vector<float>& host_table,
    std::uint32_t section_count,
    std::uint32_t energy_nodes,
    float e_min,
    float inv_dE,
    const std::vector<std::uint32_t>& query_sections,
    const std::vector<float>& query_energies,
    const std::vector<float>& query_densities,
    const std::string& device_preference = "default");

struct CtGrid;

struct Step11TestResult {
    double survival_fraction{0.0};
    double survival_fraction_std_err{0.0};
    std::vector<std::uint64_t> interaction_binned_counts{};
    std::uint64_t zero_progress_count{0};
    std::uint64_t changed_material_step_span_count{0};
    std::uint64_t face_cross_count{0};
};

// Standalone GPU test runner for Step 11 piecewise nuclear optical-depth stepping.
[[nodiscard]] Step11TestResult run_step11_piecewise_hazard_gpu_test(
    const CtGrid& grid,
    const std::vector<float>& host_schneider_table,
    std::uint32_t section_count,
    std::uint32_t energy_nodes,
    float e_min,
    float inv_dE,
    float energy_mevu,
    std::uint32_t num_histories,
    float ray_origin_x,
    float ray_origin_y,
    float ray_origin_z,
    float ray_dir_x,
    float ray_dir_y,
    float ray_dir_z,
    float max_track_length_mm,
    float bin_width_mm,
    std::uint32_t num_bins,
    const std::string& device_preference = "default");
#endif


class IonCrossSectionTables {
public:
    static constexpr std::size_t mass_stride = 32;
    static constexpr std::size_t atomic_number_slots = 10;
    static constexpr std::size_t species_slots =
        mass_stride * atomic_number_slots;

    static IonCrossSectionTables from_csv(
        const std::filesystem::path& path);

    [[nodiscard]] const std::vector<float>& values() const noexcept;
    [[nodiscard]] const std::vector<std::uint8_t>& species_present()
        const noexcept;
    [[nodiscard]] std::size_t energy_grid_size() const noexcept;
    [[nodiscard]] float minimum_energy_MeVu() const noexcept;
    [[nodiscard]] float energy_step_MeVu() const noexcept;

private:
    std::size_t energy_grid_size_{};
    float minimum_energy_MeVu_{};
    float energy_step_MeVu_{};
    std::vector<float> values_;
    std::vector<std::uint8_t> species_present_;
};

class NeutralCrossSectionTables {
public:
    static constexpr std::size_t species_count = 2;

    static NeutralCrossSectionTables from_csv(
        const std::filesystem::path& path);

    [[nodiscard]] const std::vector<float>& values() const noexcept;
    [[nodiscard]] std::size_t energy_grid_size() const noexcept;
    [[nodiscard]] float minimum_log_energy() const noexcept;
    [[nodiscard]] float log_energy_step() const noexcept;

private:
    std::size_t energy_grid_size_{};
    float minimum_log_energy_{};
    float log_energy_step_{};
    // Gamma (PDG 22), then neutron (PDG 2112).
    std::vector<float> values_;
};

}  // namespace carbon
