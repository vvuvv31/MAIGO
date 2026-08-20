#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace carbon {

// These records mirror ELPKG v1's explicitly little-endian wire layout.  They
// are value types returned through const views; the loader owns all storage.
struct ElasticEnergyBin {
    float minimum_energy_MeV_per_u{0.0F};
    float maximum_energy_MeV_per_u{0.0F};
    std::uint32_t event_offset{0};
    std::uint32_t event_count{0};
};

struct ElasticEvent {
    std::int16_t projectile_atomic_number{0};
    std::int16_t projectile_mass_number{0};
    float incident_energy_MeV_per_u{0.0F};
    float outgoing_projectile_energy_MeV_per_u{0.0F};
    float outgoing_direction_x{0.0F};
    float outgoing_direction_y{0.0F};
    float outgoing_direction_z{1.0F};
    float local_deposit_MeV{0.0F};
    std::uint32_t product_offset{0};
    std::uint32_t product_count{0};
    std::int32_t continuation{0};
    std::int32_t generation{0};
};

struct ElasticProduct {
    std::int32_t pdg_id{0};
    std::int16_t atomic_number{0};
    std::int16_t mass_number{0};
    float kinetic_energy_MeV{0.0F};
    float direction_x{0.0F};
    float direction_y{0.0F};
    float direction_z{1.0F};
    float charge_e{0.0F};
    std::int32_t generation{0};
    std::int32_t transport_disposition{0};
};

struct ElasticPackageHeader {
    std::uint32_t version{0};
    std::uint32_t header_size{0};
    std::uint32_t energy_bin_record_size{0};
    std::uint32_t event_record_size{0};
    std::uint32_t product_record_size{0};
    std::uint32_t energy_bin_count{0};
    std::uint32_t flags{0};
    std::uint64_t event_count{0};
    std::uint64_t product_count{0};
    std::uint64_t expected_file_size{0};
};

class ElasticPackageTable {
public:
    static ElasticPackageTable from_binary(const std::filesystem::path& path);

    [[nodiscard]] const ElasticPackageHeader& header() const noexcept;
    [[nodiscard]] const std::vector<ElasticEnergyBin>& energy_bins() const noexcept;
    [[nodiscard]] const std::vector<ElasticEvent>& events() const noexcept;
    [[nodiscard]] const std::vector<ElasticProduct>& products() const noexcept;

    // Projectile identity is not duplicated in the ELPKG v1 header.  The
    // loader verifies every event and exposes the common identity here.
    [[nodiscard]] int projectile_atomic_number() const noexcept;
    [[nodiscard]] int projectile_mass_number() const noexcept;
    [[nodiscard]] float minimum_energy_MeV_per_u() const noexcept;
    [[nodiscard]] float energy_bin_width_MeV_per_u() const noexcept;

    // Out-of-range and non-finite energies are clamped to the nearest valid
    // bin, matching the existing package table APIs.
    [[nodiscard]] std::size_t energy_bin_index(float energy_MeV_per_u) const noexcept;
    [[nodiscard]] const ElasticEnergyBin& energy_bin(float energy_MeV_per_u) const noexcept;
    [[nodiscard]] std::span<const ElasticEvent> event_span(
        std::size_t bin_index) const noexcept;
    [[nodiscard]] std::span<const ElasticEvent> event_span_for_energy(
        float energy_MeV_per_u) const noexcept;

    // Invalid indices return an empty span.  event() is the checked indexed
    // accessor for callers that require a diagnostic on programmer errors.
    [[nodiscard]] const ElasticEvent* event(std::size_t event_index) const noexcept;
    [[nodiscard]] std::span<const ElasticProduct> product_span(
        const ElasticEvent& event) const noexcept;
    [[nodiscard]] std::span<const ElasticProduct> product_span(
        std::size_t event_index) const noexcept;

private:
    ElasticPackageHeader header_{};
    int projectile_atomic_number_{0};
    int projectile_mass_number_{0};
    float minimum_energy_MeV_per_u_{0.0F};
    float energy_bin_width_MeV_per_u_{1.0F};
    std::vector<ElasticEnergyBin> energy_bins_;
    std::vector<ElasticEvent> events_;
    std::vector<ElasticProduct> products_;
};

}  // namespace carbon
