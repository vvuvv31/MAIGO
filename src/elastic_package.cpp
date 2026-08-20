#include "carbon/elastic_package.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace carbon {
namespace {

constexpr std::array<char, 8> expected_magic{'E', 'L', 'P', 'K', 'G', '0', '1', '\0'};
constexpr std::uint32_t current_version = 1;
constexpr std::int32_t continuation_code = 1;
constexpr float direction_tolerance = 2.0e-3F;
constexpr float energy_absolute_tolerance_MeV = 1.0e-2F;
constexpr float energy_relative_tolerance = 2.0e-3F;

#pragma pack(push, 1)
struct BinaryHeader {
    char magic[8];
    std::uint32_t version;
    std::uint32_t header_size;
    std::uint32_t energy_bin_record_size;
    std::uint32_t event_record_size;
    std::uint32_t product_record_size;
    std::uint32_t energy_bin_count;
    std::uint32_t flags;
    std::uint64_t event_count;
    std::uint64_t product_count;
    std::uint64_t expected_file_size;
};

struct BinaryEnergyBin {
    float minimum_energy_MeV_per_u;
    float maximum_energy_MeV_per_u;
    std::uint32_t event_offset;
    std::uint32_t event_count;
};

struct BinaryEvent {
    std::int16_t projectile_atomic_number;
    std::int16_t projectile_mass_number;
    float incident_energy_MeV_per_u;
    float outgoing_projectile_energy_MeV_per_u;
    float outgoing_direction_x;
    float outgoing_direction_y;
    float outgoing_direction_z;
    float local_deposit_MeV;
    std::uint32_t product_offset;
    std::uint32_t product_count;
    std::int32_t continuation;
    std::int32_t generation;
};

struct BinaryProduct {
    std::int32_t pdg_id;
    std::int16_t atomic_number;
    std::int16_t mass_number;
    float kinetic_energy_MeV;
    float direction_x;
    float direction_y;
    float direction_z;
    float charge_e;
    std::int32_t generation;
    std::int32_t transport_disposition;
};
#pragma pack(pop)

static_assert(sizeof(BinaryHeader) == 60);
static_assert(sizeof(BinaryEnergyBin) == 16);
static_assert(sizeof(BinaryEvent) == 44);
static_assert(sizeof(BinaryProduct) == 36);
static_assert(sizeof(ElasticEnergyBin) == sizeof(BinaryEnergyBin));
static_assert(sizeof(ElasticEvent) == sizeof(BinaryEvent));
static_assert(sizeof(ElasticProduct) == sizeof(BinaryProduct));
static_assert(std::is_trivially_copyable_v<BinaryHeader>);
static_assert(std::is_trivially_copyable_v<BinaryEnergyBin>);
static_assert(std::is_trivially_copyable_v<BinaryEvent>);
static_assert(std::is_trivially_copyable_v<BinaryProduct>);

[[noreturn]] void invalid(const std::filesystem::path& path, const char* message) {
    throw std::runtime_error(std::string("Invalid elastic package ") + message + ": " +
                             path.string());
}

bool finite_nonnegative(float value) {
    return std::isfinite(value) && value >= 0.0F;
}

bool unit_direction(float x, float y, float z) {
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) return false;
    const auto norm_squared = x * x + y * y + z * z;
    return std::isfinite(norm_squared) && std::abs(norm_squared - 1.0F) <= direction_tolerance;
}

bool known_neutral_pdg(std::int32_t pdg) {
    switch (pdg < 0 ? -pdg : pdg) {
    case 12:   // electron neutrino
    case 14:   // muon neutrino
    case 16:   // tau neutrino
    case 18:   // tau-prime neutrino
    case 22:   // photon
    case 111:  // pi0
    case 130:  // K0L
    case 2112: // neutron
    case 311:  // K0
    case 310:  // K0S
    case 3122: // Lambda
        return true;
    default:
        return false;
    }
}

template <typename Record>
void read_records(std::ifstream& input, std::vector<Record>& records, std::uint64_t count,
                  std::uint64_t file_size, std::uint64_t& cursor,
                  const std::filesystem::path& path, const char* label) {
    if (count > static_cast<std::uint64_t>(records.max_size()) ||
        count > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max()) /
                    sizeof(Record)) {
        invalid(path, "record count is too large");
    }
    const auto bytes = count * sizeof(Record);
    if (cursor > file_size || bytes > file_size - cursor) invalid(path, "record table is truncated");
    records.resize(static_cast<std::size_t>(count));
    if (bytes > 0 && !input.read(reinterpret_cast<char*>(records.data()),
                                 static_cast<std::streamsize>(bytes))) {
        invalid(path, label);
    }
    cursor += bytes;
}

std::optional<std::size_t> natural_bin_index(float energy, float minimum, float width,
                                             std::size_t count) {
    if (!std::isfinite(energy) || energy < minimum) return std::nullopt;
    const auto scaled = static_cast<double>(energy - minimum) / width;
    if (scaled < 0.0 || scaled >= static_cast<double>(count)) return std::nullopt;
    return static_cast<std::size_t>(scaled);
}

}  // namespace

ElasticPackageTable ElasticPackageTable::from_binary(const std::filesystem::path& path) {
    if constexpr (std::endian::native != std::endian::little) {
        throw std::runtime_error("Elastic packages require a little-endian host");
    }
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("Cannot open elastic package: " + path.string());
    const auto position = input.tellg();
    if (position < 0) throw std::runtime_error("Cannot determine elastic package size: " + path.string());
    const auto file_size = static_cast<std::uint64_t>(position);
    input.seekg(0);

    BinaryHeader binary_header{};
    if (!input.read(reinterpret_cast<char*>(&binary_header), sizeof(binary_header))) {
        invalid(path, "header is truncated");
    }
    if (!std::equal(expected_magic.begin(), expected_magic.end(), binary_header.magic)) {
        invalid(path, "magic");
    }
    if (binary_header.version != current_version || binary_header.header_size != sizeof(BinaryHeader) ||
        binary_header.energy_bin_record_size != sizeof(BinaryEnergyBin) ||
        binary_header.event_record_size != sizeof(BinaryEvent) ||
        binary_header.product_record_size != sizeof(BinaryProduct)) {
        invalid(path, "version or record layout");
    }
    if (binary_header.flags != 0 || binary_header.energy_bin_count == 0 ||
        binary_header.event_count == 0 || binary_header.expected_file_size < sizeof(BinaryHeader)) {
        invalid(path, "header values");
    }
    const auto add_size = [](std::uint64_t left, std::uint64_t right) {
        if (right > std::numeric_limits<std::uint64_t>::max() - left) {
            throw std::overflow_error("ELPKG file size overflow");
        }
        return left + right;
    };
    const auto multiply_size = [](std::uint64_t left, std::uint64_t right) {
        if (right != 0 && left > std::numeric_limits<std::uint64_t>::max() / right) {
            throw std::overflow_error("ELPKG file size overflow");
        }
        return left * right;
    };
    auto expected_size = static_cast<std::uint64_t>(sizeof(BinaryHeader));
    expected_size = add_size(expected_size,
                             multiply_size(binary_header.energy_bin_count,
                                           sizeof(BinaryEnergyBin)));
    expected_size = add_size(expected_size,
                             multiply_size(binary_header.event_count, sizeof(BinaryEvent)));
    expected_size = add_size(expected_size,
                             multiply_size(binary_header.product_count, sizeof(BinaryProduct)));
    if (binary_header.expected_file_size != expected_size || file_size != expected_size) {
        invalid(path, "file size");
    }

    ElasticPackageTable table;
    table.header_ = {binary_header.version,
                     binary_header.header_size,
                     binary_header.energy_bin_record_size,
                     binary_header.event_record_size,
                     binary_header.product_record_size,
                     binary_header.energy_bin_count,
                     binary_header.flags,
                     binary_header.event_count,
                     binary_header.product_count,
                     binary_header.expected_file_size};
    std::vector<BinaryEnergyBin> binary_bins;
    std::vector<BinaryEvent> binary_events;
    std::vector<BinaryProduct> binary_products;
    std::uint64_t cursor = sizeof(BinaryHeader);
    read_records(input, binary_bins, binary_header.energy_bin_count, file_size, cursor, path,
                 "energy-bin table");
    read_records(input, binary_events, binary_header.event_count, file_size, cursor, path,
                 "event table");
    read_records(input, binary_products, binary_header.product_count, file_size, cursor, path,
                 "product table");
    if (cursor != file_size) invalid(path, "trailing bytes");

    table.minimum_energy_MeV_per_u_ = binary_bins.front().minimum_energy_MeV_per_u;
    table.energy_bin_width_MeV_per_u_ = binary_bins.front().maximum_energy_MeV_per_u -
                                        binary_bins.front().minimum_energy_MeV_per_u;
    if (!std::isfinite(table.minimum_energy_MeV_per_u_) ||
        !std::isfinite(table.energy_bin_width_MeV_per_u_) ||
        table.energy_bin_width_MeV_per_u_ <= 0.0F) {
        invalid(path, "energy grid");
    }
    table.energy_bins_.reserve(binary_bins.size());
    std::uint64_t expected_event_offset = 0;
    for (std::size_t index = 0; index < binary_bins.size(); ++index) {
        const auto& bin = binary_bins[index];
        const auto expected_min = table.minimum_energy_MeV_per_u_ +
                                  static_cast<float>(index) * table.energy_bin_width_MeV_per_u_;
        const auto expected_max = expected_min + table.energy_bin_width_MeV_per_u_;
        if (!std::isfinite(bin.minimum_energy_MeV_per_u) ||
            !std::isfinite(bin.maximum_energy_MeV_per_u) ||
            bin.maximum_energy_MeV_per_u <= bin.minimum_energy_MeV_per_u ||
            bin.event_count == 0 || bin.event_offset != expected_event_offset ||
            static_cast<std::uint64_t>(bin.event_offset) + bin.event_count > binary_events.size() ||
            std::abs(bin.minimum_energy_MeV_per_u - expected_min) > 2.0e-4F ||
            std::abs(bin.maximum_energy_MeV_per_u - expected_max) > 2.0e-4F) {
            invalid(path, "energy-bin range");
        }
        expected_event_offset += bin.event_count;
        table.energy_bins_.push_back({bin.minimum_energy_MeV_per_u, bin.maximum_energy_MeV_per_u,
                                      bin.event_offset, bin.event_count});
    }
    if (expected_event_offset != binary_events.size()) invalid(path, "energy-bin coverage");

    table.events_.reserve(binary_events.size());
    table.products_.reserve(binary_products.size());
    std::uint64_t expected_product_offset = 0;
    for (const auto& binary_event : binary_events) {
        if (binary_event.projectile_atomic_number <= 0 ||
            binary_event.projectile_mass_number < binary_event.projectile_atomic_number ||
            !finite_nonnegative(binary_event.incident_energy_MeV_per_u) ||
            !finite_nonnegative(binary_event.outgoing_projectile_energy_MeV_per_u) ||
            !unit_direction(binary_event.outgoing_direction_x, binary_event.outgoing_direction_y,
                            binary_event.outgoing_direction_z) ||
            !finite_nonnegative(binary_event.local_deposit_MeV) ||
            binary_event.continuation != continuation_code || binary_event.generation < 0 ||
            binary_event.product_offset != expected_product_offset ||
            static_cast<std::uint64_t>(binary_event.product_offset) + binary_event.product_count >
                binary_products.size()) {
            invalid(path, "event value or product range");
        }
        if (table.projectile_atomic_number_ == 0) {
            table.projectile_atomic_number_ = binary_event.projectile_atomic_number;
            table.projectile_mass_number_ = binary_event.projectile_mass_number;
        } else if (table.projectile_atomic_number_ != binary_event.projectile_atomic_number ||
                   table.projectile_mass_number_ != binary_event.projectile_mass_number) {
            invalid(path, "inconsistent projectile identity");
        }
        const auto total_in = static_cast<double>(binary_event.incident_energy_MeV_per_u) *
                              binary_event.projectile_mass_number;
        double total_out = static_cast<double>(binary_event.outgoing_projectile_energy_MeV_per_u) *
                           binary_event.projectile_mass_number + binary_event.local_deposit_MeV;
        for (std::uint32_t product_index = 0; product_index < binary_event.product_count;
             ++product_index) {
            const auto& product = binary_products[binary_event.product_offset + product_index];
            if (product.atomic_number < 0 || product.mass_number < 0 ||
                (product.atomic_number == 0 && product.mass_number != 0) ||
                (product.mass_number > 0 && product.atomic_number > product.mass_number) ||
                !finite_nonnegative(product.kinetic_energy_MeV) ||
                !unit_direction(product.direction_x, product.direction_y, product.direction_z) ||
                !std::isfinite(product.charge_e) || product.generation < 0 ||
                product.transport_disposition < 1 || product.transport_disposition > 6 ||
                ((product.atomic_number == 0 && product.mass_number == 0) &&
                 std::abs(product.charge_e) > 1.0e-3F) ||
                (known_neutral_pdg(product.pdg_id) && std::abs(product.charge_e) > 1.0e-3F)) {
                invalid(path, "product value");
            }
            total_out += product.kinetic_energy_MeV;
        }
        const auto tolerance = std::max<double>(energy_absolute_tolerance_MeV,
                                                energy_relative_tolerance *
                                                    std::max(total_in, 1.0));
        if (std::abs(total_in - total_out) > tolerance) invalid(path, "event energy balance");
        expected_product_offset += binary_event.product_count;
        table.events_.push_back({binary_event.projectile_atomic_number,
                                 binary_event.projectile_mass_number,
                                 binary_event.incident_energy_MeV_per_u,
                                 binary_event.outgoing_projectile_energy_MeV_per_u,
                                 binary_event.outgoing_direction_x,
                                 binary_event.outgoing_direction_y,
                                 binary_event.outgoing_direction_z,
                                 binary_event.local_deposit_MeV,
                                 binary_event.product_offset,
                                 binary_event.product_count,
                                 binary_event.continuation,
                                 binary_event.generation});
    }
    if (expected_product_offset != binary_products.size()) invalid(path, "product coverage");
    for (const auto& binary_product : binary_products) {
        table.products_.push_back({binary_product.pdg_id,
                                   binary_product.atomic_number,
                                   binary_product.mass_number,
                                   binary_product.kinetic_energy_MeV,
                                   binary_product.direction_x,
                                   binary_product.direction_y,
                                   binary_product.direction_z,
                                   binary_product.charge_e,
                                   binary_product.generation,
                                   binary_product.transport_disposition});
    }

    // A bin may contain events from another natural energy bin only when it
    // is a complete byte-identical event+product clone (the compiler's
    // explicit nearest-fill representation).  Arbitrary mis-binning is rejected.
    for (std::size_t bin_index = 0; bin_index < table.energy_bins_.size(); ++bin_index) {
        const auto& bin = table.energy_bins_[bin_index];
        std::optional<std::size_t> alias_source;
        bool has_natural = false;
        for (std::uint32_t offset = 0; offset < bin.event_count; ++offset) {
            const auto& event = table.events_[bin.event_offset + offset];
            const auto natural = natural_bin_index(event.incident_energy_MeV_per_u,
                                                   table.minimum_energy_MeV_per_u_,
                                                   table.energy_bin_width_MeV_per_u_,
                                                   table.energy_bins_.size());
            if (!natural) invalid(path, "event energy outside grid");
            if (*natural == bin_index) {
                has_natural = true;
            } else if (!alias_source) {
                alias_source = *natural;
            } else if (*alias_source != *natural) {
                invalid(path, "energy bin mixes alias sources");
            }
        }
        if (!alias_source) continue;
        const auto& source = table.energy_bins_[*alias_source];
        if (has_natural || source.event_count != bin.event_count) {
            invalid(path, "incomplete nearest-fill alias");
        }
        for (std::uint32_t offset = 0; offset < bin.event_count; ++offset) {
            const auto& lhs = table.events_[bin.event_offset + offset];
            const auto& rhs = table.events_[source.event_offset + offset];
            auto lhs_payload = lhs;
            auto rhs_payload = rhs;
            // Product offsets necessarily differ after an aliased bin is
            // flattened; all other event fields must be byte-identical.
            lhs_payload.product_offset = 0;
            rhs_payload.product_offset = 0;
            if (std::memcmp(&lhs_payload, &rhs_payload, sizeof(ElasticEvent)) != 0 ||
                lhs.product_count != rhs.product_count ||
                (lhs.product_count > 0 &&
                 std::memcmp(table.products_.data() + lhs.product_offset,
                             table.products_.data() + rhs.product_offset,
                             static_cast<std::size_t>(lhs.product_count) * sizeof(ElasticProduct)) !=
                     0)) {
                invalid(path, "nearest-fill alias payload");
            }
        }
    }
    return table;
}

const ElasticPackageHeader& ElasticPackageTable::header() const noexcept { return header_; }
const std::vector<ElasticEnergyBin>& ElasticPackageTable::energy_bins() const noexcept {
    return energy_bins_;
}
const std::vector<ElasticEvent>& ElasticPackageTable::events() const noexcept { return events_; }
const std::vector<ElasticProduct>& ElasticPackageTable::products() const noexcept { return products_; }
int ElasticPackageTable::projectile_atomic_number() const noexcept { return projectile_atomic_number_; }
int ElasticPackageTable::projectile_mass_number() const noexcept { return projectile_mass_number_; }
float ElasticPackageTable::minimum_energy_MeV_per_u() const noexcept { return minimum_energy_MeV_per_u_; }
float ElasticPackageTable::energy_bin_width_MeV_per_u() const noexcept { return energy_bin_width_MeV_per_u_; }

std::size_t ElasticPackageTable::energy_bin_index(float energy_MeV_per_u) const noexcept {
    if (energy_bins_.empty() || !std::isfinite(energy_MeV_per_u) ||
        energy_MeV_per_u <= minimum_energy_MeV_per_u_) return 0;
    const auto scaled = (energy_MeV_per_u - minimum_energy_MeV_per_u_) /
                        energy_bin_width_MeV_per_u_;
    if (!std::isfinite(scaled) || scaled <= 0.0F) return 0;
    const auto index = static_cast<std::size_t>(scaled);
    return std::min(index, energy_bins_.size() - 1);
}

const ElasticEnergyBin& ElasticPackageTable::energy_bin(float energy_MeV_per_u) const noexcept {
    return energy_bins_[energy_bin_index(energy_MeV_per_u)];
}

std::span<const ElasticEvent> ElasticPackageTable::event_span(std::size_t bin_index) const noexcept {
    if (bin_index >= energy_bins_.size()) return {};
    const auto& bin = energy_bins_[bin_index];
    return std::span<const ElasticEvent>(events_).subspan(bin.event_offset, bin.event_count);
}

std::span<const ElasticEvent> ElasticPackageTable::event_span_for_energy(
    float energy_MeV_per_u) const noexcept {
    return event_span(energy_bin_index(energy_MeV_per_u));
}

const ElasticEvent* ElasticPackageTable::event(std::size_t event_index) const noexcept {
    return event_index < events_.size() ? &events_[event_index] : nullptr;
}

std::span<const ElasticProduct> ElasticPackageTable::product_span(
    const ElasticEvent& event_record) const noexcept {
    if (events_.empty()) return {};
    const auto* begin = events_.data();
    const auto* end = begin + events_.size();
    if (&event_record < begin || &event_record >= end) return {};
    return std::span<const ElasticProduct>(products_).subspan(event_record.product_offset,
                                                               event_record.product_count);
}

std::span<const ElasticProduct> ElasticPackageTable::product_span(
    std::size_t event_index) const noexcept {
    const auto* selected = event(event_index);
    return selected == nullptr ? std::span<const ElasticProduct>{} : product_span(*selected);
}

}  // namespace carbon
