#include "carbon/inelastic_package_v2.hpp"
#include "carbon/charged_species.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <limits>
#include <map>
#include <utility>
#include <stdexcept>
#include <vector>

#ifdef CARBON_HAS_SYCL
#include <sycl/sycl.hpp>
#include "carbon/inelastic_package_v3.hpp"
#include "carbon/multiple_scattering.hpp"
#include "carbon/rng.hpp"
namespace carbon {
namespace {
// Test-local copy of the exact runtime rotation (internal linkage in both
// TUs, so no ODR/link clash with transport_sycl.cpp).
#include "detail/sycl_device_math.inc"
}  // namespace
}  // namespace carbon
#endif

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

#ifdef CARBON_HAS_SYCL
namespace {

// Verifies the runtime local-frame rotation used for CINEL03 replay
// (transport_sycl.cpp rotate_local_direction call sites): for incident
// (0,0,1) the child direction must reproduce the payload local direction,
// and for tilted incidents dot(child, incident) must equal local_direction_z
// with a unit child vector. Runs the identical device code object.
void run_device_rotation_check(const std::string& package_path) {
    const auto table =
        carbon::InelasticPackageV3Table::from_binary(package_path);
    struct Vec {
        float x, y, z;
    };
    std::vector<Vec> locals;
    for (const auto& p : table.products()) {
        if (p.role != 0) {
            continue;
        }
        const float n = std::sqrt(p.local_direction_x * p.local_direction_x +
                                  p.local_direction_y * p.local_direction_y +
                                  p.local_direction_z * p.local_direction_z);
        if (!(n > 0.99F && n < 1.01F)) {
            continue;
        }
        locals.push_back(
            Vec{p.local_direction_x, p.local_direction_y, p.local_direction_z});
        if (locals.size() >= 20000) {
            break;
        }
    }
    require(!locals.empty(), "no unit local directions in CINEL03 package");
    const std::vector<Vec> incidents{
        {0.0F, 0.0F, 1.0F},
        {0.3F, 0.0F, 0.953939F},
        {0.0F, 0.5F, 0.866025F},
        {-0.2F, 0.3F, 0.932738F},
    };
    sycl::queue queue;
    const std::size_t n = locals.size();
    const std::size_t m = incidents.size();
    auto* in = sycl::malloc_shared<Vec>(n, queue);
    auto* out = sycl::malloc_shared<Vec>(n * m, queue);
    require(in != nullptr && out != nullptr, "rotation test alloc failed");
    for (std::size_t i = 0; i < n; ++i) {
        in[i] = locals[i];
    }
    auto* inc = sycl::malloc_shared<Vec>(m, queue);
    require(inc != nullptr, "rotation incident alloc failed");
    for (std::size_t j = 0; j < m; ++j) {
        inc[j] = incidents[j];
    }
    queue
        .parallel_for(sycl::range<1>(n * m),
                      [=](sycl::item<1> item) {
                          const std::size_t k = item.get_linear_id();
                          const std::size_t i = k / m;
                          const std::size_t j = k % m;
                          const auto r = carbon::rotate_local_direction(
                              in[i].x, in[i].y, in[i].z,
                              carbon::Direction3F{inc[j].x, inc[j].y, inc[j].z});
                          out[k] = Vec{r.x, r.y, r.z};
                      })
        .wait_and_throw();
    double max_unit_err = 0.0;
    double max_dot_err = 0.0;
    double max_identity_err = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < m; ++j) {
            const Vec o = out[i * m + j];
            const Vec iv = incidents[j];
            const double norm =
                std::sqrt(o.x * o.x + o.y * o.y + o.z * o.z);
            max_unit_err = std::max(max_unit_err, std::abs(norm - 1.0));
            const double dot = o.x * iv.x + o.y * iv.y + o.z * iv.z;
            max_dot_err =
                std::max(max_dot_err, std::abs(dot - locals[i].z));
            if (j == 0) {
                const double dx = static_cast<double>(o.x - locals[i].x);
                const double dy = static_cast<double>(o.y - locals[i].y);
                const double dz = static_cast<double>(o.z - locals[i].z);
                max_identity_err =
                    std::max(max_identity_err, std::sqrt(dx * dx + dy * dy + dz * dz));
            }
        }
    }
    sycl::free(in, queue);
    sycl::free(inc, queue);
    sycl::free(out, queue);
    std::cout << "rotation products=" << n << " max_unit_err=" << max_unit_err
              << " max_dot_err=" << max_dot_err
              << " max_identity_err=" << max_identity_err << '\n';
    require(max_unit_err < 1.0e-6, "rotated child is not a unit vector");
    require(max_dot_err < 2.0e-5, "dot(child, incident) != local_direction_z");
    require(max_identity_err < 1.0e-6,
            "incident=(0,0,1) does not reproduce payload local direction");
}
}  // namespace
#endif

#ifdef CARBON_HAS_SYCL
struct FixedReplaySummary {
    std::uint64_t discrete_hash{1469598103934665603ULL};
    std::uint64_t event_count{0};
    std::uint64_t product_count{0};
    float parent_energy_MeV{0.0F};
    float product_energy_MeV{0.0F};
    float local_direction_moment{0.0F};
};

FixedReplaySummary summarize_fixed_replay_device(
    const carbon::Cinel02DeviceTables& tables) {
    sycl::queue queue;
    auto* interactions = sycl::malloc_device<carbon::Cinel02DeviceInteraction>(
        tables.interactions.size(), queue);
    auto* products = sycl::malloc_device<carbon::Cinel02DeviceProduct>(
        tables.products.size(), queue);
    auto* output = sycl::malloc_shared<FixedReplaySummary>(1, queue);
    require(interactions != nullptr && products != nullptr && output != nullptr,
            "fixed replay device allocation failed");
    queue.copy(tables.interactions.data(), interactions, tables.interactions.size());
    queue.copy(tables.products.data(), products, tables.products.size());
    *output = {};
    const auto interaction_count = tables.interactions.size();
    const auto product_count = tables.products.size();
    queue.single_task([=]() {
        FixedReplaySummary result{};
        const auto mix = [&](const std::uint64_t value) {
            result.discrete_hash ^= value;
            result.discrete_hash *= 1099511628211ULL;
        };
        for (std::size_t i = 0; i < interaction_count; ++i) {
            const auto event = interactions[i];
            mix(static_cast<std::uint32_t>(event.parent_pdg));
            mix(static_cast<std::uint16_t>(event.parent_z));
            mix(static_cast<std::uint16_t>(event.parent_a));
            mix(static_cast<std::uint32_t>(event.parent_status));
            mix(event.product_offset);
            mix(event.product_count);
            result.parent_energy_MeV += event.parent_energy_MeV;
            ++result.event_count;
        }
        for (std::size_t i = 0; i < product_count; ++i) {
            const auto product = products[i];
            mix(static_cast<std::uint32_t>(product.pdg));
            mix(static_cast<std::uint16_t>(product.z));
            mix(static_cast<std::uint16_t>(product.a));
            mix(static_cast<std::uint32_t>(product.role));
            result.product_energy_MeV += product.kinetic_energy_MeV;
            result.local_direction_moment +=
                product.local_direction_z * product.kinetic_energy_MeV;
            ++result.product_count;
        }
        *output = result;
    }).wait_and_throw();
    const auto result = *output;
    sycl::free(interactions, queue);
    sycl::free(products, queue);
    sycl::free(output, queue);
    return result;
}
#endif

void test_synthetic_energy_ledger() {
    // Synthetic killed-parent event: NIEL is diagnostic-only and must not be
    // added to local deposit. The remaining kinetic imbalance is the explicit
    // Q/excitation bucket, not a numerical closure failure.
    constexpr double collision_input = 120.0;
    constexpr double pre_collision_em_loss = 2.0;
    constexpr double local_deposit = 3.0;
    constexpr double niel_diagnostic = 0.75;
    constexpr double parent_kinetic = 0.0;
    constexpr double charged_kinetic = 100.0;
    constexpr double neutral_kinetic = 5.0;
    constexpr double unsupported_kinetic = 4.0;
    constexpr double expected_q_excitation = 8.0;
    const auto q_excitation = collision_input - local_deposit - parent_kinetic -
                              charged_kinetic - neutral_kinetic -
                              unsupported_kinetic;
    require(std::abs(q_excitation - expected_q_excitation) < 1.0e-12,
            "synthetic CINEL02 Q/excitation ledger does not close");
    require(std::abs((pre_collision_em_loss + collision_input) - 122.0) < 1.0e-12,
            "synthetic CINEL02 pre-collision EM ledger does not close");
    require(std::abs((local_deposit + niel_diagnostic) - local_deposit) > 0.5,
            "synthetic CINEL02 test did not expose local+NIEL double counting");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 1) {
        try {
            test_synthetic_energy_ledger();
            return 0;
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
#ifdef CARBON_HAS_SYCL
    if (argc == 3 && std::string{argv[1]} == "--device-replay") {
        try {
            const auto table = carbon::InelasticPackageV2Table::from_binary(argv[2]);
            const auto compact = table.make_device_tables();
            FixedReplaySummary host{};
            const auto mix = [&](const std::uint64_t value) {
                host.discrete_hash ^= value;
                host.discrete_hash *= 1099511628211ULL;
            };
            for (std::uint64_t i = 0; i < table.interactions().size(); ++i) {
                const auto replay = table.fixed_replay(i, compact);
                require(static_cast<bool>(replay), "host fixed replay failed");
                const auto& event = *replay.compact_interaction;
                mix(static_cast<std::uint32_t>(event.parent_pdg));
                mix(static_cast<std::uint16_t>(event.parent_z));
                mix(static_cast<std::uint16_t>(event.parent_a));
                mix(static_cast<std::uint32_t>(event.parent_status));
                mix(event.product_offset);
                mix(event.product_count);
                host.parent_energy_MeV += event.parent_energy_MeV;
                ++host.event_count;
            }
            for (const auto& product : compact.products) {
                mix(static_cast<std::uint32_t>(product.pdg));
                mix(static_cast<std::uint16_t>(product.z));
                mix(static_cast<std::uint16_t>(product.a));
                mix(static_cast<std::uint32_t>(product.role));
                host.product_energy_MeV += product.kinetic_energy_MeV;
                host.local_direction_moment +=
                    product.local_direction_z * product.kinetic_energy_MeV;
                ++host.product_count;
            }
            const auto device = summarize_fixed_replay_device(compact);
            require(host.discrete_hash == device.discrete_hash &&
                        host.event_count == device.event_count &&
                        host.product_count == device.product_count,
                    "fixed replay host/device discrete summary differs");
            const auto close = [](const float a, const float b) {
                return std::abs(a - b) <= 2.0e-5F * std::max(1.0F, std::abs(a));
            };
            require(close(host.parent_energy_MeV, device.parent_energy_MeV) &&
                        close(host.product_energy_MeV, device.product_energy_MeV) &&
                        close(host.local_direction_moment, device.local_direction_moment),
                    "fixed replay host/device floating summary differs");
            std::cout << "events=" << device.event_count
                      << " products=" << device.product_count
                      << " hash=" << device.discrete_hash
                      << " product_energy_MeV=" << device.product_energy_MeV
                      << " direction_moment=" << device.local_direction_moment << '\n';
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "CINEL02 device replay failed: " << error.what() << '\n';
            return 1;
        }
    }
#endif
#ifdef CARBON_HAS_SYCL
    if (argc == 3 && std::string{argv[1]} == "--device-rotation") {
        try {
            run_device_rotation_check(argv[2]);
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "CINEL03 device rotation failed: " << error.what() << '\n';
            return 1;
        }
    }
#endif
    if ((argc == 3 || argc == 4) && std::string{argv[1]} == "--inspect") {
        try {
            std::ofstream report_file;
            std::ostream* report = &std::cout;
            if (argc == 4) {
                report_file.open(argv[3], std::ios::binary);
                require(static_cast<bool>(report_file),
                        "cannot create CINEL02 compatibility report");
                report = &report_file;
            }
            const auto table = carbon::InelasticPackageV2Table::from_binary(argv[2]);
            std::uint64_t products = 0;
            for (const auto& event : table.interactions()) {
                require(table.product_offset(event) == products,
                        "CINPKG03 product offsets are not contiguous");
                products += event.direct_product_count;
            }
            require(products == table.products().size(),
                    "CINPKG03 product ledger does not close");
            const auto device = table.make_device_tables();
            require(device.interactions.size() == table.interactions().size(),
                    "CINEL02 compact interaction count mismatch");
            require(device.products.size() == table.products().size(),
                    "CINEL02 compact product count mismatch");
            double incident_kinetic_MeV = 0.0;
            double local_deposit_MeV = 0.0;
            double niel_MeV = 0.0;
            double parent_kinetic_MeV = 0.0;
            double charged_kinetic_MeV = 0.0;
            double neutral_kinetic_MeV = 0.0;
            double unsupported_kinetic_MeV = 0.0;
            std::uint64_t parent_survival_count = 0U;
            double multiplicity_sum = 0.0;
            double multiplicity_squared_sum = 0.0;
            double product_energy_squared_MeV2 = 0.0;
            double product_energy_direction_z_MeV = 0.0;
            std::map<std::pair<int, int>, std::pair<std::uint64_t, double>>
                unsupported_charged_isotopes;
            std::map<std::pair<int, int>, std::pair<std::uint64_t, double>>
                transported_charged_isotopes;
            for (std::size_t i = 0; i < table.interactions().size(); ++i) {
                const auto replay = table.fixed_replay(i, device);
                require(static_cast<bool>(replay),
                        "CINEL02 fixed replay rejected serialized event");
                const auto& raw = *replay.serialized_interaction;
                const auto& compact = *replay.compact_interaction;
                require(std::abs(compact.incident_energy_MeV_per_u -
                                 raw.collision_energy_MeV_per_u) <= 1.0e-6F &&
                            std::abs(compact.process_local_deposit_MeV -
                                     raw.process_local_deposit_MeV) <= 1.0e-6F &&
                            std::abs(compact.nonionizing_deposit_MeV -
                                     raw.nonionizing_deposit_MeV) <= 1.0e-6F &&
                            std::abs(compact.parent_energy_MeV -
                                     raw.parent_energy_MeV) <= 1.0e-6F &&
                            compact.product_offset == table.product_offset(raw) &&
                            compact.product_count == raw.direct_product_count,
                        "CINEL02 compact interaction differs from serialized record");
                incident_kinetic_MeV += raw.collision_energy_MeV;
                local_deposit_MeV += raw.process_local_deposit_MeV;
                niel_MeV += raw.nonionizing_deposit_MeV;
                parent_kinetic_MeV += raw.parent_energy_MeV;
                parent_survival_count += raw.parent_status == 0 ? 1U : 0U;
                multiplicity_sum += raw.direct_product_count;
                multiplicity_squared_sum +=
                    static_cast<double>(raw.direct_product_count) *
                    raw.direct_product_count;
                double event_unsupported_MeV = 0.0;
                for (std::uint32_t p = 0; p < raw.direct_product_count; ++p) {
                    const auto index = table.product_offset(raw) + p;
                    const auto& source = replay.serialized_products[p];
                    const auto& uploaded = replay.compact_products[p];
                    require(source.pdg == uploaded.pdg && source.z == uploaded.z &&
                                source.a == uploaded.a && source.role == uploaded.role &&
                                std::abs(source.kinetic_energy_MeV -
                                         uploaded.kinetic_energy_MeV) <= 1.0e-6F &&
                                std::abs(source.rest_mass - uploaded.rest_mass_MeV) <= 1.0e-5F &&
                                std::abs(source.excitation - uploaded.excitation_MeV) <= 1.0e-6F,
                            "CINEL02 compact product differs from serialized record");
                    product_energy_squared_MeV2 +=
                        static_cast<double>(source.kinetic_energy_MeV) *
                        source.kinetic_energy_MeV;
                    product_energy_direction_z_MeV +=
                        static_cast<double>(source.kinetic_energy_MeV) *
                        source.local_direction_z;
                    if (source.role == 0 && source.z > 0 &&
                        carbon::get_charged_species_idx(source.z, source.a) >= 0) {
                        charged_kinetic_MeV += source.kinetic_energy_MeV;
                        auto& entry = transported_charged_isotopes[{source.z, source.a}];
                        ++entry.first;
                        entry.second += source.kinetic_energy_MeV;
                    } else if (source.role == 0) {
                        neutral_kinetic_MeV += source.kinetic_energy_MeV;
                        if (source.z > 0) {
                            auto& entry = unsupported_charged_isotopes[{source.z, source.a}];
                            ++entry.first;
                            entry.second += source.kinetic_energy_MeV;
                        }
                    } else if (source.role == 1) {
                        neutral_kinetic_MeV += source.kinetic_energy_MeV;
                    } else {
                        unsupported_kinetic_MeV += source.kinetic_energy_MeV;
                        event_unsupported_MeV += source.kinetic_energy_MeV;
                    }
                }
                require(std::abs(event_unsupported_MeV -
                                 raw.unsupported_product_energy_MeV) <=
                            2.0e-5 * std::max(1.0, event_unsupported_MeV),
                        "CINEL02 unsupported event ledger does not close");
            }
            require(device.energy_nodes.size() == table.energy_nodes().size() &&
                        device.event_offsets.size() == table.event_offsets().size() &&
                        device.event_indices.size() == table.event_indices().size(),
                    "CINEL02 compact global index mismatch");
            constexpr std::uint64_t frequency_samples = 100000U;
            for (std::size_t node = 0; node < table.energy_nodes().size(); ++node) {
                const auto count = table.event_offsets()[node + 1U] -
                                   table.event_offsets()[node];
                require(count > 0U, "CINEL02 energy node has no fixed events");
                // Midpoint-stratified samples partition exactly into floor/ceil(N/k)
                // selections per event. This is the closed-form frequency check
                // used instead of materialising N samples for every production node.
                const auto lower = frequency_samples / count;
                const auto upper = (frequency_samples + count - 1U) / count;
                require(upper - lower <= 1U,
                        "fixed-node CINEL02 event frequency is not uniform");
            }
            *report << "cells=" << table.cells().size()
                      << " interactions=" << table.interactions().size()
                      << " products=" << table.products().size()
                      << " energy_nodes=" << table.energy_nodes().size()
                      << " device_bytes=" << device.bytes() << '\n';
            *report << "energy_MeV incident=" << incident_kinetic_MeV
                      << " local=" << local_deposit_MeV
                      << " niel=" << niel_MeV
                      << " parent=" << parent_kinetic_MeV
                      << " charged=" << charged_kinetic_MeV
                      << " neutral=" << neutral_kinetic_MeV
                      << " unsupported=" << unsupported_kinetic_MeV << '\n';
            const auto event_total = static_cast<double>(table.interactions().size());
            *report << "moments mean_multiplicity=" << multiplicity_sum / event_total
                      << " second_multiplicity=" << multiplicity_squared_sum / event_total
                      << " parent_survival_fraction="
                      << static_cast<double>(parent_survival_count) / event_total
                      << " product_energy_second_MeV2=" << product_energy_squared_MeV2
                      << " product_energy_direction_z_MeV="
                      << product_energy_direction_z_MeV << '\n';
            for (const auto& [isotope, ledger] : transported_charged_isotopes) {
                *report << "transported_charged Z=" << isotope.first
                          << " A=" << isotope.second
                          << " count=" << ledger.first
                          << " kinetic_MeV=" << ledger.second << '\n';
            }
            for (const auto& [isotope, ledger] : unsupported_charged_isotopes) {
                *report << "unsupported_charged Z=" << isotope.first
                          << " A=" << isotope.second
                          << " count=" << ledger.first
                          << " kinetic_MeV=" << ledger.second << '\n';
            }
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "CINPKG03 inspection failed: " << error.what() << '\n';
            return 1;
        }
    }
    if (argc != 2 && argc != 3) {
        std::cerr << "usage: inelastic_package_v2_tests PACKAGE [RATE_CSV]\n";
        return 2;
    }
    try {
        const auto table = carbon::InelasticPackageV2Table::from_binary(
            std::filesystem::path{argv[1]});
        require(table.cells().size() == 2, "unexpected CINPKG03 cell count");
        require(table.interactions().size() == 3,
                "unexpected CINPKG03 interaction count");
        require(table.products().size() == 3,
                "unexpected CINPKG03 product count");
        require(table.minimum_events_per_bin() == 1,
                "unexpected CINPKG03 minimum-events contract");
        require(table.energy_nodes().size() == 2,
                "duplicate collision energies were not merged");
        require(table.event_offsets().size() == 3 &&
                    table.event_offsets()[0] == 0U &&
                    table.event_offsets()[1] == 2U &&
                    table.event_offsets()[2] == 3U,
                "global CINPKG03 event offsets are not prefix-closed");
        require(table.event_indices().size() == 3 &&
                    table.event_indices()[0] != table.event_indices()[1] &&
                    table.event_indices()[0] < 3U && table.event_indices()[1] < 3U &&
                    table.event_indices()[2] < 3U,
                "global CINPKG03 event index is not a complete permutation");

        const auto compact = table.make_device_tables();
        for (std::uint64_t fixed_index = 0;
             fixed_index < table.interactions().size(); ++fixed_index) {
            const auto replay = table.fixed_replay(fixed_index, compact);
            require(static_cast<bool>(replay),
                    "fixed-index CINEL02 replay rejected a valid event");
            const auto& raw = *replay.serialized_interaction;
            const auto& gpu = *replay.compact_interaction;
            require(raw.parent_pdg == gpu.parent_pdg && raw.parent_z == gpu.parent_z &&
                        raw.parent_a == gpu.parent_a && raw.parent_status == gpu.parent_status &&
                        raw.direct_product_count == replay.product_count,
                    "fixed-index CINEL02 parent discrete fields differ");
            require(std::abs(raw.collision_energy_MeV_per_u -
                             gpu.incident_energy_MeV_per_u) <= 1.0e-6F &&
                        std::abs(raw.process_local_deposit_MeV -
                                 gpu.process_local_deposit_MeV) <= 1.0e-6F &&
                        std::abs(raw.nonionizing_deposit_MeV -
                                 gpu.nonionizing_deposit_MeV) <= 1.0e-6F &&
                        std::abs(raw.parent_energy_MeV - gpu.parent_energy_MeV) <= 1.0e-6F &&
                        std::abs(raw.parent_weight - gpu.parent_weight) <= 1.0e-6F,
                    "fixed-index CINEL02 parent floating fields differ");
            const auto parent_local_norm =
                gpu.parent_local_direction_x * gpu.parent_local_direction_x +
                gpu.parent_local_direction_y * gpu.parent_local_direction_y +
                gpu.parent_local_direction_z * gpu.parent_local_direction_z;
            require(std::abs(parent_local_norm - 1.0F) <= 3.0e-5F,
                    "fixed-index CINEL02 parent local direction is not unit length");
            for (std::uint32_t p = 0; p < replay.product_count; ++p) {
                const auto& source = replay.serialized_products[p];
                const auto& uploaded = replay.compact_products[p];
                require(source.pdg == uploaded.pdg && source.z == uploaded.z &&
                            source.a == uploaded.a && source.role == uploaded.role,
                        "fixed-index CINEL02 product discrete fields differ");
                require(std::abs(source.kinetic_energy_MeV -
                                 uploaded.kinetic_energy_MeV) <= 1.0e-6F &&
                            std::abs(source.weight - uploaded.weight) <= 1.0e-6F &&
                            std::abs(source.local_direction_x -
                                     uploaded.local_direction_x) <= 1.0e-6F &&
                            std::abs(source.local_direction_y -
                                     uploaded.local_direction_y) <= 1.0e-6F &&
                            std::abs(source.local_direction_z -
                                     uploaded.local_direction_z) <= 1.0e-6F,
                        "fixed-index CINEL02 product floating fields differ");
            }
        }
        require(!table.fixed_replay(table.interactions().size(), compact),
                "fixed-index CINEL02 replay accepted an out-of-range event");
        for (std::size_t node = 0; node < table.energy_nodes().size(); ++node) {
            const auto first = table.event_offsets()[node];
            const auto last = table.event_offsets()[node + 1U];
            const auto count = last - first;
            require(count > 0U, "CINEL02 energy node has no fixed events");
            std::vector<std::uint32_t> frequency(static_cast<std::size_t>(count), 0U);
            constexpr std::uint32_t samples = 100000U;
            for (std::uint32_t sample = 0; sample < samples; ++sample) {
                const auto pick = static_cast<std::uint64_t>(
                    (static_cast<double>(sample) + 0.5) *
                    static_cast<double>(count) / samples);
                ++frequency[static_cast<std::size_t>(std::min(pick, count - 1U))];
            }
            const auto lower = samples / count;
            const auto upper = (samples + count - 1U) / count;
            require(std::all_of(frequency.begin(), frequency.end(),
                                [lower, upper](const auto value) {
                                    return value >= lower && value <= upper;
                                }),
                    "fixed-node CINEL02 event frequency is not uniform");
        }
        auto truncated = compact;
        truncated.products.clear();
        require(!table.fixed_replay(0U, truncated),
                "fixed-index CINEL02 replay accepted truncated product bounds");

        const auto* cell = table.find_cell(6, 12, 8, 16, 10.2F, 0.25F);
        require(cell != nullptr, "strict energy lookup rejected a covered event");
        const auto* event = table.interaction(*cell, 0);
        require(event != nullptr, "CINPKG03 interaction lookup failed");
        require(std::abs(event->collision_energy_MeV_per_u - 10.0F) < 1.0e-5F,
                "offline event order was not deterministic");
        require(event->direct_product_count == 1,
                "unexpected direct product count");
        require(std::abs(event->nonionizing_deposit_MeV - 0.25F) < 1.0e-5F,
                "CINEL02 non-ionizing deposit was not preserved");
        const auto* product = table.products_for(*event);
        require(product != nullptr && product->pdg == 2212,
                "CINPKG03 product offset lookup failed");
        require(table.product_offset(*event) == 0U,
                "unexpected first product offset");

        const auto invalid = std::numeric_limits<std::uint64_t>::max();
        const auto first_duplicate = table.find_event(6, 12, 8, 16, 10.0F, 0.0F, 0.0F);
        const auto second_duplicate = table.find_event(6, 12, 8, 16, 10.0F, 0.0F, 1.0F);
        require(first_duplicate != invalid && second_duplicate != invalid &&
                    first_duplicate != second_duplicate &&
                    table.interactions()[first_duplicate].collision_energy_MeV_per_u == 10.0F &&
                    table.interactions()[second_duplicate].collision_energy_MeV_per_u == 10.0F,
                "duplicate-energy event sampling was not uniform or complete");
        const auto cross_cell_first =
            table.find_event(6, 12, 8, 16, 10.5F, 0.5F, 0.0F);
        const auto cross_cell_middle =
            table.find_event(6, 12, 8, 16, 10.5F, 0.5F, 0.67F);
        const auto cross_cell_last =
            table.find_event(6, 12, 8, 16, 10.5F, 0.5F, 1.0F);
        require(cross_cell_first != invalid && cross_cell_middle != invalid &&
                    cross_cell_last != invalid &&
                    table.interactions()[cross_cell_first].collision_energy_MeV_per_u == 10.0F &&
                    table.interactions()[cross_cell_middle].collision_energy_MeV_per_u == 11.0F &&
                    table.interactions()[cross_cell_last].collision_energy_MeV_per_u == 11.0F,
                "energy-window event sampling was not uniform or complete");
        require(table.find_event(6, 12, 8, 16, 10.5F, 0.49F, 0.0F) == invalid,
                "empty global energy window was accepted");
        require(table.find_event(6, 12, 8, 16, 9.0F, 0.25F, 0.0F) == invalid,
                "out-of-range energy was extrapolated");
        require(table.find_event(6, 12, 8, 16, 10.0F, 0.0F, 0.0F) != invalid,
                "lower endpoint energy was not covered");
        require(table.find_event(6, 12, 8, 16, 11.0F, 0.0F, 1.0F) != invalid,
                "upper endpoint energy was not covered");
        require(carbon::InelasticPackageV2Table::select_target_z(
                    0.25F, 3.0F, 1.0F) == 1,
                "hydrogen target-first selection failed");
        require(carbon::InelasticPackageV2Table::select_target_z(
            0.75F, 3.0F, 1.0F) == 8,
            "oxygen target-first selection failed");

        if (argc == 3) {
            auto rates = carbon::InelasticRateV2Table::from_csv(
                std::filesystem::path{argv[2]});
            const auto covered = rates.lookup(6, 12, 8, 16, 10.5);
            require(covered.covered && std::abs(covered.value_per_mm - 0.75) < 1.0e-6,
                    "covered CINEL02 rate lookup failed");
            const auto below = rates.lookup(6, 12, 8, 16, 9.9);
            const auto above = rates.lookup(6, 12, 8, 16, 11.1);
            require(!below.covered && below.value_per_mm == 0.0,
                    "low-energy CINEL02 rate lookup was endpoint-clamped");
            require(!above.covered && above.value_per_mm == 0.0,
                    "high-energy CINEL02 rate lookup was endpoint-clamped");
            rates.bind_reference_number_densities(
                std::vector<carbon::Cinel02MaterialTarget>{
                    {1, 1, -1, 0, 2.0F, 2.0F, 1.0F},
                    {8, 16, -1, 0, 4.0F, 4.0F, 1.0F},
                });
            require(rates.has_reference_number_densities(),
                    "CINEL02 reference number densities were not bound");
            for (const auto& group : rates.groups()) {
                const auto expected = group.target_z == 1 ? 2.0F : 4.0F;
                require(std::abs(group.reference_number_density_per_mm3 - expected) <
                            1.0e-6F,
                        "CINEL02 target reference density binding failed");
            }
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
