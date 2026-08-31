#include "carbon/inelastic_package_v2.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 3 && std::string{argv[1]} == "--inspect") {
        try {
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
            require(device.energy_nodes.size() == table.energy_nodes().size() &&
                        device.event_offsets.size() == table.event_offsets().size() &&
                        device.event_indices.size() == table.event_indices().size(),
                    "CINEL02 compact global index mismatch");
            std::cout << "cells=" << table.cells().size()
                      << " interactions=" << table.interactions().size()
                      << " products=" << table.products().size()
                      << " energy_nodes=" << table.energy_nodes().size()
                      << " device_bytes=" << device.bytes() << '\n';
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
                "cross-cell global energy window sampling failed");
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
