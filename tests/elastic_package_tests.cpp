#include "carbon/elastic_package.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 4) {
        std::cerr << "usage: elastic_package_tests PACKAGE [Z A]\n";
        return EXIT_FAILURE;
    }
    try {
        const auto table = carbon::ElasticPackageTable::from_binary(argv[1]);
        require(table.header().version == 1, "unexpected ELPKG version");
        require(table.header().header_size == 60, "unexpected ELPKG header size");
        require(table.header().event_count == table.events().size(), "event count mismatch");
        require(table.header().product_count == table.products().size(), "product count mismatch");
        require(!table.energy_bins().empty() && !table.events().empty(), "empty ELPKG");
        if (argc == 4) {
            require(table.projectile_atomic_number() == std::stoi(argv[2]),
                    "projectile Z mismatch");
            require(table.projectile_mass_number() == std::stoi(argv[3]),
                    "projectile A mismatch");
        }
        for (std::size_t bin = 0; bin < table.energy_bins().size(); ++bin) {
            const auto span = table.event_span(bin);
            require(!span.empty(), "empty energy-bin event span");
            for (const auto& event : span) {
                require(event.continuation == 1, "event does not continue primary");
                require(std::isfinite(event.incident_energy_MeV_per_u),
                        "non-finite incident energy");
                const auto products = table.product_span(event);
                require(products.size() == event.product_count, "product span mismatch");
                for (const auto& product : products) {
                    require(std::isfinite(product.kinetic_energy_MeV),
                            "non-finite product energy");
                }
            }
        }
        require(table.event_span_for_energy(table.minimum_energy_MeV_per_u()).size() > 0,
                "energy lookup failed");
        require(table.event_span(table.energy_bins().size()).empty(),
                "out-of-range bin did not return empty span");
        require(table.product_span(table.events().size()).empty(),
                "out-of-range event did not return empty product span");
        std::cout << "ELPKG reader passed: bins=" << table.energy_bins().size()
                  << " events=" << table.events().size()
                  << " products=" << table.products().size() << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
