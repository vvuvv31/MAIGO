#pragma once

#include <filesystem>
#include <string>

namespace carbon {

enum class PackageIdentityValidation {
    strict,
    warn,
    off,
};

struct PackageIdentityExpectation {
    std::string kind;
    int projectile_atomic_number{0};
    int projectile_mass_number{0};
    std::string material;
    std::string physics_model;
    double minimum_energy_MeV_per_u{0.0};
    double maximum_energy_MeV_per_u{0.0};
};

// Validates a v2 sibling .compiled.json sidecar.  Older repository packages
// may be described by the explicit override manifest while binary layouts stay v1.
void validate_package_identity(const std::filesystem::path& package_path,
                               const PackageIdentityExpectation& expectation,
                               PackageIdentityValidation mode,
                               const std::filesystem::path& override_manifest_path);

[[nodiscard]] PackageIdentityValidation parse_package_identity_validation(
    const std::string& value);

}  // namespace carbon
