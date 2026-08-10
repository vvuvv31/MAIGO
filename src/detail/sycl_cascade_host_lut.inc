// Shared SYCL transport helpers included by transport_sycl_legacy.cpp and
// transport_sycl.cpp. Do not compile this file as a standalone translation unit.
//
// Include context (required):
//   - inside `namespace carbon`
//   - for helpers below, typically inside the file-local anonymous namespace
//   - SYCL headers and carbon/* headers already included by the parent .cpp
//
// Keep body text bit-identical to the historical legacy definitions so OFF/ON
// isolation remains numerically stable.

// Host: interpolate each cascade projectile's XS samples onto the water SP energy
// grid so the device kernel only does clamp/index/lerp (no binary search per step).
std::vector<float> build_cascade_xs_energy_lut(
    const CascadePackageTable& packages,
    const std::vector<double>& energies_MeVu) {
    const auto projectile_count = packages.projectiles().size();
    const auto energy_count = energies_MeVu.size();
    std::vector<float> lut(projectile_count * energy_count, 0.0F);
    const auto& samples = packages.cross_sections();
    for (std::size_t projectile_index = 0; projectile_index < projectile_count;
         ++projectile_index) {
        const auto& projectile = packages.projectiles()[projectile_index];
        const auto offset = projectile.cross_section_offset;
        const auto count = projectile.cross_section_count;
        if (count == 0) {
            continue;
        }
        for (std::size_t energy_index = 0; energy_index < energy_count; ++energy_index) {
            const auto energy = static_cast<float>(energies_MeVu[energy_index]);
            std::uint32_t lower = 0;
            std::uint32_t upper = count;
            while (lower < upper) {
                const auto middle = lower + (upper - lower) / 2U;
                if (samples[offset + middle].energy_MeV_per_u < energy) {
                    lower = middle + 1U;
                } else {
                    upper = middle;
                }
            }
            float value = 0.0F;
            if (lower == 0U) {
                value = samples[offset].macroscopic_cross_section_per_mm;
            } else if (lower >= count) {
                value = samples[offset + count - 1U].macroscopic_cross_section_per_mm;
            } else {
                const auto& lo = samples[offset + lower - 1U];
                const auto& hi = samples[offset + lower];
                const auto interval = hi.energy_MeV_per_u - lo.energy_MeV_per_u;
                const auto frac =
                    interval > 0.0F
                        ? std::clamp((energy - lo.energy_MeV_per_u) / interval, 0.0F, 1.0F)
                        : 0.0F;
                value = lo.macroscopic_cross_section_per_mm +
                        frac * (hi.macroscopic_cross_section_per_mm -
                                lo.macroscopic_cross_section_per_mm);
            }
            lut[projectile_index * energy_count + energy_index] = value;
        }
    }
    return lut;
}

// Build a material mass-XS LUT in the projectile order of the production
// package. Sparse material packages fall back species-by-species to water.
std::vector<float> build_aligned_material_cascade_xs_lut(
    const CascadePackageTable& production_packages,
    const CascadePackageTable& material_packages,
    const std::vector<double>& energies_MeVu,
    const float material_reference_density_g_per_cm3) {
    auto lut =
        build_cascade_xs_energy_lut(production_packages, energies_MeVu);
    const auto energy_count = energies_MeVu.size();
    const auto& samples = material_packages.cross_sections();
    for (std::size_t projectile_index = 0;
         projectile_index < production_packages.projectiles().size();
         ++projectile_index) {
        const auto& production_projectile =
            production_packages.projectiles()[projectile_index];
        const auto* projectile = material_packages.find_projectile(
            production_projectile.atomic_number,
            production_projectile.mass_number);
        if (projectile == nullptr || projectile->cross_section_count == 0) {
            continue;
        }
        const auto offset = projectile->cross_section_offset;
        const auto count = projectile->cross_section_count;
        for (std::size_t energy_index = 0; energy_index < energy_count;
             ++energy_index) {
            const auto energy =
                static_cast<float>(energies_MeVu[energy_index]);
            std::uint32_t lower = 0;
            std::uint32_t upper = count;
            while (lower < upper) {
                const auto middle = lower + (upper - lower) / 2U;
                if (samples[offset + middle].energy_MeV_per_u < energy) {
                    lower = middle + 1U;
                } else {
                    upper = middle;
                }
            }
            float value = 0.0F;
            if (lower == 0U) {
                value = samples[offset].macroscopic_cross_section_per_mm;
            } else if (lower >= count) {
                value =
                    samples[offset + count - 1U].macroscopic_cross_section_per_mm;
            } else {
                const auto& lo = samples[offset + lower - 1U];
                const auto& hi = samples[offset + lower];
                const auto interval =
                    hi.energy_MeV_per_u - lo.energy_MeV_per_u;
                const auto fraction =
                    interval > 0.0F
                        ? std::clamp(
                              (energy - lo.energy_MeV_per_u) / interval,
                              0.0F, 1.0F)
                        : 0.0F;
                value = lo.macroscopic_cross_section_per_mm +
                        fraction *
                            (hi.macroscopic_cross_section_per_mm -
                             lo.macroscopic_cross_section_per_mm);
            }
            lut[projectile_index * energy_count + energy_index] =
                value / material_reference_density_g_per_cm3;
        }
    }
    return lut;
}
