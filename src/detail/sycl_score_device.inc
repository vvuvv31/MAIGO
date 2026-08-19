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

// Deposit energy along +z with exponential attenuation from z0.
// lambda_mm <= 0 → local dump in the production bin.
// renormalize=true → scale so full amount is deposited in remaining phantom
//   (used for interim neutral kerma; avoids high-E tail escape underdose).
// renormalize=false → omit energy past phantom (electronic delta near exit).
inline void score_exponential_depth(
    const float amount_MeV,
    const float z_mm,
    const float bin_width_mm,
    const float phantom_length_mm,
    const std::uint32_t number_of_bins,
    const float lambda_mm,
    DoseAtomicT* dose_row,
    const bool renormalize = false) noexcept {
    if (amount_MeV <= 0.0F || dose_row == nullptr || number_of_bins == 0 ||
        bin_width_mm <= 0.0F) {
        return;
    }
    auto z0 = z_mm;
    if (z0 < 0.0F) {
        z0 = 0.0F;
    }
    if (z0 >= phantom_length_mm) {
        return;
    }
    if (lambda_mm <= 1.0e-3F) {
        auto bin = static_cast<int>(z0 / bin_width_mm);
        if (bin < 0) {
            bin = 0;
        }
        if (bin >= static_cast<int>(number_of_bins)) {
            bin = static_cast<int>(number_of_bins) - 1;
        }
        sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed, sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            atomic_dose(dose_row[static_cast<std::size_t>(bin)]);
        atomic_dose.fetch_add(static_cast<DoseAtomicT>(amount_MeV));
        return;
    }
    const auto inv_lambda = 1.0F / lambda_mm;
    const auto remaining = phantom_length_mm - z0;
    auto norm = 1.0F;
    if (renormalize) {
        const auto contained = 1.0F - sycl::exp(-remaining * inv_lambda);
        if (contained > 1.0e-6F) {
            norm = 1.0F / contained;
        }
    }
    const auto start_bin = static_cast<std::uint32_t>(z0 / bin_width_mm);
    for (std::uint32_t bin = start_bin; bin < number_of_bins; ++bin) {
        const auto z_lo =
            sycl::fmax(z0, static_cast<float>(bin) * bin_width_mm);
        const auto z_hi = sycl::fmin(phantom_length_mm,
                                     static_cast<float>(bin + 1U) * bin_width_mm);
        if (z_hi <= z_lo) {
            continue;
        }
        const auto s_lo = z_lo - z0;
        const auto s_hi = z_hi - z0;
        // ∫_{s_lo}^{s_hi} (1/λ) exp(-s/λ) ds
        const auto frac =
            (sycl::exp(-s_lo * inv_lambda) - sycl::exp(-s_hi * inv_lambda)) * norm;
        if (frac <= 0.0F) {
            continue;
        }
        sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed, sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            atomic_dose(dose_row[bin]);
        atomic_dose.fetch_add(static_cast<DoseAtomicT>(amount_MeV * frac));
        if (sycl::exp(-s_hi * inv_lambda) < 1.0e-5F) {
            break;
        }
    }
}

// Voxel analogue of score_exponential_depth. The proxy keeps the production
// voxel's transverse (x,y) index and distributes energy only along +z.
inline void score_exponential_voxel_depth(
    const float amount_MeV,
    const float z_mm,
    const float bin_width_mm,
    const float phantom_length_mm,
    const std::uint32_t number_of_bins,
    const float lambda_mm,
    const std::size_t voxel_plane_size,
    const std::size_t production_voxel_index,
    DoseAtomicT* voxel_dose,
    const bool renormalize = false) noexcept {
    if (amount_MeV <= 0.0F || voxel_dose == nullptr || number_of_bins == 0 ||
        voxel_plane_size == 0 || bin_width_mm <= 0.0F) {
        return;
    }
    auto z0 = z_mm;
    if (z0 < 0.0F) {
        z0 = 0.0F;
    }
    if (z0 >= phantom_length_mm) {
        return;
    }
    const auto voxel_in_plane = production_voxel_index % voxel_plane_size;
    if (lambda_mm <= 1.0e-3F) {
        auto bin = static_cast<int>(z0 / bin_width_mm);
        bin = sycl::max(0, sycl::min(bin, static_cast<int>(number_of_bins) - 1));
        sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed, sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            atomic_dose(voxel_dose[static_cast<std::size_t>(bin) * voxel_plane_size +
                                   voxel_in_plane]);
        atomic_dose.fetch_add(static_cast<DoseAtomicT>(amount_MeV));
        return;
    }
    const auto inv_lambda = 1.0F / lambda_mm;
    const auto remaining = phantom_length_mm - z0;
    auto norm = 1.0F;
    if (renormalize) {
        const auto contained = 1.0F - sycl::exp(-remaining * inv_lambda);
        if (contained > 1.0e-6F) {
            norm = 1.0F / contained;
        }
    }
    const auto start_bin = static_cast<std::uint32_t>(z0 / bin_width_mm);
    for (std::uint32_t bin = start_bin; bin < number_of_bins; ++bin) {
        const auto z_lo = sycl::fmax(z0, static_cast<float>(bin) * bin_width_mm);
        const auto z_hi = sycl::fmin(phantom_length_mm,
                                     static_cast<float>(bin + 1U) * bin_width_mm);
        if (z_hi <= z_lo) {
            continue;
        }
        const auto s_lo = z_lo - z0;
        const auto s_hi = z_hi - z0;
        const auto frac =
            (sycl::exp(-s_lo * inv_lambda) - sycl::exp(-s_hi * inv_lambda)) * norm;
        if (frac <= 0.0F) {
            continue;
        }
        sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed, sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            atomic_dose(voxel_dose[static_cast<std::size_t>(bin) * voxel_plane_size +
                                   voxel_in_plane]);
        atomic_dose.fetch_add(static_cast<DoseAtomicT>(amount_MeV * frac));
        if (sycl::exp(-s_hi * inv_lambda) < 1.0e-5F) {
            break;
        }
    }
}

// Energy-dependent electronic delta fraction (0 at low E → full scale at 400 MeV/u).
inline float electronic_buildup_fraction_at_energy(
    const float energy_MeVu, const float fraction_at_400) noexcept {
    if (fraction_at_400 <= 0.0F) {
        return 0.0F;
    }
    const auto scale = energy_MeVu > 0.0F ? energy_MeVu / 400.0F : 0.0F;
    const auto clamped = scale < 1.0F ? scale : 1.0F;
    return fraction_at_400 * clamped;
}

// The maximum transferable delta-electron energy, and therefore its range,
// falls with projectile velocity. Keep the configured MFP as the 400 MeV/u
// reference instead of applying the same transport length near the Bragg peak.
inline float electronic_buildup_mfp_at_energy(
    const float energy_MeVu, const float mfp_at_400_mm) noexcept {
    if (mfp_at_400_mm <= 0.0F || energy_MeVu <= 0.0F) {
        return 0.0F;
    }
    return mfp_at_400_mm * sycl::fmin(energy_MeVu / 400.0F, 1.0F);
}

// Kerma fraction: f0 for E<=200 MeV/u; linear ramp to f0*high_scale at E>=400.
inline float neutral_kerma_fraction_at_energy(const float energy_MeVu,
                                               const float f0,
                                               const float high_scale) noexcept {
    if (f0 <= 0.0F) {
        return 0.0F;
    }
    if (high_scale <= 1.0F || energy_MeVu <= 200.0F) {
        return f0;
    }
    auto t = (energy_MeVu - 200.0F) / 200.0F;
    if (t > 1.0F) {
        t = 1.0F;
    }
    return f0 * (1.0F + (high_scale - 1.0F) * t);
}

void add_unrestricted_depth_dose_device(
    DoseAtomicT amount_MeV, int bin,
    DoseAtomicT* unrestricted_secondary_dose_device) noexcept;

void score_secondary_dose_device(
    const DoseAtomicT amount_MeV,
    const bool is_neutral_lineage,
    const std::size_t species_index,
    const std::size_t neutral_origin,
    const int bin,
    const std::size_t number_of_bins,
    const bool enable_voxel_scoring,
    const bool enable_charged_origin_voxel_scoring,
    const std::size_t voxel_index,
    const std::size_t charged_origin_voxel_offset,
    const std::size_t neutral_origin_voxel_offset,
    DoseAtomicT* aggregate_secondary_dose_device,
    DoseAtomicT* fragment_dose_device,
    DoseAtomicT* voxel_dose_device,
    DoseAtomicT* charged_origin_voxel_dose_device,
    DoseAtomicT* neutral_origin_dose_device,
    DoseAtomicT* neutral_origin_voxel_dose_device) noexcept {
    if (amount_MeV <= DoseAtomicT{0}) {
        return;
    }
    // Neutral-transport lineage is accumulated by its dedicated origin scorer
    // and merged on the host. All other secondary deposits must reach the
    // total depth-dose scorer independently of optional species rows.
    if (!is_neutral_lineage && aggregate_secondary_dose_device != nullptr) {
        sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            atomic_aggregate(
                aggregate_secondary_dose_device[static_cast<std::size_t>(bin)]);
        atomic_aggregate.fetch_add(amount_MeV);
    }
    if (is_neutral_lineage) {
        sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed, sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            atomic_origin(neutral_origin_dose_device[neutral_origin * number_of_bins +
                                                     static_cast<std::size_t>(bin)]);
        atomic_origin.fetch_add(amount_MeV);
        if (enable_voxel_scoring && neutral_origin_voxel_dose_device != nullptr) {
            sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed, sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                atomic_voxel(
                    neutral_origin_voxel_dose_device[neutral_origin_voxel_offset +
                                                     voxel_index]);
            atomic_voxel.fetch_add(amount_MeV);
        }
        // Neutral-origin dose is a category split, not a replacement for the
        // aggregate voxel scorer used to write the total 3D dose.
        if (enable_voxel_scoring && voxel_dose_device != nullptr) {
            sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                atomic_total_voxel(voxel_dose_device[voxel_index]);
            atomic_total_voxel.fetch_add(amount_MeV);
        }
        // Also attribute to the Z/A species row so n/γ-born p/He match TOPAS.
        if (fragment_dose_device != nullptr) {
            sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                atomic_fragment(
                    fragment_dose_device[species_index * number_of_bins +
                                         static_cast<std::size_t>(bin)]);
            atomic_fragment.fetch_add(amount_MeV);
        }
        return;
    }
    if (fragment_dose_device != nullptr) {
        sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            atomic_fragment(
                fragment_dose_device[species_index * number_of_bins +
                                     static_cast<std::size_t>(bin)]);
        atomic_fragment.fetch_add(amount_MeV);
    }
    if (enable_voxel_scoring) {
        sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed, sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            atomic_voxel(voxel_dose_device[voxel_index]);
        atomic_voxel.fetch_add(amount_MeV);
        if (enable_charged_origin_voxel_scoring) {
            sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed, sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                atomic_category(
                    charged_origin_voxel_dose_device[charged_origin_voxel_offset +
                                                     voxel_index]);
            atomic_category.fetch_add(amount_MeV);
        }
    }
}

inline void score_letd_moments_device(
    LetAtomicT* moments,
    const std::size_t number_of_bins,
    const std::size_t bin,
    LetAtomicT* species_moments,
    const std::size_t species_count,
    const std::size_t species_category,
    LetAtomicT* isotope_moments,
    const std::size_t isotope_count,
    const std::size_t isotope_category,
    const float local_energy_deposit_MeV,
    const float total_collisional_loss_MeV,
    const float stopping_power_MeV_per_mm,
    const float density_g_per_cm3,
    const bool is_c12) noexcept {
    if (moments == nullptr || bin >= number_of_bins ||
        local_energy_deposit_MeV <= 0.0F || total_collisional_loss_MeV <= 0.0F ||
        stopping_power_MeV_per_mm <= 0.0F || density_g_per_cm3 <= 0.0F) {
        return;
    }
    const auto numerator =
        static_cast<LetAtomicT>(local_energy_deposit_MeV) *
        static_cast<LetAtomicT>(stopping_power_MeV_per_mm) /
        static_cast<LetAtomicT>(density_g_per_cm3);
    const auto denominator =
        static_cast<LetAtomicT>(total_collisional_loss_MeV);
    const auto accumulate = [&](const std::size_t offset,
                                const LetAtomicT value) {
        sycl::atomic_ref<LetAtomicT, sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            atomic_value(moments[offset * number_of_bins + bin]);
        atomic_value.fetch_add(value);
    };
    if (is_c12) {
        accumulate(0, numerator);
        accumulate(1, denominator);
    }
    accumulate(2, numerator);
    accumulate(3, denominator);
    if (species_moments != nullptr && species_category < species_count) {
        const auto species_accumulate = [&](const std::size_t moment,
                                            const LetAtomicT value) {
            const auto offset =
                (moment * species_count + species_category) * number_of_bins +
                bin;
            sycl::atomic_ref<LetAtomicT, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                atomic_value(species_moments[offset]);
            atomic_value.fetch_add(value);
        };
        species_accumulate(0, numerator);
        species_accumulate(1, denominator);
    }
    if (isotope_moments != nullptr && isotope_category < isotope_count) {
        const auto isotope_accumulate = [&](const std::size_t moment,
                                            const LetAtomicT value) {
            const auto offset =
                (moment * isotope_count + isotope_category) * number_of_bins +
                bin;
            sycl::atomic_ref<LetAtomicT, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                atomic_value(isotope_moments[offset]);
            atomic_value.fetch_add(value);
        };
        isotope_accumulate(0, numerator);
        isotope_accumulate(1, denominator);
    }
}

// Score an optional diagnostic-species birth into histogram buffers.
// No-op when buffers are null or the product is not a light isotope.
inline void score_fragment_birth_device(
    std::uint64_t* counts_by_generation,
    double* ke_sum_by_generation,
    std::uint64_t* mevu_hist,
    std::uint64_t* depth_hist,
    const std::size_t depth_bin_count,
    const float depth_bin_width_mm,
    std::uint64_t* cos_hist,
    std::uint64_t* parent_mevu_hist,
    std::uint64_t* parent_z_hist,
    std::uint64_t* parent_product_mevu_hist,
    const int atomic_number,
    const int mass_number,
    const float kinetic_energy_MeV,
    const float position_z_mm,
    const float direction_z,
    const std::uint8_t generation,
    const int parent_atomic_number,
    const int parent_mass_number,
    const float parent_kinetic_energy_MeV) noexcept {
    if (counts_by_generation == nullptr || mass_number <= 0 ||
        kinetic_energy_MeV <= 0.0F) {
        return;
    }
    const auto category = light_isotope_category(atomic_number, mass_number);
    if (category >= light_isotope_category_count) {
        return;
    }
    const auto gen = birth_generation_bin(generation);
    const auto gen_index = category * birth_generation_bin_count + gen;
    {
        sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            atomic_count(counts_by_generation[gen_index]);
        atomic_count.fetch_add(1ULL);
    }
    if (ke_sum_by_generation != nullptr) {
        sycl::atomic_ref<double, sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            atomic_ke(ke_sum_by_generation[gen_index]);
        atomic_ke.fetch_add(static_cast<double>(kinetic_energy_MeV));
    }
    if (mevu_hist != nullptr) {
        const auto ebin =
            birth_mevu_bin(static_cast<double>(kinetic_energy_MeV), mass_number);
        const auto index =
            birth_hist_index(category, gen, ebin, birth_mevu_bin_count);
        sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            atomic_e(mevu_hist[index]);
        atomic_e.fetch_add(1ULL);
    }
    if (depth_hist != nullptr && depth_bin_count > 0 && depth_bin_width_mm > 0.0F &&
        position_z_mm >= 0.0F) {
        auto zbin = static_cast<std::size_t>(position_z_mm / depth_bin_width_mm);
        if (zbin >= depth_bin_count) {
            zbin = depth_bin_count - 1;
        }
        const auto index = birth_hist_index(category, gen, zbin, depth_bin_count);
        sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            atomic_z(depth_hist[index]);
        atomic_z.fetch_add(1ULL);
    }
    if (cos_hist != nullptr) {
        const auto cbin = birth_cos_bin(static_cast<double>(direction_z));
        const auto index =
            birth_hist_index(category, gen, cbin, birth_cos_bin_count);
        sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            atomic_c(cos_hist[index]);
        atomic_c.fetch_add(1ULL);
    }
    if (parent_mevu_hist != nullptr) {
        const auto pbin = birth_parent_mevu_bin(
            static_cast<double>(parent_kinetic_energy_MeV), parent_mass_number);
        const auto index =
            birth_hist_index(category, gen, pbin, birth_parent_mevu_bin_count);
        sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            atomic_p(parent_mevu_hist[index]);
        atomic_p.fetch_add(1ULL);
    }
    if (parent_z_hist != nullptr) {
        const auto zbin = birth_parent_z_bin(parent_atomic_number);
        const auto index =
            birth_hist_index(category, gen, zbin, birth_parent_z_bin_count);
        sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            atomic_pz(parent_z_hist[index]);
        atomic_pz.fetch_add(1ULL);
    }
    if (parent_product_mevu_hist != nullptr) {
        const auto pbin = birth_parent_mevu_bin(
            static_cast<double>(parent_kinetic_energy_MeV), parent_mass_number);
        const auto ebin =
            birth_mevu_bin(static_cast<double>(kinetic_energy_MeV), mass_number);
        const auto index = birth_joint_index(category, gen, pbin, ebin);
        sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            atomic_j(parent_product_mevu_hist[index]);
        atomic_j.fetch_add(1ULL);
    }
}
