// Shared SYCL device inelastic helper. Included inside namespace carbon anonymous namespace.

inline InelasticProductSet sample_carbon_inelastic_products_device(
    const float incident_energy_MeV,
    const float pos_x_mm,
    const float pos_y_mm,
    const float pos_z_mm,
    const Direction3F parent_dir,
    const float u_channel,
    const float u_energy,
    const float u_theta,
    const float u_phi,
    const float u_sub) noexcept {

    InelasticProductSet set{};
    const auto e_per_u = incident_energy_MeV / 12.0F;
    if (e_per_u <= 0.5F) {
        set.local_deposit_MeV = incident_energy_MeV;
        return set;
    }

    // FRED Section 8.1: Energy-dependent charged fragment energy retention factor
    const auto energy_loss_ratio = sycl::fmin(0.18F, sycl::fmax(0.0F, (e_per_u - 100.0F) / 300.0F * 0.18F));
    const auto eta_charged = 1.0F - energy_loss_ratio;

    // Binding energy subtraction near Bragg peak (Q ~ 16 MeV)
    const auto bind_loss_per_u = (e_per_u < 50.0F) ? sycl::fmin(e_per_u * 0.10F, 16.0F / 12.0F) : 0.0F;
    const auto effective_e_u = sycl::fmax(0.1F, e_per_u - bind_loss_per_u);

    constexpr float two_pi = 6.2831853071795864769F;
    const auto phi = two_pi * u_phi;

    // Helper to sample product with Goldhaber forward Gaussian + spectator deceleration + light fragment forward penetration
    auto make_product = [&](int16_t z, int16_t a, float angle_scale, bool has_target_halo) {
        SecondaryParticle p{};
        p.z = z;
        p.a = a;
        p.pos_x_mm = pos_x_mm;
        p.pos_y_mm = pos_y_mm;
        p.pos_z_mm = pos_z_mm;
        p.weight = 1.0F;

        const auto af = sycl::fmax(1.0F, static_cast<float>(a));

        float long_e_u = effective_e_u;
        if (a >= 6) {
            // Heavy spectator projectile residue (B, Be, Li): Goldhaber deceleration shift
            const auto v_ratio = 1.0F - (12.0F - af) / 24.0F;
            long_e_u = effective_e_u * (v_ratio * v_ratio) * (1.0F - 0.03F * u_energy);
        } else {
            // Light stripped projectile fragments (p, d, alpha): keep forward beam velocity
            const auto base_frac = (a == 4) ? 0.88F : 0.82F;
            long_e_u = effective_e_u * (base_frac + 0.08F * (1.0F - u_energy));
        }

        p.energy_MeV = sycl::fmax(0.0F, af * long_e_u * eta_charged);

        const auto p_nucleon = sycl::sqrt(sycl::fmax(1.0e-4F, long_e_u * (long_e_u + 1863.0F)));
        // Goldhaber fragmentation momentum dispersion (sigma_0 ~ 90 MeV/c)
        const auto goldhaber_factor = sycl::sqrt(sycl::fmax(0.01F, (12.0F - af) / (11.0F * af)));
        const auto sigma_th = (90.0F / p_nucleon) * goldhaber_factor * angle_scale;

        float theta = 0.0F;
        // Light fragments (p, d, alpha): 22% probability of large-angle target recoil/evaporation
        if (has_target_halo && u_sub > 0.78F) {
            constexpr float mean_halo_angle_rad = 0.45F;
            theta = -mean_halo_angle_rad * sycl::log(sycl::fmax(u_theta, 1.0e-8F));
            theta = sycl::fmin(theta, 1.50F);
            p.energy_MeV *= 0.65F;
        } else {
            theta = sigma_th * sycl::sqrt(-2.0F * sycl::log(sycl::fmax(u_theta, 1.0e-10F)));
        }

        const auto sin_th = sycl::sin(theta);
        const auto cos_th = sycl::cos(theta);

        const auto local_x = sin_th * sycl::cos(phi);
        const auto local_y = sin_th * sycl::sin(phi);
        const auto local_z = cos_th;

        const auto rot_dir = rotate_local_direction(local_x, local_y, local_z, parent_dir);
        p.dir_x = rot_dir.x;
        p.dir_y = rot_dir.y;
        p.dir_z = rot_dir.z;
        return p;
    };

    const auto base_local_MeV = 12.0F + 0.04F * e_per_u;

    // Branching channels based on Geant4 INCL++ and FRED measurements:
    if (u_channel < 0.18F) {
        // Channel 0: B-11 + p + (n)
        set.products[0] = make_product(5, 11, 1.0F, false);
        set.products[1] = make_product(1, 1, 1.2F, true);
        set.count = 2;
        set.local_deposit_MeV = base_local_MeV;
    } else if (u_channel < 0.32F) {
        // Channel 1: B-10 + d / p + (n)
        set.products[0] = make_product(5, 10, 1.0F, false);
        if (u_sub < 0.5F) {
            set.products[1] = make_product(1, 2, 1.2F, true);
        } else {
            set.products[1] = make_product(1, 1, 1.2F, true);
        }
        set.count = 2;
        set.local_deposit_MeV = base_local_MeV + 6.0F;
    } else if (u_channel < 0.44F) {
        // Channel 2: Be-9 + alpha
        set.products[0] = make_product(4, 9, 1.0F, false);
        set.products[1] = make_product(2, 4, 1.1F, true);
        set.count = 2;
        set.local_deposit_MeV = base_local_MeV + 8.0F;
    } else if (u_channel < 0.54F) {
        // Channel 3: Li-7 + alpha + p
        set.products[0] = make_product(3, 7, 1.0F, false);
        set.products[1] = make_product(2, 4, 1.1F, true);
        set.products[2] = make_product(1, 1, 1.2F, true);
        set.count = 3;
        set.local_deposit_MeV = base_local_MeV + 10.0F;
    } else if (u_channel < 0.72F) {
        // Channel 4: Triple alpha (3 x He-4)
        set.products[0] = make_product(2, 4, 1.0F, true);
        set.products[1] = make_product(2, 4, 1.1F, true);
        set.products[2] = make_product(2, 4, 1.2F, true);
        set.count = 3;
        set.local_deposit_MeV = 7.3F + 0.02F * e_per_u;
    } else if (u_channel < 0.88F) {
        // Channel 5: Alpha + 2 protons + (2n)
        set.products[0] = make_product(2, 4, 1.1F, true);
        set.products[1] = make_product(1, 1, 1.2F, true);
        set.products[2] = make_product(1, 1, 1.3F, true);
        set.count = 3;
        set.local_deposit_MeV = base_local_MeV + 12.0F;
    } else {
        // Channel 6: Protons (4 x p + 4n)
        set.products[0] = make_product(1, 1, 1.2F, true);
        set.products[1] = make_product(1, 1, 1.2F, true);
        set.products[2] = make_product(1, 1, 1.3F, true);
        set.products[3] = make_product(1, 1, 1.4F, true);
        set.count = 4;
        set.local_deposit_MeV = base_local_MeV + 16.0F;
    }

    return set;
}
