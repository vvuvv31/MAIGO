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

    constexpr float two_pi = 6.2831853071795864769F;
    const auto phi = two_pi * u_phi;

    // Helper to sample product with Goldhaber forward Gaussian + wide target recoil
    auto make_product = [&](int16_t z, int16_t a, float ke_per_u, float angle_scale, bool has_target_halo) {
        SecondaryParticle p{};
        p.z = z;
        p.a = a;
        p.energy_MeV = sycl::fmax(0.0F, static_cast<float>(a) * ke_per_u);
        p.pos_x_mm = pos_x_mm;
        p.pos_y_mm = pos_y_mm;
        p.pos_z_mm = pos_z_mm;
        p.weight = 1.0F;

        const auto af = sycl::fmax(1.0F, static_cast<float>(a));
        const auto p_nucleon = sycl::sqrt(sycl::fmax(1.0e-4F, ke_per_u * (ke_per_u + 1863.0F)));
        // Goldhaber fragmentation momentum dispersion (sigma_0 ~ 90 MeV/c)
        const auto goldhaber_factor = sycl::sqrt(sycl::fmax(0.01F, (12.0F - af) / (11.0F * af)));
        const auto sigma_th = (90.0F / p_nucleon) * goldhaber_factor * angle_scale;

        float theta = 0.0F;
        // If has_target_halo (light fragments p, d, alpha), 22% probability of large-angle target recoil/evaporation
        if (has_target_halo && u_sub > 0.78F) {
            // Exponential angular tail with mean ~ 0.45 rad (~26 deg)
            constexpr float mean_halo_angle_rad = 0.45F;
            theta = -mean_halo_angle_rad * sycl::log(sycl::fmax(u_theta, 1.0e-8F));
            theta = sycl::fmin(theta, 1.50F); // Cap at 85 degrees
            p.energy_MeV *= 0.65F; // Target fragments carry lower kinetic energy
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

    // Branching channels based on Geant4 INCL++ and FRED measurements:
    if (u_channel < 0.18F) {
        // Channel 0: B-11 + p + (n)
        const auto ke_u = e_per_u * (0.95F - 0.04F * u_energy);
        set.products[0] = make_product(5, 11, ke_u, 1.0F, false);
        set.products[1] = make_product(1, 1, e_per_u * (0.75F + 0.15F * u_sub), 1.2F, true);
        set.count = 2;
        set.local_deposit_MeV = 12.0F;
    } else if (u_channel < 0.32F) {
        // Channel 1: B-10 + d / p + (n)
        const auto ke_u = e_per_u * (0.93F - 0.05F * u_energy);
        set.products[0] = make_product(5, 10, ke_u, 1.0F, false);
        if (u_sub < 0.5F) {
            set.products[1] = make_product(1, 2, e_per_u * 0.70F, 1.2F, true);
        } else {
            set.products[1] = make_product(1, 1, e_per_u * 0.75F, 1.2F, true);
        }
        set.count = 2;
        set.local_deposit_MeV = 18.0F;
    } else if (u_channel < 0.44F) {
        // Channel 2: Be-9 + alpha
        const auto ke_u = e_per_u * (0.92F - 0.06F * u_energy);
        set.products[0] = make_product(4, 9, ke_u, 1.0F, false);
        set.products[1] = make_product(2, 4, e_per_u * (0.80F + 0.10F * u_sub), 1.1F, true);
        set.count = 2;
        set.local_deposit_MeV = 20.0F;
    } else if (u_channel < 0.54F) {
        // Channel 3: Li-7 + alpha + p
        const auto ke_u = e_per_u * (0.88F - 0.08F * u_energy);
        set.products[0] = make_product(3, 7, ke_u, 1.0F, false);
        set.products[1] = make_product(2, 4, e_per_u * 0.78F, 1.1F, true);
        set.products[2] = make_product(1, 1, e_per_u * 0.72F, 1.2F, true);
        set.count = 3;
        set.local_deposit_MeV = 22.0F;
    } else if (u_channel < 0.72F) {
        // Channel 4: Triple alpha (3 x He-4)
        const auto ke_u = e_per_u * (0.94F - 0.06F * u_energy);
        set.products[0] = make_product(2, 4, ke_u, 1.0F, true);
        set.products[1] = make_product(2, 4, ke_u * (0.90F + 0.08F * u_sub), 1.1F, true);
        set.products[2] = make_product(2, 4, ke_u * (0.85F + 0.10F * u_sub), 1.2F, true);
        set.count = 3;
        set.local_deposit_MeV = 7.3F;
    } else if (u_channel < 0.88F) {
        // Channel 5: Alpha + 2 protons + (2n)
        const auto ke_u = e_per_u * (0.86F - 0.08F * u_energy);
        set.products[0] = make_product(2, 4, ke_u, 1.1F, true);
        set.products[1] = make_product(1, 1, e_per_u * 0.75F, 1.2F, true);
        set.products[2] = make_product(1, 1, e_per_u * (0.70F + 0.15F * u_sub), 1.3F, true);
        set.count = 3;
        set.local_deposit_MeV = 25.0F;
    } else {
        // Channel 6: Protons (4 x p + 4n)
        const auto ke_u = e_per_u * (0.80F - 0.10F * u_energy);
        set.products[0] = make_product(1, 1, ke_u, 1.2F, true);
        set.products[1] = make_product(1, 1, ke_u * 0.92F, 1.2F, true);
        set.products[2] = make_product(1, 1, ke_u * 0.85F, 1.3F, true);
        set.products[3] = make_product(1, 1, ke_u * 0.80F, 1.4F, true);
        set.count = 4;
        set.local_deposit_MeV = 30.0F;
    }

    return set;
}
