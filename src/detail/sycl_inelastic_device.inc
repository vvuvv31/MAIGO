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

    auto make_product = [&](int16_t z, int16_t a, float ke_per_u, float angle_scale) {
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
        const auto theta = sigma_th * sycl::sqrt(-2.0F * sycl::log(sycl::fmax(u_theta, 1.0e-10F)));
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

    if (u_channel < 0.18F) {
        const auto ke_u = e_per_u * (0.96F - 0.04F * u_energy);
        set.products[0] = make_product(5, 11, ke_u, 1.0F);
        set.products[1] = make_product(1, 1, e_per_u * (0.80F + 0.15F * u_sub), 1.0F);
        set.count = 2;
        set.local_deposit_MeV = 16.0F;
    } else if (u_channel < 0.32F) {
        const auto ke_u = e_per_u * (0.94F - 0.05F * u_energy);
        set.products[0] = make_product(5, 10, ke_u, 1.0F);
        if (u_sub < 0.5F) {
            set.products[1] = make_product(1, 2, e_per_u * 0.75F, 1.0F);
            set.count = 2;
        } else {
            set.products[1] = make_product(1, 1, e_per_u * 0.80F, 1.0F);
            set.count = 2;
        }
        set.local_deposit_MeV = 25.0F;
    } else if (u_channel < 0.44F) {
        const auto ke_u = e_per_u * (0.93F - 0.06F * u_energy);
        set.products[0] = make_product(4, 9, ke_u, 1.0F);
        set.products[1] = make_product(2, 4, e_per_u * (0.85F + 0.10F * u_sub), 1.0F);
        set.count = 2;
        set.local_deposit_MeV = 26.0F;
    } else if (u_channel < 0.54F) {
        const auto ke_u = e_per_u * (0.90F - 0.08F * u_energy);
        set.products[0] = make_product(3, 7, ke_u, 1.0F);
        set.products[1] = make_product(2, 4, e_per_u * 0.82F, 1.0F);
        set.products[2] = make_product(1, 1, e_per_u * 0.78F, 1.0F);
        set.count = 3;
        set.local_deposit_MeV = 31.0F;
    } else if (u_channel < 0.72F) {
        const auto ke_u = e_per_u * (0.95F - 0.06F * u_energy);
        set.products[0] = make_product(2, 4, ke_u, 1.0F);
        set.products[1] = make_product(2, 4, ke_u * (0.92F + 0.08F * u_sub), 1.0F);
        set.products[2] = make_product(2, 4, ke_u * (0.88F + 0.10F * u_sub), 1.0F);
        set.count = 3;
        set.local_deposit_MeV = 8.0F;
    } else if (u_channel < 0.88F) {
        const auto ke_u = e_per_u * (0.88F - 0.08F * u_energy);
        set.products[0] = make_product(2, 4, ke_u, 1.0F);
        set.products[1] = make_product(1, 1, e_per_u * 0.80F, 1.0F);
        set.products[2] = make_product(1, 1, e_per_u * (0.75F + 0.15F * u_sub), 1.0F);
        set.count = 3;
        set.local_deposit_MeV = 35.0F;
    } else {
        const auto ke_u = e_per_u * (0.85F - 0.10F * u_energy);
        set.products[0] = make_product(1, 1, ke_u, 1.0F);
        set.products[1] = make_product(1, 1, ke_u * 0.95F, 1.0F);
        set.products[2] = make_product(1, 1, ke_u * 0.90F, 1.0F);
        set.products[3] = make_product(1, 1, ke_u * 0.85F, 1.0F);
        set.count = 4;
        set.local_deposit_MeV = 45.0F;
    }

    return set;
}
