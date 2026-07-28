// Shared SyclTransportContext::Impl definition for dual SYCL TUs.
// Included by transport_sycl_legacy.cpp and transport_sycl.cpp.
// Method definitions are emitted only when CARBON_DEFINE_SYCL_CONTEXT=1.

struct SyclTransportContext::Impl {
    explicit Impl(const std::string& requested_device_name)
        : device_name(requested_device_name), queue(make_sycl_queue(requested_device_name)) {}

    ~Impl() { clear(); }

    void clear() noexcept {
        const auto release = [this](auto*& pointer) {
            if (pointer != nullptr) {
                sycl::free(pointer, queue);
                pointer = nullptr;
            }
        };
        release(table_device);
        release(cross_section_device);
        release(reaction_bins_device);
        release(reactions_device);
        release(reaction_secondaries_device);
        release(cascade_projectiles_device);
        release(cascade_cross_sections_device);
        release(cascade_interactions_device);
        release(cascade_products_device);
        release(neutral_projectiles_device);
        release(neutral_cross_sections_device);
        release(neutral_interactions_device);
        release(neutral_products_device);
        initialized = false;
    }

    void ensure_initialized(const StoppingPowerTable& stopping_power,
                            const CrossSectionTable& cross_section,
                            const ReactionPackageTable* reaction_packages,
                            const CascadePackageTable* cascade_packages,
                            const NeutralPackageTable* neutral_packages) {
        if (initialized) {
            if (stopping_power_host != &stopping_power ||
                cross_section_host != &cross_section || reaction_packages_host != reaction_packages ||
                cascade_packages_host != cascade_packages ||
                neutral_packages_host != neutral_packages) {
                throw std::invalid_argument(
                    "SyclTransportContext cannot be reused with different physics tables");
            }
            return;
        }

        stopping_power_host = &stopping_power;
        cross_section_host = &cross_section;
        reaction_packages_host = reaction_packages;
        cascade_packages_host = cascade_packages;
        neutral_packages_host = neutral_packages;

        try {
            table_device = sycl::malloc_device<float>(stopping_power.values().size(), queue);
            cross_section_device =
                sycl::malloc_device<float>(cross_section.values().size(), queue);
            if (reaction_packages != nullptr) {
                reaction_bins_device = sycl::malloc_device<ReactionEnergyBin>(
                    reaction_packages->energy_bins().size(), queue);
                reactions_device = sycl::malloc_device<ReactionPackage>(
                    reaction_packages->reactions().size(), queue);
                reaction_secondaries_device = sycl::malloc_device<ReactionSecondary>(
                    reaction_packages->secondaries().size(), queue);
            }
            if (cascade_packages != nullptr) {
                cascade_projectiles_device = sycl::malloc_device<CascadeProjectile>(
                    cascade_packages->projectiles().size(), queue);
                cascade_cross_sections_device = sycl::malloc_device<CascadeCrossSectionSample>(
                    cascade_packages->cross_sections().size(), queue);
                cascade_interactions_device = sycl::malloc_device<CascadeInteraction>(
                    cascade_packages->interactions().size(), queue);
                cascade_products_device = sycl::malloc_device<ReactionSecondary>(
                    cascade_packages->products().size(), queue);
            }
            if (neutral_packages != nullptr) {
                neutral_projectiles_device = sycl::malloc_device<NeutralProjectile>(
                    neutral_packages->projectiles().size(), queue);
                neutral_cross_sections_device = sycl::malloc_device<NeutralCrossSectionSample>(
                    neutral_packages->cross_sections().size(), queue);
                neutral_interactions_device = sycl::malloc_device<NeutralInteraction>(
                    neutral_packages->interactions().size(), queue);
                neutral_products_device = sycl::malloc_device<ReactionSecondary>(
                    neutral_packages->products().size(), queue);
            }

            const auto allocation_failed =
                table_device == nullptr || cross_section_device == nullptr ||
                (reaction_packages != nullptr &&
                 (reaction_bins_device == nullptr || reactions_device == nullptr ||
                  reaction_secondaries_device == nullptr)) ||
                (cascade_packages != nullptr &&
                 (cascade_projectiles_device == nullptr ||
                  cascade_cross_sections_device == nullptr ||
                  cascade_interactions_device == nullptr || cascade_products_device == nullptr)) ||
                (neutral_packages != nullptr &&
                 (neutral_projectiles_device == nullptr ||
                  neutral_cross_sections_device == nullptr ||
                  neutral_interactions_device == nullptr || neutral_products_device == nullptr));
            if (allocation_failed) {
                throw std::bad_alloc();
            }

            std::vector<float> table_host(stopping_power.values().size());
            std::transform(stopping_power.values().begin(), stopping_power.values().end(),
                           table_host.begin(),
                           [](double value) { return static_cast<float>(value); });
            std::vector<float> xs_host(cross_section.values().size());
            std::transform(cross_section.values().begin(), cross_section.values().end(),
                           xs_host.begin(),
                           [](double value) { return static_cast<float>(value); });
            queue.copy(table_host.data(), table_device, table_host.size());
            queue.copy(xs_host.data(), cross_section_device, xs_host.size());
            if (reaction_packages != nullptr) {
                queue.copy(reaction_packages->energy_bins().data(), reaction_bins_device,
                           reaction_packages->energy_bins().size());
                queue.copy(reaction_packages->reactions().data(), reactions_device,
                           reaction_packages->reactions().size());
                queue.copy(reaction_packages->secondaries().data(), reaction_secondaries_device,
                           reaction_packages->secondaries().size());
            }
            if (cascade_packages != nullptr) {
                queue.copy(cascade_packages->projectiles().data(), cascade_projectiles_device,
                           cascade_packages->projectiles().size());
                queue.copy(cascade_packages->cross_sections().data(),
                           cascade_cross_sections_device,
                           cascade_packages->cross_sections().size());
                queue.copy(cascade_packages->interactions().data(), cascade_interactions_device,
                           cascade_packages->interactions().size());
                queue.copy(cascade_packages->products().data(), cascade_products_device,
                           cascade_packages->products().size());
            }
            if (neutral_packages != nullptr) {
                queue.copy(neutral_packages->projectiles().data(), neutral_projectiles_device,
                           neutral_packages->projectiles().size());
                queue.copy(neutral_packages->cross_sections().data(),
                           neutral_cross_sections_device,
                           neutral_packages->cross_sections().size());
                queue.copy(neutral_packages->interactions().data(), neutral_interactions_device,
                           neutral_packages->interactions().size());
                queue.copy(neutral_packages->products().data(), neutral_products_device,
                           neutral_packages->products().size());
            }
            queue.wait_and_throw();
            initialized = true;
        } catch (...) {
            clear();
            throw;
        }
    }

    std::string device_name;
    sycl::queue queue;
    bool initialized{false};
    const StoppingPowerTable* stopping_power_host{nullptr};
    const CrossSectionTable* cross_section_host{nullptr};
    const ReactionPackageTable* reaction_packages_host{nullptr};
    const CascadePackageTable* cascade_packages_host{nullptr};
    const NeutralPackageTable* neutral_packages_host{nullptr};
    float* table_device{nullptr};
    float* cross_section_device{nullptr};
    ReactionEnergyBin* reaction_bins_device{nullptr};
    ReactionPackage* reactions_device{nullptr};
    ReactionSecondary* reaction_secondaries_device{nullptr};
    CascadeProjectile* cascade_projectiles_device{nullptr};
    CascadeCrossSectionSample* cascade_cross_sections_device{nullptr};
    CascadeInteraction* cascade_interactions_device{nullptr};
    ReactionSecondary* cascade_products_device{nullptr};
    NeutralProjectile* neutral_projectiles_device{nullptr};
    NeutralCrossSectionSample* neutral_cross_sections_device{nullptr};
    NeutralInteraction* neutral_interactions_device{nullptr};
    ReactionSecondary* neutral_products_device{nullptr};
};
