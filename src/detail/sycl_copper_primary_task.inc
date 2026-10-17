// Primary copper reaction task used by the production child queue.
// Positions are mm, directions are unit vectors, and rotation coefficients
// map sampled products to the parent event frame. RNG identity belongs to the
// source history/event/product, never the atomically allocated queue slot.
struct CopperPrimaryChildTask {
    std::uint64_t global_history{}, rng_history{}, spot_seed{}, beamline_step{};
    std::uint32_t flat_product_index{}, product_index{};
    float x{}, y{}, z{}, dx{}, dy{}, dz{}, event_cos{}, event_sin{};
};
static_assert(sizeof(CopperPrimaryChildTask) == 72);
