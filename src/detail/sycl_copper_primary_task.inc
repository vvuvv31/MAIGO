// Primary copper reaction task used by the production child queue.
struct CopperPrimaryChildTask {
    std::uint64_t global_history{},rng_history{},spot_seed{},beamline_step{};
    std::uint32_t flat_product_index{},product_index{};
    float x{},y{},z{},dx{},dy{},dz{},event_cos{},event_sin{};
};
static_assert(sizeof(CopperPrimaryChildTask)==72);
