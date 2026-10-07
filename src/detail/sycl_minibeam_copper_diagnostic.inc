// Included inside carbon's anonymous namespace. Diagnostic only: no RNG draws.
struct CopperPrimaryChildTask {
    std::uint64_t global_history{},rng_history{},spot_seed{},beamline_step{};
    std::uint32_t flat_product_index{},product_index{};
    float x{},y{},z{},dx{},dy{},dz{},event_cos{},event_sin{};
};
static_assert(sizeof(CopperPrimaryChildTask)==72);
struct CopperHistoryDiagnostic {
    float entry_x{}, entry_y{}, entry_z{}, entry_energy{};
    float entry_dx{}, entry_dy{}, entry_dz{}, wall_distance{};
    float terminal_x{}, terminal_y{}, terminal_z{}, terminal_energy{};
    std::uint32_t copper_proposals{}, copper_elastic{}, copper_inelastic{}, loop_iterations{};
    // 0 unrecorded, 1 stopped, 2 inelastic, 3 backward, 4 iteration cap,
    // 5 downstream face, 6 other. Cohort0 is entrance slit/air, 1-4 copper.
    std::uint32_t cohort{}, outcome{};
};
static_assert(sizeof(CopperHistoryDiagnostic) == 72);

inline unsigned copper_diagnostic_cohort(float distance) {
    return distance <= 0 ? 0U : distance < .25F ? 1U :
           distance < .5F ? 2U : distance < 1.F ? 3U : 4U;
}

inline std::vector<std::uint8_t> read_copper_history_mask(
    const std::string& path, std::uint64_t histories, std::uint64_t seed) {
    if (path.empty()) return {};
    std::ifstream in(path, std::ios::binary);
    char magic[8]{};
    std::uint64_t count{}, mask_seed{};
    in.read(magic, 8);
    in.read(reinterpret_cast<char*>(&count), 8);
    in.read(reinterpret_cast<char*>(&mask_seed), 8);
    if (!in || std::string_view(magic,8) != "MCUCOH01" ||
        count != histories || mask_seed != seed)
        throw std::invalid_argument("Copper diagnostic mask header/history count/seed mismatch");
    std::vector<std::uint8_t> mask(histories);
    in.read(reinterpret_cast<char*>(mask.data()), mask.size());
    if (!in || in.peek() != std::char_traits<char>::eof() ||
        std::any_of(mask.begin(), mask.end(), [](auto value) { return value > 1; }) ||
        std::none_of(mask.begin(), mask.end(), [](auto value) { return value == 1; }))
        throw std::invalid_argument("Copper diagnostic mask requires exactly N bytes of 0/1 and a nonempty cohort");
    return mask;
}

inline void write_copper_history_diagnostic(
    const std::string& path, const std::vector<CopperHistoryDiagnostic>& records,
    const std::vector<float>& survivors, const std::vector<float>& deposited,
    const std::vector<std::uint32_t>& steps,
    const std::vector<std::uint8_t>& mask) {
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot open copper diagnostic output: " + path);
    out.precision(9);
    out << "history,selected,cohort,entry_x_mm,entry_y_mm,entry_z_mm,entry_energy_MeV,"
           "entry_dx,entry_dy,entry_dz,wall_distance_mm,copper_proposals,copper_elastic,"
           "copper_inelastic,loop_iterations,outcome,terminal_x_mm,terminal_y_mm,"
           "terminal_z_mm,terminal_energy_MeV,water_primary_energy_MeV,"
           "water_primary_steps,history_deposited_MeV\n";
    for (std::size_t h=0;h<records.size();++h) {
        const auto& r=records[h];
        out << h << ',' << (mask.empty() || mask[h] ? 1 : 0) << ',' << r.cohort << ','
            << r.entry_x << ',' << r.entry_y << ',' << r.entry_z << ',' << r.entry_energy << ','
            << r.entry_dx << ',' << r.entry_dy << ',' << r.entry_dz << ',' << r.wall_distance << ','
            << r.copper_proposals << ',' << r.copper_elastic << ',' << r.copper_inelastic << ','
            << r.loop_iterations << ',' << r.outcome << ','
            << r.terminal_x << ',' << r.terminal_y << ',' << r.terminal_z << ',' << r.terminal_energy << ','
            << survivors[h] << ',' << steps[h] << ',' << deposited[h] << '\n';
    }
    if (!out) throw std::runtime_error("Copper diagnostic output write failed: " + path);
}

template<class Record>
void write_copper_capture_records(const std::filesystem::path& path,
    const std::vector<Record>& records) {
    static_assert(std::is_trivially_copyable_v<Record>);
    std::ofstream out(path,std::ios::binary);
    const std::uint64_t size=sizeof(Record),count=records.size();
    out.write("MCUCAP01",8);
    out.write(reinterpret_cast<const char*>(&size),8);
    out.write(reinterpret_cast<const char*>(&count),8);
    out.write(reinterpret_cast<const char*>(records.data()),records.size()*sizeof(Record));
    if(!out)throw std::runtime_error("Copper response capture write failed: "+path.string());
}
