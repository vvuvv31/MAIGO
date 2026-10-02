// Conditional Cu response research ABI. Charged particles are at water entry; neutrons are Cu births.
#if defined(CARBON_ENABLE_MINIBEAM)
inline std::string copper_response_signature(const TransportConfig& config) {
    std::ostringstream out; out.precision(17);
    out << "MCURESP01\n";
    out << "beam_energy_spread=" << config.beam_energy_spread << "\n";
    out << "primary_atomic_number=" << config.primary_atomic_number << "\n";
    out << "primary_mass_number=" << config.primary_mass_number << "\n";
    out << "minibeam_copper_production_cut_mm=" << config.minibeam_copper_production_cut_mm << "\n";
    out << "minibeam_water_production_cut_mm=" << config.minibeam_water_production_cut_mm << "\n";
    out << "phantom_length_mm=" << config.phantom_length_mm << "\n";
    out << "em_model=" << config.em_model << "\n";
    out << "em_package_sha256=" << config.em_package_sha256 << "\n";
    out << "em_delta_moments_sha256=" << config.em_delta_moments_sha256 << "\n";
    out << "maximum_step_mm=" << config.maximum_step_mm << "\n";
    out << "maximum_relative_energy_loss=" << config.maximum_relative_energy_loss << "\n";
    out << "energy_cutoff_MeV=" << config.energy_cutoff_MeV << "\n";
    out << "water_density_g_per_cm3=" << config.water_density_g_per_cm3 << "\n";
    out << "all_ion_elastic_sha256=" << config.all_ion_elastic_sha256 << "\n";
    out << "ct_schneider_physics_bundle_file=" << config.ct_schneider_physics_bundle_file << "\n";
    out << "secondary_out_of_scope_nuclear_policy=" << config.secondary_out_of_scope_nuclear_policy << "\n";
    out << "voxel_bins_x=" << config.voxel_bins_x << "\n";
    out << "voxel_bins_y=" << config.voxel_bins_y << "\n";
    out << "voxel_bins_z=" << config.voxel_bins_z << "\n";
    out << "voxel_size_x_mm=" << config.voxel_size_x_mm << "\n";
    out << "voxel_size_y_mm=" << config.voxel_size_y_mm << "\n";
    out << "voxel_size_z_mm=" << config.voxel_size_z_mm << "\n";
    out << "enable_energy_straggling=" << config.enable_energy_straggling << "\n";
    out << "enable_secondary_energy_straggling=" << config.enable_secondary_energy_straggling << "\n";
    out << "enable_inelastic=" << config.enable_inelastic << "\n";
    out << "cinel02_max_secondary_inelastic_generations=" << config.cinel02_max_secondary_inelastic_generations << "\n";
    out << "enable_secondary_transport=" << config.enable_secondary_transport << "\n";
    out << "multiple_scattering_model=" << config.multiple_scattering_model << "\n";
    out << "urban_mcs_package_sha256=" << config.urban_mcs_package_sha256 << "\n";
    out << "minibeam_water_proton_helium_mcs_model=" << config.minibeam_water_proton_helium_mcs_model << "\n";
    out << "enable_flat_source=" << config.enable_flat_source << "\n";
    out << "flat_source_half_width_x_mm=" << config.flat_source_half_width_x_mm << "\n";
    out << "flat_source_half_width_y_mm=" << config.flat_source_half_width_y_mm << "\n";
    out << "enable_emittance_source=" << config.enable_emittance_source << "\n";
    out << "emittance_sigma_x_mm=" << config.emittance_sigma_x_mm << "\n";
    out << "emittance_sigma_y_mm=" << config.emittance_sigma_y_mm << "\n";
    out << "emittance_sigma_x_prime=" << config.emittance_sigma_x_prime << "\n";
    out << "emittance_sigma_y_prime=" << config.emittance_sigma_y_prime << "\n";
    out << "emittance_correlation_x=" << config.emittance_correlation_x << "\n";
    out << "emittance_correlation_y=" << config.emittance_correlation_y << "\n";
    out << "source_origin_x_mm=" << config.source_origin_x_mm << "\n";
    out << "source_origin_y_mm=" << config.source_origin_y_mm << "\n";
    out << "source_origin_z_mm=" << config.source_origin_z_mm << "\n";
    out << "beam_ux_x=" << config.beam_ux_x << "\n";
    out << "beam_uy_x=" << config.beam_uy_x << "\n";
    out << "beam_uz_x=" << config.beam_uz_x << "\n";
    out << "minibeam_transport_mode=" << config.minibeam_transport_mode << "\n";
    out << "minibeam_material=" << config.minibeam_material << "\n";
    out << "minibeam_collimator_center_to_isocenter_mm=" << config.minibeam_collimator_center_to_isocenter_mm << "\n";
    out << "minibeam_collimator_width_mm=" << config.minibeam_collimator_width_mm << "\n";
    out << "minibeam_collimator_length_mm=" << config.minibeam_collimator_length_mm << "\n";
    out << "minibeam_collimator_thickness_mm=" << config.minibeam_collimator_thickness_mm << "\n";
    out << "minibeam_slit_length_mm=" << config.minibeam_slit_length_mm << "\n";
    out << "minibeam_slit_thickness_mm=" << config.minibeam_slit_thickness_mm << "\n";
    out << "minibeam_radius_mm=" << config.minibeam_radius_mm << "\n";
    out << "minibeam_thickness_mm=" << config.minibeam_thickness_mm << "\n";
    out << "minibeam_exit_to_phantom_mm=" << config.minibeam_exit_to_phantom_mm << "\n";
    out << "minibeam_slit_count=" << config.minibeam_slit_count << "\n";
    out << "minibeam_slit_width_mm=" << config.minibeam_slit_width_mm << "\n";
    out << "minibeam_slit_pitch_mm=" << config.minibeam_slit_pitch_mm << "\n";
    out << "minibeam_slit_half_length_mm=" << config.minibeam_slit_half_length_mm << "\n";
    out << "minibeam_slit_offset_mm=" << config.minibeam_slit_offset_mm << "\n";
    out << "minibeam_collimator_angle_deg=" << config.minibeam_collimator_angle_deg << "\n";
    out << "minibeam_copper_stopping_power_file=" << config.minibeam_copper_stopping_power_file << "\n";
    out << "minibeam_air_stopping_power_file=" << config.minibeam_air_stopping_power_file << "\n";
    out << "minibeam_air_ionization_rate_file=" << config.minibeam_air_ionization_rate_file << "\n";
    out << "minibeam_air_ionization_rate_sha256=" << config.minibeam_air_ionization_rate_sha256 << "\n";
    out << "minibeam_air_mcs_model=" << config.minibeam_air_mcs_model << "\n";
    out << "minibeam_air_slit_max_step_mm=" << config.minibeam_air_slit_max_step_mm << "\n";
    out << "minibeam_air_urban_package_file=" << config.minibeam_air_urban_package_file << "\n";
    out << "minibeam_air_urban_package_sha256=" << config.minibeam_air_urban_package_sha256 << "\n";
    out << "minibeam_copper_loss_range_file=" << config.minibeam_copper_loss_range_file << "\n";
    out << "minibeam_copper_density_g_per_cm3=" << config.minibeam_copper_density_g_per_cm3 << "\n";
    out << "minibeam_copper_radiation_length_g_per_cm2=" << config.minibeam_copper_radiation_length_g_per_cm2 << "\n";
    out << "minibeam_copper_max_step_mm=" << config.minibeam_copper_max_step_mm << "\n";
    out << "minibeam_copper_exact_material_boundaries=" << config.minibeam_copper_exact_material_boundaries << "\n";
    out << "minibeam_copper_enable_mcs=" << config.minibeam_copper_enable_mcs << "\n";
    out << "minibeam_copper_mcs_model=" << config.minibeam_copper_mcs_model << "\n";
    out << "minibeam_copper_enable_elastic=" << config.minibeam_copper_enable_elastic << "\n";
    out << "minibeam_copper_enable_energy_straggling=" << config.minibeam_copper_enable_energy_straggling << "\n";
    out << "minibeam_copper_straggling_scale=" << config.minibeam_copper_straggling_scale << "\n";
    out << "minibeam_copper_proton_straggling_model=" << config.minibeam_copper_proton_straggling_model << "\n";
    out << "minibeam_copper_primary_native_mean_loss=" << config.minibeam_copper_primary_native_mean_loss << "\n";
    out << "minibeam_copper_fragment_enable_energy_straggling=" << config.minibeam_copper_fragment_enable_energy_straggling << "\n";
    out << "minibeam_copper_fragment_straggling_scale=" << config.minibeam_copper_fragment_straggling_scale << "\n";
    out << "minibeam_copper_mcs_scale=" << config.minibeam_copper_mcs_scale << "\n";
    out << "minibeam_copper_fragment_mcs_scale=" << config.minibeam_copper_fragment_mcs_scale << "\n";
    out << "minibeam_copper_fragment_cascade_generations=" << config.minibeam_copper_fragment_cascade_generations << "\n";
    out << "minibeam_copper_survivor_energy_loss_scale=" << config.minibeam_copper_survivor_energy_loss_scale << "\n";
    out << "minibeam_water_primary_stopping_power_scale=" << config.minibeam_water_primary_stopping_power_scale << "\n";
    out << "minibeam_water_low_energy_mcs_transition_MeVu=" << config.minibeam_water_low_energy_mcs_transition_MeVu << "\n";
    out << "minibeam_water_primary_low_energy_mcs_scale=" << config.minibeam_water_primary_low_energy_mcs_scale << "\n";
    out << "minibeam_water_fragment_low_energy_mcs_scale=" << config.minibeam_water_fragment_low_energy_mcs_scale << "\n";
    out << "minibeam_water_primary_mcs_model=" << config.minibeam_water_primary_mcs_model << "\n";
    out << "minibeam_water_primary_mcs_max_segment_mm=" << config.minibeam_water_primary_mcs_max_segment_mm << "\n";
    out << "minibeam_water_primary_urban_max_step_mm=" << config.minibeam_water_primary_urban_max_step_mm << "\n";
    out << "minibeam_water_delta_response_model=" << config.minibeam_water_delta_response_model << "\n";
    out << "minibeam_water_delta_response_table_file=" << config.minibeam_water_delta_response_table_file << "\n";
    out << "minibeam_water_urban_loss_range_file=" << config.minibeam_water_urban_loss_range_file << "\n";
    out << "minibeam_water_secondary_c12_mcs_model=" << config.minibeam_water_secondary_c12_mcs_model << "\n";
    out << "minibeam_water_secondary_c12_mcs_max_segment_mm=" << config.minibeam_water_secondary_c12_mcs_max_segment_mm << "\n";
    out << "minibeam_water_secondary_c12_post_sample_loss_scale=" << config.minibeam_water_secondary_c12_post_sample_loss_scale << "\n";
    out << "minibeam_water_primary_mcs_tail_strength=" << config.minibeam_water_primary_mcs_tail_strength << "\n";
    out << "minibeam_water_primary_mcs_tail_width=" << config.minibeam_water_primary_mcs_tail_width << "\n";
    out << "minibeam_water_touched_primary_surface_boost=" << config.minibeam_water_touched_primary_surface_boost << "\n";
    out << "minibeam_water_touched_primary_surface_sigma_mm=" << config.minibeam_water_touched_primary_surface_sigma_mm << "\n";
    out << "minibeam_water_touched_primary_deficit=" << config.minibeam_water_touched_primary_deficit << "\n";
    out << "minibeam_water_touched_primary_deficit_center_mm=" << config.minibeam_water_touched_primary_deficit_center_mm << "\n";
    out << "minibeam_water_touched_primary_deficit_sigma_mm=" << config.minibeam_water_touched_primary_deficit_sigma_mm << "\n";
    out << "minibeam_copper_enable_nuclear_attenuation=" << config.minibeam_copper_enable_nuclear_attenuation << "\n";
    out << "minibeam_copper_cross_section_file=" << config.minibeam_copper_cross_section_file << "\n";
    out << "minibeam_copper_inclxx_file=" << config.minibeam_copper_inclxx_file << "\n";
    out << "minibeam_copper_elastic_file=" << config.minibeam_copper_elastic_file << "\n";
    out << "minibeam_copper_ion_stopping_power_file=" << config.minibeam_copper_ion_stopping_power_file << "\n";
    out << "minibeam_copper_ion_cross_section_file=" << config.minibeam_copper_ion_cross_section_file << "\n";
    out << "minibeam_copper_neutral_cross_section_file=" << config.minibeam_copper_neutral_cross_section_file << "\n";
    out << "minibeam_water_entrance_world_y_mm=" << config.minibeam_water_entrance_world_y_mm << "\n";
    out << "enable_neutron_transport=" << config.enable_neutron_transport << "\n";
    out << "neutron_world_half_width_mm=" << config.neutron_world_half_width_mm << "\n";
    out << "neutron_max_collisions=" << config.neutron_max_collisions << "\n";
    out << "neutron_elastic_two_body=" << config.neutron_elastic_two_body << "\n";
    out << "neutron_water_recoil_urban=" << config.neutron_water_recoil_urban << "\n";
    out << "neutron_exact_plane_boundaries=" << config.neutron_exact_plane_boundaries << "\n";
    out << "minibeam_electronic_buildup_primary_only=" << config.minibeam_electronic_buildup_primary_only << "\n";
    out << "minibeam_phase_space_at_copper_exit=" << config.minibeam_phase_space_at_copper_exit << "\n";
    out << "minibeam_water_entry_secondary_replay_allow_primary_c12=" << config.minibeam_water_entry_secondary_replay_allow_primary_c12 << "\n";
    out << "minibeam_water_entry_secondary_replay_allow_internal_births=" << config.minibeam_water_entry_secondary_replay_allow_internal_births << "\n";
    if(!config.ct_schneider_physics_bundle_file.empty()) out << "sha256:ct_schneider_physics_bundle_file=" << compute_file_sha256_hex(config.ct_schneider_physics_bundle_file) << "\n";
    if(!config.minibeam_copper_stopping_power_file.empty()) out << "sha256:minibeam_copper_stopping_power_file=" << compute_file_sha256_hex(config.minibeam_copper_stopping_power_file) << "\n";
    if(!config.minibeam_air_stopping_power_file.empty()) out << "sha256:minibeam_air_stopping_power_file=" << compute_file_sha256_hex(config.minibeam_air_stopping_power_file) << "\n";
    if(!config.minibeam_air_ionization_rate_file.empty()) out << "sha256:minibeam_air_ionization_rate_file=" << compute_file_sha256_hex(config.minibeam_air_ionization_rate_file) << "\n";
    if(!config.minibeam_air_urban_package_file.empty()) out << "sha256:minibeam_air_urban_package_file=" << compute_file_sha256_hex(config.minibeam_air_urban_package_file) << "\n";
    if(!config.minibeam_copper_loss_range_file.empty()) out << "sha256:minibeam_copper_loss_range_file=" << compute_file_sha256_hex(config.minibeam_copper_loss_range_file) << "\n";
    if(!config.minibeam_water_delta_response_table_file.empty()) out << "sha256:minibeam_water_delta_response_table_file=" << compute_file_sha256_hex(config.minibeam_water_delta_response_table_file) << "\n";
    if(!config.minibeam_water_urban_loss_range_file.empty()) out << "sha256:minibeam_water_urban_loss_range_file=" << compute_file_sha256_hex(config.minibeam_water_urban_loss_range_file) << "\n";
    if(!config.minibeam_copper_cross_section_file.empty()) out << "sha256:minibeam_copper_cross_section_file=" << compute_file_sha256_hex(config.minibeam_copper_cross_section_file) << "\n";
    if(!config.minibeam_copper_inclxx_file.empty()) out << "sha256:minibeam_copper_inclxx_file=" << compute_file_sha256_hex(config.minibeam_copper_inclxx_file) << "\n";
    if(!config.minibeam_copper_elastic_file.empty()) out << "sha256:minibeam_copper_elastic_file=" << compute_file_sha256_hex(config.minibeam_copper_elastic_file) << "\n";
    if(!config.minibeam_copper_ion_stopping_power_file.empty()) out << "sha256:minibeam_copper_ion_stopping_power_file=" << compute_file_sha256_hex(config.minibeam_copper_ion_stopping_power_file) << "\n";
    if(!config.minibeam_copper_ion_cross_section_file.empty()) out << "sha256:minibeam_copper_ion_cross_section_file=" << compute_file_sha256_hex(config.minibeam_copper_ion_cross_section_file) << "\n";
    if(!config.minibeam_copper_neutral_cross_section_file.empty()) out << "sha256:minibeam_copper_neutral_cross_section_file=" << compute_file_sha256_hex(config.minibeam_copper_neutral_cross_section_file) << "\n";
    return out.str();
}

struct CopperResponseEvent {
    PrimaryMediumEntryState primary{};
    std::uint32_t charged_begin{},charged_count{},neutron_begin{},neutron_count{};
    float removed{},air_loss{},input_x{},input_y{};
};
static_assert(sizeof(CopperResponseEvent)==96);

template<class Record>
std::vector<Record> read_copper_response_records(const std::filesystem::path& path) {
    static_assert(std::is_trivially_copyable_v<Record>);
    std::ifstream in(path,std::ios::binary);
    char magic[8]{};std::uint64_t size{},count{};
    in.read(magic,8);in.read(reinterpret_cast<char*>(&size),8);in.read(reinterpret_cast<char*>(&count),8);
    const auto bytes=std::filesystem::file_size(path);
    if(!in || std::string_view(magic,8)!="MCUCAP01" || size!=sizeof(Record) ||
        bytes<24 || count!=(bytes-24)/sizeof(Record) || (bytes-24)%sizeof(Record))
        throw std::invalid_argument("Invalid Cu response native record header: "+path.string());
    std::vector<Record> records(count);
    in.read(reinterpret_cast<char*>(records.data()),count*sizeof(Record));
    if(!in)throw std::runtime_error("Truncated Cu response: "+path.string());
    return records;
}
struct CopperResponseBank {
    std::vector<CopperResponseEvent> events;
    std::vector<SecondaryParticle> charged;
    std::vector<NeutronBirthRecord> neutrons;
    std::vector<std::uint32_t> offsets;
};

inline CopperResponseBank load_copper_response(const std::filesystem::path& root,
    const std::string& expected_sha256,const TransportConfig& config) {
    const auto valid_hash=[](const std::string& s) {
        return s.size()==64 && std::all_of(s.begin(),s.end(),[](char c){
            return (c>='0' && c<='9') || (c>='a' && c<='f');});
    };
    if(config.run_mode!=RunMode::research || !config.enable_minibeam || config.enable_ct_grid ||
        config.enable_tps_source || config.enable_tps_coordinate_system ||
        config.primary_atomic_number!=1 || config.primary_mass_number!=1 ||
        config.initial_total_energy_MeV()!=170 || config.beam_energy_spread!=0 ||
        !config.enable_flat_source || config.enable_emittance_source ||
        config.source_origin_x_mm!=0 || config.source_origin_y_mm!=0 ||
        config.flat_source_half_width_x_mm!=15 || config.flat_source_half_width_y_mm!=15 ||
        config.minibeam_collimator_angle_deg!=0 || config.minibeam_primary_pipeline!="compact" ||
        !config.minibeam_primary_specialization || config.multiple_scattering_model!="urban_v2" ||
        !config.enable_secondary_transport || !config.enable_neutron_transport ||
        !config.minibeam_water_entry_secondary_replay_file.empty() ||
        !config.minibeam_phase_space_output_file.empty() ||
        !config.minibeam_fragment_phase_space_output_file.empty() ||
        !config.minibeam_water_primary_plane_output_file.empty())
        throw std::invalid_argument("Conditional Cu response domain is fixed170MeV flat30x30 proton compact Urban water; snapshots/TPS/CT/replay forbidden");
    if(!valid_hash(expected_sha256) || !file_sha256_matches(root/"manifest.txt",expected_sha256))
        throw std::invalid_argument("Conditional Cu response manifest SHA256 mismatch");
    const std::array<std::string,5> names={"signature.txt","events.bin","charged.bin","neutrons.bin","offsets.bin"};
    std::ifstream manifest(root/"manifest.txt");std::string schema;std::getline(manifest,schema);
    if(schema!="MCURESP01")throw std::invalid_argument("Conditional Cu response schema mismatch");
    for(const auto& name:names) {
        std::string file,sha;manifest>>file>>sha;
        if(file!=name || !valid_hash(sha) || !file_sha256_matches(root/name,sha))
            throw std::invalid_argument("Conditional Cu response record SHA256 mismatch: "+name);
    }
    std::string extra;if(manifest>>extra)throw std::invalid_argument("Unexpected Cu response manifest field");
    std::ifstream signature(root/"signature.txt");
    const std::string actual((std::istreambuf_iterator<char>(signature)),{});
    if(actual!=copper_response_signature(config))throw std::invalid_argument("Conditional Cu response physics/source/geometry signature mismatch");
    CopperResponseBank bank;
    bank.events=read_copper_response_records<CopperResponseEvent>(root/"events.bin");
    bank.charged=read_copper_response_records<SecondaryParticle>(root/"charged.bin");
    bank.neutrons=read_copper_response_records<NeutronBirthRecord>(root/"neutrons.bin");
    bank.offsets=read_copper_response_records<std::uint32_t>(root/"offsets.bin");
    if(bank.events.empty() || bank.events.size()>UINT32_MAX || bank.offsets.size()!=9001 ||
        bank.offsets.front()!=0 || bank.offsets.back()!=bank.events.size() ||
        !std::is_sorted(bank.offsets.begin(),bank.offsets.end()))
        throw std::invalid_argument("Cu response CSR offsets invalid");
    for(std::size_t cell=0;cell<9000;++cell)for(auto i=bank.offsets[cell];i<bank.offsets[cell+1];++i) {
        const auto& e=bank.events[i];
        if(!std::isfinite(e.input_x) || !std::isfinite(e.input_y) ||
            int(std::floor((e.input_x+15)*10))+300*int(std::floor(e.input_y+15))!=int(cell) ||
            std::uint64_t(e.charged_begin)+e.charged_count>bank.charged.size() ||
            std::uint64_t(e.neutron_begin)+e.neutron_count>bank.neutrons.size() ||
            !std::isfinite(e.removed) || e.removed<0 || !std::isfinite(e.air_loss) || e.air_loss<0 ||
            !std::isfinite(e.primary.energy_MeV) || e.primary.energy_MeV<0 || e.primary.energy_MeV>170.001F)
            throw std::invalid_argument("Invalid Cu response event");
    }
    for(const auto& c:bank.charged)
        if(c.birth_region!=1 || !std::isfinite(c.energy_MeV) || c.energy_MeV<0 || c.weight!=1)
            throw std::invalid_argument("Invalid Cu response charged entry");
    for(const auto& n:bank.neutrons)
        if(n.material_section!=-2 || !std::isfinite(n.kinetic_energy_MeV) || n.kinetic_energy_MeV<0 || n.weight!=1)
            throw std::invalid_argument("Invalid Cu response neutron birth");
    return bank;
}
#endif
