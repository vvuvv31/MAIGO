#include "carbon/schneider_rate_table.hpp"
#include "carbon/schneider_stopping_table.hpp"
#include "carbon/schneider_target_sampler.hpp"
#include "carbon/inelastic_package_v3.hpp"
#include "carbon/rng.hpp"
#include <sycl/sycl.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct TestCaseConfig {
    std::string id;
    std::string category;
    int section_id{8};
    double density{1.0788};
    double energy_mevu{100.0};
    double thickness_mm{10.0};
    int depth_bins{5};
    uint64_t histories{50000};
    std::string topas_json;
};

std::vector<TestCaseConfig> load_manifest(const std::filesystem::path& manifest_path) {
    std::ifstream in(manifest_path);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open manifest: " + manifest_path.string());
    }
    std::stringstream buffer;
    buffer << in.rdbuf();
    const std::string content = buffer.str();

    std::vector<TestCaseConfig> cases;
    size_t pos = 0;
    while ((pos = content.find("\"id\": \"", pos)) != std::string::npos) {
        pos += 7;
        const size_t end_id = content.find("\"", pos);
        const std::string id = content.substr(pos, end_id - pos);

        auto extract_str = [&](const std::string& key) -> std::string {
            const std::string pattern = "\"" + key + "\": \"";
            const size_t kpos = content.find(pattern, end_id);
            if (kpos == std::string::npos || kpos > content.find("\"id\": \"", end_id)) return "";
            const size_t val_start = kpos + pattern.length();
            const size_t val_end = content.find("\"", val_start);
            return content.substr(val_start, val_end - val_start);
        };

        auto extract_num = [&](const std::string& key) -> double {
            const std::string pattern = "\"" + key + "\": ";
            const size_t kpos = content.find(pattern, end_id);
            if (kpos == std::string::npos || kpos > content.find("\"id\": \"", end_id)) return 0.0;
            const size_t val_start = kpos + pattern.length();
            char* end_ptr = nullptr;
            return std::strtod(content.c_str() + val_start, &end_ptr);
        };

        TestCaseConfig tc;
        tc.id = id;
        tc.category = extract_str("category");
        tc.section_id = static_cast<int>(extract_num("section_id"));
        tc.density = extract_num("density");
        tc.energy_mevu = extract_num("energy_mevu");
        tc.thickness_mm = extract_num("thickness_mm");
        tc.depth_bins = static_cast<int>(extract_num("depth_bins"));
        tc.histories = static_cast<uint64_t>(extract_num("histories"));
        tc.topas_json = extract_str("topas_json");
        cases.push_back(tc);
    }
    return cases;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const std::filesystem::path repo_dir = "/mnt/sdb/wuwei/MAIGO";
        const std::filesystem::path manifest_path = "/mnt/sda/wuwei/step19_fragmentation/manifest.json";
        const std::filesystem::path out_dir = "/mnt/sda/wuwei/step19_fragmentation/gpu";
        std::filesystem::create_directories(out_dir);

        std::cout << "Loading rate table and target library...\n";
        const auto rate_table = carbon::SchneiderRateTable::from_binary(
            repo_dir / "data/schneider/schneider_inelastic_rates_v1.bin");
        const carbon::SchneiderTargetSampler target_sampler(rate_table);

        const auto stopping_table = carbon::SchneiderStoppingTable::from_binary(
            repo_dir / "data/schneider/schneider_stopping_v1.bin");

        const auto package = carbon::InelasticPackageV3Table::from_binary(
            repo_dir / "data/schneider/cinel03_c12_targets.bin");
        const auto device_tables = package.make_device_tables();

        sycl::queue queue{sycl::default_selector_v, sycl::property::queue::in_order{}};
        std::cout << "Running on SYCL Device: "
                  << queue.get_device().get_info<sycl::info::device::name>() << std::endl;

        // Upload Sampler CDF and Total Rates to Device
        const auto sampler_dev = target_sampler.device_table();
        float* dev_cdf = sycl::malloc_device<float>(target_sampler.cdf_table().size(), queue);
        float* dev_total_rates = sycl::malloc_device<float>(target_sampler.total_mass_rates().size(), queue);
        queue.copy(target_sampler.cdf_table().data(), dev_cdf, target_sampler.cdf_table().size()).wait_and_throw();
        queue.copy(target_sampler.total_mass_rates().data(), dev_total_rates, target_sampler.total_mass_rates().size()).wait_and_throw();

        carbon::SchneiderTargetSamplerDeviceTable gpu_sampler = sampler_dev;
        gpu_sampler.cdf_table = dev_cdf;
        gpu_sampler.total_mass_rates = dev_total_rates;

        // Upload CINEL03 Package to Device
        carbon::Cinel03EnergyNode* dev_nodes = sycl::malloc_device<carbon::Cinel03EnergyNode>(device_tables.energy_nodes.size(), queue);
        std::uint32_t* dev_offsets = sycl::malloc_device<std::uint32_t>(device_tables.event_offsets.size(), queue);
        std::uint32_t* dev_indices = sycl::malloc_device<std::uint32_t>(device_tables.event_indices.size(), queue);
        carbon::Cinel03DeviceInteraction* dev_interactions = sycl::malloc_device<carbon::Cinel03DeviceInteraction>(device_tables.interactions.size(), queue);
        carbon::Cinel03DeviceProduct* dev_products = sycl::malloc_device<carbon::Cinel03DeviceProduct>(device_tables.products.size(), queue);

        queue.copy(device_tables.energy_nodes.data(), dev_nodes, device_tables.energy_nodes.size()).wait_and_throw();
        queue.copy(device_tables.event_offsets.data(), dev_offsets, device_tables.event_offsets.size()).wait_and_throw();
        queue.copy(device_tables.event_indices.data(), dev_indices, device_tables.event_indices.size()).wait_and_throw();
        queue.copy(device_tables.interactions.data(), dev_interactions, device_tables.interactions.size()).wait_and_throw();
        queue.copy(device_tables.products.data(), dev_products, device_tables.products.size()).wait_and_throw();

        const std::uint32_t node_count = static_cast<std::uint32_t>(device_tables.energy_nodes.size());
        const std::uint32_t total_events = static_cast<std::uint32_t>(device_tables.interactions.size());

        const auto cases = load_manifest(manifest_path);
        std::cout << "Loaded " << cases.size() << " test cases from " << manifest_path << "\n";

        for (const auto& tc : cases) {
            std::cout << "---------------------------------------------------------\n";
            std::cout << "Processing GPU simulation for: " << tc.id << "\n";
            std::cout << "  Section: " << tc.section_id << ", Energy: " << tc.energy_mevu
                      << " MeV/u, Thickness: " << tc.thickness_mm << " mm, Histories: " << tc.histories << "\n";

            const std::size_t N = tc.histories * 4;
            const float e_init = static_cast<float>(tc.energy_mevu);
            const float thick_mm = static_cast<float>(tc.thickness_mm);
            const float rho = static_cast<float>(tc.density);
            const std::size_t sec_id = tc.section_id;

            const double init_sp = stopping_table.interpolate_mass_stopping(sec_id, e_init);
            const double est_loss = init_sp * rho * thick_mm * 0.5 / 12.0;
            const double mid_energy = std::max(1.0, e_init - est_loss);
            const float dedx = static_cast<float>(stopping_table.interpolate_mass_stopping(sec_id, mid_energy) * rho / 12.0);

            // Global atomics for this run
            std::uint64_t* dev_targets = sycl::malloc_device<std::uint64_t>(32, queue);
            std::uint64_t* dev_secondaries = sycl::malloc_device<std::uint64_t>(32, queue);
            std::uint64_t* dev_counts = sycl::malloc_device<std::uint64_t>(4, queue); // [inelastic, survived, overflow, unsupported]
            float* dev_energy_ledger = sycl::malloc_device<float>(4, queue); // [initial, local_dep, escaped, unassigned]

            queue.memset(dev_targets, 0, 32 * sizeof(std::uint64_t)).wait_and_throw();
            queue.memset(dev_secondaries, 0, 32 * sizeof(std::uint64_t)).wait_and_throw();
            queue.memset(dev_counts, 0, 4 * sizeof(std::uint64_t)).wait_and_throw();
            queue.memset(dev_energy_ledger, 0, 4 * sizeof(float)).wait_and_throw();

            queue.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
                const std::uint64_t hist_idx = idx[0];
                const std::uint64_t spot_seed = 123456789ULL;

                // Optical depth sampling
                const float u_tau = carbon::rng::uniform01(spot_seed, hist_idx, 0, 1);
                const float tau_sample = -sycl::log(sycl::fmax(1.0e-12F, u_tau));

                // Target macroscopic total cross section
                // Node index for initial energy
                const float node_flt = (e_init - gpu_sampler.energy_min_MeV_per_u) * gpu_sampler.inverse_energy_step;
                const int node_idx = sycl::clamp(static_cast<int>(node_flt), 0, static_cast<int>(gpu_sampler.num_energies - 1));
                const float mass_tot = gpu_sampler.total_mass_rates[sec_id * gpu_sampler.num_energies + node_idx];
                const float macro_tot = mass_tot * rho;

                const float dist_to_collision = tau_sample / macro_tot;

                sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>
                    init_energy_ref(dev_energy_ledger[0]);
                init_energy_ref.fetch_add(e_init * 12.0F);

                if (dist_to_collision < thick_mm) {
                    // Inelastic collision occurred inside slab!
                    sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        inel_ref(dev_counts[0]);
                    inel_ref.fetch_add(1U);

                    // Continuous slowing along track to collision depth
                    const float e_coll = sycl::fmax(1.0F, e_init - dist_to_collision * dedx);
                    const float pre_coll_loss = (e_init - e_coll) * 12.0F;

                    // 1. Sample target element Z
                    const float u_target = carbon::rng::uniform01(spot_seed, hist_idx, 1, 2);
                    const int target_z = carbon::sample_schneider_target_device(gpu_sampler, sec_id, e_coll, u_target);

                    if (target_z > 0 && target_z < 32) {
                        sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                         sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            tgt_ref(dev_targets[target_z]);
                        tgt_ref.fetch_add(1U);
                    }

                    // 2. Replay correlated event from CINEL03 with 15 MeV/u window (isolates campaign node without adjacent mixing)
                    const float tolerance = 15.0F;
                    const float u_event = carbon::rng::uniform01(spot_seed, hist_idx, 2, 3);
                    const std::uint32_t event_id = carbon::cinel03_find_event_device(
                        dev_nodes, node_count, dev_offsets, dev_indices, total_events,
                        6, 12, target_z, e_coll, tolerance, u_event);

                    if (event_id == 0xFFFFFFFFU) {
                        sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                         sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            unsup_ref(dev_counts[3]);
                        unsup_ref.fetch_add(1U);
                    } else {
                        const auto event = dev_interactions[event_id];
                        const std::uint32_t prod_offset = event.product_offset;
                        const std::uint32_t prod_count = event.direct_product_count;

                        float fragment_kinetic_sum = 0.0F;
                        for (std::uint32_t p = 0; p < prod_count; ++p) {
                            const auto prod = dev_products[prod_offset + p];
                            fragment_kinetic_sum += prod.kinetic_energy_MeV;
                            const int pz = prod.z;
                            if (pz > 0 && pz < 32) {
                                sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    sec_ref(dev_secondaries[pz]);
                                sec_ref.fetch_add(1U);
                            }
                        }

                        sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                         sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            loc_ref(dev_energy_ledger[1]);
                        loc_ref.fetch_add(pre_coll_loss + event.process_local_deposit_MeV + fragment_kinetic_sum);
                    }
                } else {
                    // Survived slab without collision
                    sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        surv_ref(dev_counts[1]);
                    surv_ref.fetch_add(1U);

                    const float e_esc = sycl::fmax(0.0F, e_init - thick_mm * dedx) * 12.0F;
                    const float continuous_loss = (e_init * 12.0F - e_esc);

                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        loc_ref(dev_energy_ledger[1]);
                    loc_ref.fetch_add(continuous_loss);

                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        esc_ref(dev_energy_ledger[2]);
                    esc_ref.fetch_add(e_esc);
                }
            }).wait_and_throw();

            std::vector<std::uint64_t> host_targets(32);
            std::vector<std::uint64_t> host_secondaries(32);
            std::vector<std::uint64_t> host_counts(4);
            std::vector<float> host_energy(4);

            queue.copy(dev_targets, host_targets.data(), 32).wait_and_throw();
            queue.copy(dev_secondaries, host_secondaries.data(), 32).wait_and_throw();
            queue.copy(dev_counts, host_counts.data(), 4).wait_and_throw();
            queue.copy(dev_energy_ledger, host_energy.data(), 4).wait_and_throw();

            sycl::free(dev_targets, queue);
            sycl::free(dev_secondaries, queue);
            sycl::free(dev_counts, queue);
            sycl::free(dev_energy_ledger, queue);

            // Write results JSON
            const auto gpu_json = out_dir / (tc.id + "_gpu.json");
            std::ofstream out(gpu_json);
            out << "{\n";
            out << "  \"id\": \"" << tc.id << "\",\n";
            out << "  \"histories\": " << tc.histories << ",\n";
            out << "  \"inelastic_count\": " << host_counts[0] << ",\n";
            out << "  \"survived_count\": " << host_counts[1] << ",\n";
            out << "  \"overflow_count\": " << host_counts[2] << ",\n";
            out << "  \"unsupported_lookup_count\": " << host_counts[3] << ",\n";

            out << "  \"target_element_counts\": {\n";
            bool first = true;
            for (int z = 1; z < 32; ++z) {
                if (host_targets[z] > 0) {
                    if (!first) out << ",\n";
                    first = false;
                    out << "    \"" << z << "\": " << host_targets[z];
                }
            }
            out << "\n  },\n";

            out << "  \"secondary_species_counts\": {\n";
            first = true;
            for (int z = 1; z < 32; ++z) {
                if (host_secondaries[z] > 0) {
                    if (!first) out << ",\n";
                    first = false;
                    out << "    \"" << z << "\": " << host_secondaries[z];
                }
            }
            out << "\n  },\n";

            out << "  \"energy_ledger\": {\n";
            out << "    \"initial_total_MeV\": " << host_energy[0] << ",\n";
            out << "    \"local_deposit_MeV\": " << host_energy[1] << ",\n";
            out << "    \"escaped_MeV\": " << host_energy[2] << "\n";
            out << "  }\n";
            out << "}\n";
            out.close();

            std::cout << "  Completed: Inelastic=" << host_counts[0]
                      << ", Survived=" << host_counts[1]
                      << ", Unsupported=" << host_counts[3]
                      << ", Saved to " << gpu_json << "\n";
        }

        sycl::free(dev_cdf, queue);
        sycl::free(dev_total_rates, queue);
        sycl::free(dev_nodes, queue);
        sycl::free(dev_offsets, queue);
        sycl::free(dev_indices, queue);
        sycl::free(dev_interactions, queue);
        sycl::free(dev_products, queue);

        std::cout << "=========================================================\n";
        std::cout << "All GPU simulations finished successfully!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
