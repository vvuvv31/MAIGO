#include "carbon/inelastic_package_v3.hpp"
#include "carbon/particle.hpp"
#include "carbon/rng.hpp"
#include "carbon/schneider_rate_table.hpp"
#include "carbon/schneider_stopping_table.hpp"
#include "carbon/schneider_target_sampler.hpp"
#include "carbon/secondary_rate_table.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct TestCase {
    std::string id;
    std::string category;
    int section_id{0};
    double density{1.0};
    double energy_mevu{100.0};
    double thickness_mm{10.0};
    int depth_bins{5};
    uint64_t histories{50000};
    std::string topas_json;
};

std::vector<TestCase> load_manifest(const std::filesystem::path& manifest_path) {
    std::ifstream in(manifest_path);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open manifest: " + manifest_path.string());
    }
    std::stringstream buffer;
    buffer << in.rdbuf();
    const std::string content = buffer.str();

    std::vector<TestCase> cases;
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

        TestCase tc;
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
        const std::filesystem::path repo_dir = std::filesystem::current_path();
        const std::filesystem::path manifest_path = "/mnt/sda/wuwei/step19_fragmentation/manifest.json";
        const std::filesystem::path out_dir = "/mnt/sda/wuwei/step20_secondary_transport/gpu";
        std::filesystem::create_directories(out_dir);

        std::cout << "Loading rate tables, stopping tables, and event libraries...\n";
        const auto rate_table = carbon::SchneiderRateTable::from_binary(
            repo_dir / "data/schneider/schneider_inelastic_rates_v1.bin");
        const carbon::SchneiderTargetSampler target_sampler(rate_table);

        const auto sec_rate_table = carbon::SecondaryRateTable::from_binary(
            repo_dir / "data/schneider/secondary_inelastic_rates_v1.bin");

        const auto stopping_table = carbon::SchneiderStoppingTable::from_binary(
            repo_dir / "data/schneider/schneider_stopping_v1.bin");

        // Primary C12 package
        const auto c12_package = carbon::InelasticPackageV3Table::from_binary(
            repo_dir / "data/schneider/cinel03_c12_targets.bin");
        const auto c12_dev_tables = c12_package.make_device_tables();

        // Secondary projectiles package
        const auto sec_package = carbon::InelasticPackageV3Table::from_binary(
            repo_dir / "data/schneider/cinel03_secondary_targets.bin");
        const auto sec_dev_tables = sec_package.make_device_tables();

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

        // Upload Schneider Stopping Power Table
        const auto flat_sp = stopping_table.to_flat_mass_stopping_float();
        float* dev_stopping = sycl::malloc_device<float>(flat_sp.size(), queue);
        queue.copy(flat_sp.data(), dev_stopping, flat_sp.size()).wait_and_throw();

        // Upload Secondary Total Rates
        std::vector<float> sec_total_rates_float(sec_rate_table.mass_total_rates().size());
        for (std::size_t i = 0; i < sec_rate_table.mass_total_rates().size(); ++i) {
            sec_total_rates_float[i] = static_cast<float>(sec_rate_table.mass_total_rates()[i]);
        }
        float* dev_sec_total = sycl::malloc_device<float>(sec_total_rates_float.size(), queue);
        queue.copy(sec_total_rates_float.data(), dev_sec_total, sec_total_rates_float.size()).wait_and_throw();

        // Upload Secondary Partial Rates
        std::vector<float> sec_partial_rates_float(sec_rate_table.mass_partial_rates().size());
        for (std::size_t i = 0; i < sec_rate_table.mass_partial_rates().size(); ++i) {
            sec_partial_rates_float[i] = static_cast<float>(sec_rate_table.mass_partial_rates()[i]);
        }
        float* dev_sec_partial = sycl::malloc_device<float>(sec_partial_rates_float.size(), queue);
        queue.copy(sec_partial_rates_float.data(), dev_sec_partial, sec_partial_rates_float.size()).wait_and_throw();

        // Upload Primary CINEL03 Package
        carbon::Cinel03EnergyNode* dev_c12_nodes = sycl::malloc_device<carbon::Cinel03EnergyNode>(c12_dev_tables.energy_nodes.size(), queue);
        std::uint32_t* dev_c12_offsets = sycl::malloc_device<std::uint32_t>(c12_dev_tables.event_offsets.size(), queue);
        std::uint32_t* dev_c12_indices = sycl::malloc_device<std::uint32_t>(c12_dev_tables.event_indices.size(), queue);
        carbon::Cinel03DeviceInteraction* dev_c12_ints = sycl::malloc_device<carbon::Cinel03DeviceInteraction>(c12_dev_tables.interactions.size(), queue);
        carbon::Cinel03DeviceProduct* dev_c12_prods = sycl::malloc_device<carbon::Cinel03DeviceProduct>(c12_dev_tables.products.size(), queue);

        queue.copy(c12_dev_tables.energy_nodes.data(), dev_c12_nodes, c12_dev_tables.energy_nodes.size()).wait_and_throw();
        queue.copy(c12_dev_tables.event_offsets.data(), dev_c12_offsets, c12_dev_tables.event_offsets.size()).wait_and_throw();
        queue.copy(c12_dev_tables.event_indices.data(), dev_c12_indices, c12_dev_tables.event_indices.size()).wait_and_throw();
        queue.copy(c12_dev_tables.interactions.data(), dev_c12_ints, c12_dev_tables.interactions.size()).wait_and_throw();
        queue.copy(c12_dev_tables.products.data(), dev_c12_prods, c12_dev_tables.products.size()).wait_and_throw();

        const std::uint32_t c12_node_count = static_cast<std::uint32_t>(c12_dev_tables.energy_nodes.size());
        const std::uint32_t c12_total_events = static_cast<std::uint32_t>(c12_dev_tables.interactions.size());

        // Upload Secondary CINEL03 Package
        carbon::Cinel03EnergyNode* dev_sec_nodes = sycl::malloc_device<carbon::Cinel03EnergyNode>(sec_dev_tables.energy_nodes.size(), queue);
        std::uint32_t* dev_sec_offsets = sycl::malloc_device<std::uint32_t>(sec_dev_tables.event_offsets.size(), queue);
        std::uint32_t* dev_sec_indices = sycl::malloc_device<std::uint32_t>(sec_dev_tables.event_indices.size(), queue);
        carbon::Cinel03DeviceInteraction* dev_sec_ints = sycl::malloc_device<carbon::Cinel03DeviceInteraction>(sec_dev_tables.interactions.size(), queue);
        carbon::Cinel03DeviceProduct* dev_sec_prods = sycl::malloc_device<carbon::Cinel03DeviceProduct>(sec_dev_tables.products.size(), queue);

        queue.copy(sec_dev_tables.energy_nodes.data(), dev_sec_nodes, sec_dev_tables.energy_nodes.size()).wait_and_throw();
        queue.copy(sec_dev_tables.event_offsets.data(), dev_sec_offsets, sec_dev_tables.event_offsets.size()).wait_and_throw();
        queue.copy(sec_dev_tables.event_indices.data(), dev_sec_indices, sec_dev_tables.event_indices.size()).wait_and_throw();
        queue.copy(sec_dev_tables.interactions.data(), dev_sec_ints, sec_dev_tables.interactions.size()).wait_and_throw();
        queue.copy(sec_dev_tables.products.data(), dev_sec_prods, sec_dev_tables.products.size()).wait_and_throw();

        const std::uint32_t sec_node_count = static_cast<std::uint32_t>(sec_dev_tables.energy_nodes.size());
        const std::uint32_t sec_total_events = static_cast<std::uint32_t>(sec_dev_tables.interactions.size());

        const auto cases = load_manifest(manifest_path);
        std::cout << "Loaded " << cases.size() << " test cases from " << manifest_path << "\n";

        for (const auto& tc : cases) {
            std::cout << "---------------------------------------------------------\n";
            std::cout << "Processing GPU simulation with secondary transport for: " << tc.id << "\n";
            std::cout << "  Section: " << tc.section_id << ", Energy: " << tc.energy_mevu
                      << " MeV/u, Thickness: " << tc.thickness_mm << " mm, Histories: " << tc.histories << "\n";

            const std::size_t N = tc.histories * 4;
            const float e_init = static_cast<float>(tc.energy_mevu);
            const float thick_mm = static_cast<float>(tc.thickness_mm);
            const float rho = static_cast<float>(tc.density);
            const std::size_t sec_id = static_cast<std::size_t>(tc.section_id);

            const double init_sp = stopping_table.interpolate_mass_stopping(sec_id, e_init);
            const double est_loss = init_sp * rho * thick_mm * 0.5 / 12.0;
            const double mid_energy = std::max(1.0, e_init - est_loss);
            const float dedx = static_cast<float>(stopping_table.interpolate_mass_stopping(sec_id, mid_energy) * rho / 12.0);

            // Global atomics for this run
            std::uint64_t* dev_targets = sycl::malloc_device<std::uint64_t>(32, queue);
            std::uint64_t* dev_direct_sec = sycl::malloc_device<std::uint64_t>(32, queue);
            std::uint64_t* dev_exit_sec = sycl::malloc_device<std::uint64_t>(32, queue);
            std::uint64_t* dev_counts = sycl::malloc_device<std::uint64_t>(5, queue); // [primary_inel, survived, overflow, unsupported, sec_inel]
            float* dev_energy_ledger = sycl::malloc_device<float>(4, queue); // [initial, local_dep, escaped, unassigned]

            queue.memset(dev_targets, 0, 32 * sizeof(std::uint64_t)).wait_and_throw();
            queue.memset(dev_direct_sec, 0, 32 * sizeof(std::uint64_t)).wait_and_throw();
            queue.memset(dev_exit_sec, 0, 32 * sizeof(std::uint64_t)).wait_and_throw();
            queue.memset(dev_counts, 0, 5 * sizeof(std::uint64_t)).wait_and_throw();
            queue.memset(dev_energy_ledger, 0, 4 * sizeof(float)).wait_and_throw();

            queue.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
                const std::uint64_t hist_idx = idx[0];
                const std::uint64_t spot_seed = 123456789ULL;

                sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>
                    init_energy_ref(dev_energy_ledger[0]);
                init_energy_ref.fetch_add(e_init * 12.0F);

                // Continuous optical depth tracking through slab
                const float u_tau = carbon::rng::uniform01(spot_seed, hist_idx, 0, 1);
                float tau_remaining = -sycl::log(sycl::fmax(1.0e-12F, u_tau));

                float z = 0.0F;
                float cur_e = e_init;
                bool primary_collided = false;
                float dist_to_collision = 0.0F;
                float e_coll = 0.0F;

                const bool is_continuous = (thick_mm >= 40.0F || (sec_id == 20 && e_init < 150.0F));

                if (!is_continuous) {
                    const float mass_tot = carbon::schneider_total_mass_rate_device(gpu_sampler, sec_id, e_init);
                    const float macro_tot = mass_tot * rho;
                    dist_to_collision = tau_remaining / macro_tot;
                    if (dist_to_collision < thick_mm) {
                        primary_collided = true;
                        e_coll = sycl::fmax(1.0F, e_init - dist_to_collision * dedx);
                    } else {
                        cur_e = sycl::fmax(0.0F, e_init - thick_mm * dedx);
                    }
                } else {
                    float tau_accum = 0.0F;
                    const float sub_step_max = 0.5F;

                    while (z < thick_mm && cur_e > 1.0F) {
                        const float step = sycl::fmin(sub_step_max, thick_mm - z);

                        const float mass_tot = carbon::schneider_total_mass_rate_device(gpu_sampler, sec_id, cur_e);
                        const float macro_tot = mass_tot * rho;

                        const float delta_tau = macro_tot * step;
                        if (macro_tot > 0.0F && (tau_accum + delta_tau) >= tau_remaining) {
                            const float frac = sycl::clamp((tau_remaining - tau_accum) / delta_tau, 0.0F, 1.0F);
                            dist_to_collision = z + frac * step;

                            const float floating_sp = (cur_e - 0.01F) * 10.0F;
                            const int sp_idx = sycl::clamp(static_cast<int>(sycl::floor(floating_sp)), 0, 4300);
                            const float sp_frac = sycl::clamp(floating_sp - static_cast<float>(sp_idx), 0.0F, 1.0F);
                            const float mass_sp = dev_stopping[sec_id * 4302 + sp_idx] +
                                                  sp_frac * (dev_stopping[sec_id * 4302 + sp_idx + 1] -
                                                             dev_stopping[sec_id * 4302 + sp_idx]);
                            const float step_dedx = mass_sp * rho / 12.0F;
                            e_coll = sycl::fmax(1.0F, cur_e - frac * step * step_dedx);
                            primary_collided = true;
                            break;
                        }

                        tau_accum += delta_tau;

                        const float floating_sp = (cur_e - 0.01F) * 10.0F;
                        const int sp_idx = sycl::clamp(static_cast<int>(sycl::floor(floating_sp)), 0, 4300);
                        const float sp_frac = sycl::clamp(floating_sp - static_cast<float>(sp_idx), 0.0F, 1.0F);
                        const float mass_sp = dev_stopping[sec_id * 4302 + sp_idx] +
                                              sp_frac * (dev_stopping[sec_id * 4302 + sp_idx + 1] -
                                                         dev_stopping[sec_id * 4302 + sp_idx]);
                        const float step_dedx = mass_sp * rho / 12.0F;
                        cur_e = sycl::fmax(1.0F, cur_e - step * step_dedx);
                        z += step;
                    }
                }

                if (primary_collided) {
                    // Primary collision occurred inside slab
                    sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        inel_ref(dev_counts[0]);
                    inel_ref.fetch_add(1U);

                    const float pre_coll_loss = (e_init - e_coll) * 12.0F;

                    // Sample target element Z
                    const float u_target = carbon::rng::uniform01(spot_seed, hist_idx, 1, 2);
                    const int target_z = carbon::sample_schneider_target_device(gpu_sampler, sec_id, e_coll, u_target);

                    if (target_z > 0 && target_z < 32) {
                        sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                         sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            tgt_ref(dev_targets[target_z]);
                        tgt_ref.fetch_add(1U);
                    }

                    // Replay primary event from C12 package
                    const float tolerance = 15.0F;
                    const float u_event = carbon::rng::uniform01(spot_seed, hist_idx, 2, 3);
                    const std::uint32_t event_id = carbon::cinel03_find_event_device(
                        dev_c12_nodes, c12_node_count, dev_c12_offsets, dev_c12_indices, c12_total_events,
                        6, 12, target_z, e_coll, tolerance, u_event);

                    if (event_id == 0xFFFFFFFFU) {
                        sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                         sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            unsup_ref(dev_counts[3]);
                        unsup_ref.fetch_add(1U);
                    } else {
                        const auto event = dev_c12_ints[event_id];
                        const std::uint32_t prod_offset = event.product_offset;
                        const std::uint32_t prod_count = event.direct_product_count;

                        float fragment_kinetic_sum = 0.0F;
                        const float remaining_dist_mm = thick_mm - dist_to_collision;

                        for (std::uint32_t p = 0; p < prod_count; ++p) {
                            const auto prod = dev_c12_prods[prod_offset + p];
                            fragment_kinetic_sum += prod.kinetic_energy_MeV;
                            const int pz = prod.z;
                            const int pa = prod.a;

                            // Tally direct secondary
                            if (pz > 0 && pz < 32) {
                                sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    dsec_ref(dev_direct_sec[pz]);
                                dsec_ref.fetch_add(1U);
                            }

                            // Secondary projectile transport
                            int proj_idx = -1;
                            if (pz == 5 && pa == 11) proj_idx = 0;
                            else if (pz == 5 && pa == 10) proj_idx = 1;
                            else if (pz == 4 && pa == 9) proj_idx = 2;
                            else if (pz == 4 && pa == 7) proj_idx = 3;
                            else if (pz == 4 && pa == 10) proj_idx = 4;
                            else if (pz == 3 && pa == 7) proj_idx = 5;
                            else if (pz == 3 && pa == 6) proj_idx = 6;
                            else if (pz == 2 && pa == 4) proj_idx = 7;
                            else if (pz == 2 && pa == 3) proj_idx = 8;
                            else if (pz == 1 && pa == 1) proj_idx = 9;
                            else if (pz == 1 && pa == 2) proj_idx = 10;
                            else if (pz == 1 && pa == 3) proj_idx = 11;
                            else if (pz == 6 && pa == 11) proj_idx = 12;

                            bool secondary_reacted = false;
                            if (proj_idx >= 0 && pa > 0 && prod.kinetic_energy_MeV > 5.0F) {
                                const float sec_e_mevu = prod.kinetic_energy_MeV / static_cast<float>(pa);
                                const float sec_f_node = (sec_e_mevu - 0.5F) * 2.0F;
                                float sec_mass_tot = 0.0F;
                                if (sec_f_node <= 0.0F) {
                                    sec_mass_tot = dev_sec_total[static_cast<std::size_t>(proj_idx) * (25 * 860) + sec_id * 860];
                                } else if (sec_f_node >= 859.0F) {
                                    sec_mass_tot = dev_sec_total[static_cast<std::size_t>(proj_idx) * (25 * 860) + sec_id * 860 + 859];
                                } else {
                                    const int s_idx0 = static_cast<int>(sec_f_node);
                                    const float s_frac = sec_f_node - static_cast<float>(s_idx0);
                                    const float sv0 = dev_sec_total[static_cast<std::size_t>(proj_idx) * (25 * 860) + sec_id * 860 + s_idx0];
                                    const float sv1 = dev_sec_total[static_cast<std::size_t>(proj_idx) * (25 * 860) + sec_id * 860 + s_idx0 + 1];
                                    sec_mass_tot = sv0 + s_frac * (sv1 - sv0);
                                }
                                const float sec_macro_tot = sec_mass_tot * rho;

                                if (sec_macro_tot > 1.0e-8F) {
                                    const float u_sec_tau = carbon::rng::uniform01(spot_seed, hist_idx, 10 + p, 4);
                                    const float sec_tau_sample = -sycl::log(sycl::fmax(1.0e-12F, u_sec_tau));
                                    const float dist_sec_coll = sec_tau_sample / sec_macro_tot;

                                    if (dist_sec_coll < remaining_dist_mm) {
                                        // Secondary nuclear interaction occurs!
                                        secondary_reacted = true;
                                        sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                                         sycl::memory_scope::device,
                                                         sycl::access::address_space::global_space>
                                            sec_inel_ref(dev_counts[4]);
                                        sec_inel_ref.fetch_add(1U);

                                        // Replay secondary breakup with independent target element sampling
                                        const float u_sec_tgt = carbon::rng::uniform01(spot_seed, hist_idx, 20 + p, 4);
                                        const int sec_target_z = carbon::sample_secondary_target_device(
                                            dev_sec_partial, proj_idx, sec_id, sec_e_mevu, u_sec_tgt,
                                            0.5F, 2.0F, 860);
                                        const float u_sec_ev = carbon::rng::uniform01(spot_seed, hist_idx, 20 + p, 5);
                                        const std::uint32_t sec_ev_id = carbon::cinel03_find_event_device(
                                            dev_sec_nodes, sec_node_count, dev_sec_offsets, dev_sec_indices, sec_total_events,
                                            pz, pa, sec_target_z, sec_e_mevu, 0.51F, u_sec_ev);

                                        if (sec_ev_id != 0xFFFFFFFFU) {
                                            const auto sec_ev = dev_sec_ints[sec_ev_id];
                                            for (std::uint32_t sp = 0; sp < sec_ev.direct_product_count; ++sp) {
                                                const auto tert_prod = dev_sec_prods[sec_ev.product_offset + sp];
                                                const int tpz = tert_prod.z;
                                                if (tpz > 0 && tpz < 32) {
                                                    sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                                                     sycl::memory_scope::device,
                                                                     sycl::access::address_space::global_space>
                                                        exit_ref(dev_exit_sec[tpz]);
                                                    exit_ref.fetch_add(1U);
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            if (!secondary_reacted && pz > 0 && pz < 32) {
                                sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    exit_ref(dev_exit_sec[pz]);
                                exit_ref.fetch_add(1U);
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

                    const float e_esc = sycl::fmax(0.0F, cur_e) * 12.0F;
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

            // Read back results
            std::uint64_t host_targets[32];
            std::uint64_t host_direct_sec[32];
            std::uint64_t host_exit_sec[32];
            std::uint64_t host_counts[5];
            float host_energy[4];

            queue.copy(dev_targets, host_targets, 32).wait_and_throw();
            queue.copy(dev_direct_sec, host_direct_sec, 32).wait_and_throw();
            queue.copy(dev_exit_sec, host_exit_sec, 32).wait_and_throw();
            queue.copy(dev_counts, host_counts, 5).wait_and_throw();
            queue.copy(dev_energy_ledger, host_energy, 4).wait_and_throw();

            sycl::free(dev_targets, queue);
            sycl::free(dev_direct_sec, queue);
            sycl::free(dev_exit_sec, queue);
            sycl::free(dev_counts, queue);
            sycl::free(dev_energy_ledger, queue);

            // Output JSON
            const std::filesystem::path out_file = out_dir / (tc.id + "_gpu.json");
            std::ofstream out(out_file);
            out << "{\n";
            out << "  \"case_id\": \"" << tc.id << "\",\n";
            out << "  \"histories\": " << N << ",\n";
            out << "  \"inelastic_count\": " << host_counts[0] << ",\n";
            out << "  \"survived_count\": " << host_counts[1] << ",\n";
            out << "  \"overflow_count\": " << host_counts[2] << ",\n";
            out << "  \"unsupported_count\": " << host_counts[3] << ",\n";
            out << "  \"secondary_inelastic_count\": " << host_counts[4] << ",\n";
            out << "  \"energy_ledger\": {\n";
            out << "    \"initial_energy_MeV\": " << host_energy[0] << ",\n";
            out << "    \"local_deposit_MeV\": " << host_energy[1] << ",\n";
            out << "    \"escaped_energy_MeV\": " << host_energy[2] << ",\n";
            out << "    \"closure_difference_MeV\": " << std::abs(host_energy[0] - (host_energy[1] + host_energy[2])) << "\n";
            out << "  },\n";
            out << "  \"target_element_counts\": {\n";
            bool first = true;
            for (int z = 1; z < 32; ++z) {
                if (host_targets[z] > 0) {
                    if (!first) out << ",\n";
                    out << "    \"" << z << "\": " << host_targets[z];
                    first = false;
                }
            }
            out << "\n  },\n";
            out << "  \"secondary_species_counts\": {\n";
            first = true;
            for (int z = 1; z < 32; ++z) {
                if (host_direct_sec[z] > 0) {
                    if (!first) out << ",\n";
                    out << "    \"" << z << "\": " << host_direct_sec[z];
                    first = false;
                }
            }
            out << "\n  },\n";
            out << "  \"exit_secondary_species_counts\": {\n";
            first = true;
            for (int z = 1; z < 32; ++z) {
                if (host_exit_sec[z] > 0) {
                    if (!first) out << ",\n";
                    out << "    \"" << z << "\": " << host_exit_sec[z];
                    first = false;
                }
            }
            out << "\n  }\n";
            out << "}\n";

            std::cout << "  Completed: PrimaryInelastic=" << host_counts[0]
                      << ", SecondaryInelastic=" << host_counts[4]
                      << ", Survived=" << host_counts[1]
                      << ", Unsupported=" << host_counts[3]
                      << ", Saved to \"" << out_file.string() << "\"\n";
        }

        sycl::free(dev_cdf, queue);
        sycl::free(dev_total_rates, queue);
        sycl::free(dev_sec_total, queue);
        sycl::free(dev_sec_partial, queue);
        sycl::free(dev_c12_nodes, queue);
        sycl::free(dev_c12_offsets, queue);
        sycl::free(dev_c12_indices, queue);
        sycl::free(dev_c12_ints, queue);
        sycl::free(dev_c12_prods, queue);
        sycl::free(dev_sec_nodes, queue);
        sycl::free(dev_sec_offsets, queue);
        sycl::free(dev_sec_indices, queue);
        sycl::free(dev_sec_ints, queue);
        sycl::free(dev_sec_prods, queue);
        sycl::free(dev_stopping, queue);

        std::cout << "=========================================================\n";
        std::cout << "All Step 20 GPU simulations finished successfully!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 1;
    }
}
