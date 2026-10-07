#include "carbon/run_session.hpp"
#include "carbon/batch_run.hpp"
#include "carbon/device.hpp"
#include "carbon/io.hpp"
#include "carbon/plan_run.hpp"
#include "carbon/run_quality.hpp"
#include "carbon/runtime_timing.hpp"
#include "carbon/tps_source.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <stdexcept>
namespace carbon {
namespace {
void prepare_source(TransportConfig& config,ImmutableInputCache& inputs) {
    if(!config.enable_tps_source)return;
    const StoppingPowerTable* air=nullptr;
    if(config.spots_enable_upstream_air_energy_loss)
        air=&inputs.get<StoppingPowerTable>(input_cache_key(config.spots_upstream_air_stopping_power_file),[&]{
            return StoppingPowerTable::from_csv(config.spots_upstream_air_stopping_power_file);});
    auto plan=TpsSourcePlan::from_config(config);
    auto batch=plan.make_primary_batch(config,air);
    if(batch.empty())throw std::invalid_argument("TPS source has no active spots");
    config.number_of_histories=batch.back().history_end;
    config.enable_emittance_source=std::any_of(batch.begin(),batch.end(),[](const auto& p){
        return p.emittance_sigma_x_mm()>0 || p.emittance_sigma_y_mm()>0 ||
               p.emittance_sigma_x_prime()>0 || p.emittance_sigma_y_prime()>0;});
    config.initial_energy_MeVu=batch.front().initial_energy_MeV()/config.primary_mass_number;
    config.primary_spot_batch=std::move(batch);
    config.validate();
}
void write_result(const TransportConfig& config,const TransportResult& result,
                  const std::filesystem::path& directory) {
    const auto quality=evaluate_run_quality(config,result);
    write_run_quality_report_json(directory/"quality_report.json",quality);
    write_energy_ledger_json(directory/"energy_ledger.json",config,result);
    if(config.is_schneider_ct_mode() || config.unified_water_nuclear_transport) {
        write_schneider_miss_log_json(directory/"schneider_miss_log.json",result);
        write_schneider_unsupported_tracks_json(directory/"schneider_unsupported_tracks.json",result);
    }
    if(!quality.accepted && config.run_mode==RunMode::production)
        throw std::runtime_error(quality.summary());
    if(config.enable_voxel_scoring && !config.voxel_dose_mhd_output_file.empty())
        write_dense_voxel_dose_mhd(config.voxel_dose_mhd_output_file,config,result);
    if(!quality.accepted)throw std::runtime_error(quality.summary());
    std::cout<<"Quality status: "<<quality.status()<<"\nQuality report: "<<directory/"quality_report.json"<<'\n';
}
}
void run_session(const CliState& cli) {
    RuntimeScope timer("process_main");
    RunShard base; base.config_path=cli.config_path;base.config=load_config(cli.config_path);
    if(cli.seed_overridden)base.config.random_seed=cli.seed;
    if(cli.histories_overridden)base.config.number_of_histories=cli.histories;
    base.config.validate();
    ImmutableInputCache inputs;
    std::vector<RunShard> batches;
    if(cli.batch_count) {
        if(std::filesystem::exists(cli.batch_output))throw std::invalid_argument("Batch output already exists: "+cli.batch_output.string());
        batches=make_repeated_batches(base,cli,inputs);write_batch_manifest(batches,cli);
    } else {
        prepare_source(base.config,inputs);
        base.output_directory=std::filesystem::path("out")/cli.config_path.stem();
        if(std::filesystem::exists(base.output_directory))throw std::invalid_argument("Output already exists: "+base.output_directory.string());
        batches.push_back(std::move(base));
    }
    std::cout<<"SYCL device: "<<describe_sycl_device(batches.front().config.device)<<'\n';
    auto context=std::make_unique<SyclTransportContext>(batches.front().config.device);
    for(std::size_t i=0;i<batches.size();++i) {
        const auto start=std::chrono::steady_clock::now();const auto& batch=batches[i];const auto& config=batch.config;
        const auto& stopping=inputs.get<StoppingPowerTable>(input_cache_key(config.primary_stopping_power_file),[&]{
            return StoppingPowerTable::from_csv(config.primary_stopping_power_file);});
        const auto result=run_transport(config,stopping,context.get());
        write_result(config,result,batch.output_directory);
        std::cout<<"[batch-wall] index="<<i<<" seconds="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()
                 <<" histories="<<config.number_of_histories<<'\n';
    }
}
}
