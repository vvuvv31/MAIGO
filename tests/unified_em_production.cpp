#include "carbon/transport_config.hpp"
#include <iostream>
#include <stdexcept>
#include <string>
#include <limits>
int main(int argc,char** argv) {
    if(argc!=2) throw std::runtime_error("usage: unified_em_production production-config");
    const auto baseline=carbon::load_config(argv[1]);
    if(baseline.run_mode!=carbon::RunMode::production || baseline.em_model!="g4_material_joint_v1")
        throw std::runtime_error("Expected production unified EM config");
    if (!baseline.enable_secondary_unified_em || !baseline.secondary_species_grouping)
        throw std::runtime_error("Production preset must enable full secondary EM and grouping");
    baseline.validate();
    unsigned checks=1;
    auto ungrouped = baseline;
    ungrouped.secondary_species_grouping = false;
    ungrouped.validate(); ++checks;
    auto rejects=[&](auto change,const std::string& expected) {
        auto config=baseline;change(config);
        try {config.validate();} catch(const std::invalid_argument& e) {
            if(std::string(e.what()).find(expected)==std::string::npos)throw;
            ++checks;return;
        }
        throw std::runtime_error("Missing rejection: "+expected);
    };
    rejects([](auto& c){c.em_package_sha256=std::string(64,'0');},"authorized package");
    rejects([](auto& c){c.enable_energy_straggling=false;},"primary/secondary fluctuations");
    rejects([](auto& c){c.enable_secondary_energy_straggling=false;},"primary/secondary fluctuations");
    rejects([](auto& c){c.straggling_scale=1.2;},"native unscaled fluctuations");
    rejects([](auto& c){c.ct_secondary_exact_faces=false;},"exact CT faces");
    rejects([](auto& c){c.primary_em_model="g4_joint_water_v1";},"stacked");
    rejects([](auto& c){c.ct_schneider_delta_tail_file="unused.csv";},"stacking is forbidden");
    rejects([](auto& c){c.em_primary_step_scale=1.25;},"research unified EM");
    rejects([](auto& c){c.em_secondary_step_scale=1.25;},"research unified EM");
    rejects([](auto& c){c.em_primary_step_scale=std::numeric_limits<double>::quiet_NaN();},"finite");
    rejects([](auto& c){c.em_secondary_step_scale=1.51;},"finite");
    auto research=baseline;research.run_mode=carbon::RunMode::research;
    research.em_primary_step_scale=1.25;research.em_secondary_step_scale=1.5;
    research.validate();++checks;
    std::cout<<"production configuration checks="<<checks<<" passed\n";
}
