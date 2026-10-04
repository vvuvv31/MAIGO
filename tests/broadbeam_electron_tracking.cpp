#include "carbon/transport_config.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <source_location>
#include <sstream>
#include <stdexcept>

namespace fs=std::filesystem;
void check(bool ok,std::source_location loc=std::source_location::current()) {
    if(!ok)throw std::runtime_error("Broadbeam tracking test failed at line "+std::to_string(loc.line()));
}
int main() {
    const auto root=fs::current_path();
    const auto tmp=root/"out"/("tracking_config_test_"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(tmp);
    struct Cleanup {fs::path p;~Cleanup(){fs::remove_all(p);}} cleanup{tmp};
    auto read=[](const fs::path& p){std::ifstream f(p);std::ostringstream s;s<<f.rdbuf();return s.str();};
    auto replace=[](std::string text,const std::string& from,const std::string& to){
        const auto i=text.find(from);check(i!=std::string::npos);text.replace(i,from.size(),to);return text;};
    auto load=[&](const std::string& text){const auto p=tmp/"case.yaml";std::ofstream(p)<<text;return carbon::load_config(p);};
    auto rejects=[&](const std::string& text,const std::string& message){
        bool failed=false;try{load(text);}catch(const std::exception& e){failed=std::string(e.what()).find(message)!=std::string::npos;}
        check(failed);
    };
    for(const auto name:{"proton_water_broadbeam","carbon_water_broadbeam","proton_ct_plan_fe","carbon_ct_plan_fe_joint"}) {
        const auto original=read(root/"config"/(std::string(name)+".yaml"));
        auto on=load(original);check(on.broadbeam_electron_tracking);
        check(on.broadbeam_independent_electrons==on.enable_ct_grid);
        check(on.ct_electron_joint_response_diagnostic_file.empty() && on.ct_schneider_delta_tail_file.empty());
        if(on.enable_ct_grid)check(!on.material_electron_charged_index_file.empty() && on.material_electron_precision==4);
        else check(on.material_electron_charged_index_file.empty() && !on.material_electron_retain_bank);
        auto off_text=replace(original,"broadbeam_electron_tracking: true","broadbeam_electron_tracking: false");
        // Explicit OFF overrides stale joint/tail keys as well as declared
        // shared package paths, so absence of any electron package is allowed.
        off_text+="\nct_electron_joint_response_diagnostic_file: does-not-exist.csv\nct_schneider_delta_tail_file: does-not-exist-tail.csv\nct_electron_joint_execution_mode: queued\n";
        auto off=load(off_text);check(!off.broadbeam_electron_tracking && !off.broadbeam_independent_electrons);
        check(off.material_electron_birth_file.empty() && off.material_electron_charged_index_file.empty() && off.material_photon_response_index_file.empty());
        check(off.material_electron_joint_pilot_file.empty() && !off.material_electron_retain_bank);
        check(off.ct_electron_joint_response_diagnostic_file.empty() && off.ct_schneider_delta_tail_file.empty() && off.ct_electron_joint_execution_mode=="inline");
        check(off.material_electron_execution_mode=="inline");
        if(on.enable_ct_grid) {
            auto metric_off=load(off_text+"\nbroadbeam_electron_region_metric: euclidean\n");
            check(!metric_off.broadbeam_independent_electrons);
            auto air_off=load(off_text+"\nbroadbeam_electron_external_air_local: true\nbroadbeam_electron_body_mask_file: missing-body.bin\n");
            check(air_off.broadbeam_electron_body_mask_file.empty());
            rejects(original+"\nbroadbeam_electron_external_air_local: true\n","pinned BODY mask");
            const auto anatomical=original+"\nbroadbeam_electron_region_diagnostics: true\nbroadbeam_electron_body_mask_file: body.bin\nbroadbeam_electron_body_mask_sha256: "+std::string(64,'a')+"\n";
            const auto diag=load(anatomical);
            check(diag.broadbeam_electron_region_diagnostics && !diag.broadbeam_electron_external_air_local && !diag.broadbeam_electron_body_mask_file.empty());
            const auto diag_roundtrip=load(diag.canonical_config_text);
            check(diag_roundtrip.broadbeam_electron_body_mask_file==diag.broadbeam_electron_body_mask_file && diag_roundtrip.broadbeam_electron_body_mask_sha256==diag.broadbeam_electron_body_mask_sha256);
            rejects(original+"\nbroadbeam_electron_region_diagnostics: true\nbroadbeam_electron_body_mask_file: body.bin\n","pinned BODY mask");
            const auto dormant=load(original+"\nbroadbeam_electron_body_mask_file: missing-body.bin\nbroadbeam_electron_body_mask_sha256: invalid\n");
            check(dormant.broadbeam_electron_body_mask_file.empty());
            // Low-density variance reduction: preset value, explicit override and
            // roundtrip, inert when tracking is OFF, and range guards.
            const std::string key="broadbeam_electron_low_density_survival:";
            auto strip=[&](std::string text){const auto at=text.find(key);
                return at==std::string::npos?text:text.substr(0,at)+text.substr(text.find('\n',at)+1);};
            const auto without=strip(original);
            check(load(without).broadbeam_electron_low_density_survival==1 && !load(without).broadbeam_electron_expected_scoring);
            const auto vr=load(without+"\n"+key+" 0.25\nbroadbeam_electron_expected_scoring: true\nbroadbeam_electron_expected_min_energy_MeV: 0.01\n");
            check(vr.broadbeam_electron_low_density_survival==0.25 && vr.broadbeam_electron_expected_scoring && vr.broadbeam_electron_expected_min_energy_MeV==0.01);
            const auto vr_roundtrip=load(vr.canonical_config_text);
            check(vr_roundtrip.broadbeam_electron_low_density_survival==0.25 && vr_roundtrip.broadbeam_electron_expected_scoring);
            const auto vr_off=load(strip(off_text)+"\n"+key+" 0.25\nbroadbeam_electron_expected_scoring: true\n");
            check(vr_off.broadbeam_electron_low_density_survival==1 && !vr_off.broadbeam_electron_expected_scoring);
            rejects(without+"\n"+key+" 0\n","(0,1]");
            rejects(without+"\n"+key+" 1.5\n","(0,1]");
            rejects(without+"\nbroadbeam_electron_expected_min_energy_MeV: -1\n","(0,1]");
        } else {
            rejects(original+"\nbroadbeam_electron_low_density_survival: 0.5\n","CT broadbeam shared-electron");
            rejects(original+"\nbroadbeam_electron_expected_scoring: true\n","CT broadbeam shared-electron");
        }
        auto roundtrip=load(on.canonical_config_text);check(roundtrip.broadbeam_electron_tracking && roundtrip.broadbeam_independent_electrons==on.enable_ct_grid);
        rejects(original+"\nbroadbeam_independent_electrons: false\n","Conflicting broadbeam");
        rejects(replace(original,"broadbeam_electron_tracking: true","broadbeam_electron_tracking: invalid"),"boolean");
        rejects(replace(original,"broadbeam_electron_halo_mm: 5.0","broadbeam_electron_halo_mm: -1"),"valid density/halo");
        if(!on.enable_ct_grid) {
            rejects(replace(original,"broadbeam_electron_density_threshold: 0.1","broadbeam_electron_density_threshold: 1.0"),"Water_75eV");
        }
        std::cout<<"PASS "<<name<<" ON/OFF, stale-path removal, roundtrip, selector/boolean/halo/variance-reduction guards\n";
    }
    const auto mini=read(root/"config/proton_ct_minibeam.yaml");
    rejects(mini+"\nbroadbeam_electron_tracking: false\n","broadbeam_fast setting");
    std::cout<<"PASS minibeam scope guard\n";
}
