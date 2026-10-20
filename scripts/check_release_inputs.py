#!/usr/bin/env python3
"""Check current projectile identity, mass conventions, TPS schema and CT geometry."""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
CHECKER = r'''
#include "ct_grid.hpp"
#include "schneider_rate_table.hpp"
#include "transport_config.hpp"
#include "tps_source.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
int main(int argc,char** argv) {
    if(argc!=5)return 2;
    const auto valid=carbon::SchneiderRateTable::from_binary(argv[1]);
    if(valid.projectile_z()!=6||valid.projectile_a()!=12)throw std::runtime_error("C12 identity");
    bool rejected=false;
    try { carbon::SchneiderRateTable::from_binary(argv[1],argv[2]); }
    catch(const std::runtime_error&) { rejected=true; }
    if(!rejected)throw std::runtime_error("Missing projectile accepted");
    for(const auto& [name, expected] : {std::pair{"wrong_magic", "Invalid magic"},
                                     std::pair{"wrong_version", "version mismatch"},
                                     std::pair{"legacy_only", "companion metadata"}}) {
        rejected=false;
        try { carbon::SchneiderRateTable::from_binary(std::filesystem::path(argv[4])/name/std::filesystem::path(argv[1]).filename()); }
        catch(const std::runtime_error& error) { rejected=std::string(error.what()).find(expected)!=std::string::npos; }
        if(!rejected)throw std::runtime_error(std::string("Retired rate contract accepted: ")+name);
    }
    carbon::TransportConfig cfg;
    cfg.require_primary_reference_mass(11174.86328125,"Urban/EM");
    cfg.require_primary_reference_mass(11174.86338798439,"Elastic");
    cfg.require_primary_reference_mass(11177.9296875,"CINEL03",carbon::PrimaryMassConvention::Kinematic);
    for (bool kinematic : {false,true}) {
        rejected=false;
        try { cfg.require_primary_reference_mass(kinematic?11174.86328125:11177.9296875,"Wrong convention",kinematic?carbon::PrimaryMassConvention::Kinematic:carbon::PrimaryMassConvention::Nuclear); }
        catch(const std::invalid_argument&) { rejected=true; }
        if(!rejected)throw std::runtime_error("Mass convention mismatch accepted");
    }
    cfg.primary_atomic_number=1;cfg.primary_mass_number=1;
    cfg.require_primary_reference_mass(938.27203369140625,"Proton EM");
    cfg.require_primary_reference_mass(938.272013,"Proton elastic");
    rejected=false;
    try { carbon::make_primary_ion_definition(2,4); }
    catch(const std::invalid_argument&) { rejected=true; }
    if(!rejected)throw std::runtime_error("Generic primary accepted");
    const auto plan=carbon::TpsSourcePlan::from_csv(argv[3]);
    if(plan.spots.size()!=1 || plan.spots[0].energy_total_MeV!=3000.0 || plan.spots[0].mu_weight!=2500000.0)
        throw std::runtime_error("Migrated carbon source values changed");
    for(const char* name : {"energy", "energy_mevu", "x", "y", "mu_weight", "duplicate"}) {
        rejected=false;
        try { (void)carbon::TpsSourcePlan::from_csv(std::filesystem::path(argv[4])/(std::string(name)+".csv")); }
        catch(const std::runtime_error&) { rejected=true; }
        if(!rejected)throw std::runtime_error("Retired or duplicate TPS column accepted");
    }
    const float face=carbon::clamp_step_to_ct_faces(.5f,.9f,.5f,.5f,1.f,0.f,0.f,0.f,0.f,0.f,1.f,1.f,1.f,2,2,2);
    if(std::fabs(face-.1f)>1e-6f)throw std::runtime_error("Face geometry");
    const float bad=carbon::clamp_step_to_ct_faces(.5f,.9f,.5f,.5f,1.f,0.f,0.f,0.f,0.f,0.f,1.f,1.f,1.f,0,0,0);
    if(!std::isnan(bad))throw std::runtime_error("Missing grid silently accepted");
    std::cout<<"Strict identity/mass/TPS/geometry checks passed\n";
}
'''
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    binary=ROOT/'data/nuclear/schneider/carbon/inelastic_rates.bin'
    metadata=binary.with_suffix('.metadata.json')
    with tempfile.TemporaryDirectory(prefix='maigo_release_inputs_') as temporary:
        temporary=Path(temporary)
        payload=json.loads(metadata.read_text())
        assert payload['projectile']=={'z':6,'a':12}
        del payload['projectile']
        missing=temporary/'missing.json';missing.write_text(json.dumps(payload))
        for name,header in dict(energy='energy,x_mm,y_mm,weight',
                energy_mevu='energy_MeVu,x_mm,y_mm,weight',x='energy_MeV,x,y_mm,weight',
                y='energy_MeV,x_mm,y,weight',mu_weight='energy_MeV,x_mm,y_mm,mu_weight',
                duplicate='energy_MeV,x_mm,y_mm,weight,energy_MeV').items():
            (temporary/(name+'.csv')).write_text(header+'\n3000,0,0,2500000,3000\n')
        for name in ['wrong_magic','wrong_version','legacy_only']:
            directory=temporary/name;directory.mkdir()
            local=directory/binary.name
            if name=='legacy_only':
                local.symlink_to(binary)
                Path(str(local)+'.metadata.json').symlink_to(metadata)
            else:
                raw=bytearray(binary.read_bytes())
                if name=='wrong_magic':raw[:8]=b'SCHNELXS'
                else:raw[8:12]=(1).to_bytes(4,'little')
                local.write_bytes(raw)
                local.with_suffix('.metadata.json').symlink_to(metadata)
        cpp=temporary/'check.cpp';cpp.write_text(CHECKER);exe=temporary/'check'
        sources=['stopping_power.cpp','schneider_rate_table.cpp','tps_source.cpp','source_pose.cpp']
        subprocess.run(['g++','-std=c++20','-O2','-ffunction-sections','-fdata-sections','-Wl,--gc-sections','-I',str(ROOT/'include'),str(cpp),
                        *(str(ROOT/'src'/s) for s in sources),'-o',str(exe)],check=True)
        subprocess.run([str(exe),str(binary),str(missing),str(ROOT/'inputs/water_minibeam/carbon/spots.csv'),str(temporary)],check=True)
    checks=['retired_rate_magic_rejected','retired_rate_version_rejected','retired_metadata_filename_rejected','explicit_C12_projectile','missing_projectile_rejected','nuclear_mass_checked',
            'CINEL03_kinematic_mass_checked','wrong_mass_conventions_rejected',
            'proton_reference_masses_checked','generic_primary_rejected','carbon_source_numeric_values',
            'retired_TPS_columns_rejected','duplicate_TPS_columns_rejected',
            'CT_face_distance','missing_CT_dimensions_invalid']
    args.output.write_text(json.dumps(dict(passed=True,checks=checks),indent=2)+'\n')
if __name__=='__main__':main()
