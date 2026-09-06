#include "carbon/longitudinal_ray.hpp"
#include "carbon/electron_joint_response.hpp"
#include "carbon/sha256.hpp"
#include "carbon/schneider_delta_tail.hpp"
#include "carbon/transport_config.hpp"
#include "carbon/transport.hpp"
#include "carbon/run_quality.hpp"
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#ifdef CARBON_HAS_SYCL
#include <sycl/sycl.hpp>
#endif
using namespace carbon;
static void require(bool ok, const char* msg) {
    if (!ok) throw std::runtime_error(msg);
}
template<class F> static void rejects(F f) {
    bool rejected=false; try { f(); } catch (const std::exception&) { rejected=true; }
    require(rejected,"expected rejection");
}
// This SAME function runs on host and device, including lookup and ray geometry.
static std::array<double,12> exercise_joint() {
    std::array<ElectronJointChannel,74> c{};
    for(auto& v:c)v={0,2,.25};
    const ElectronJointSample s[]={{.4,-.01,.02},{1,.03,.04}};
    const std::array<double,2> low{155,0},high{185,185};
    const auto a=sample_electron_joint_device(0,175,.2,c.data(),s,low,high);
    const auto b=sample_electron_joint_device(8,185,.4,c.data(),s,low,high);
    std::array<double,12> out{a.fraction,a.longitudinal_mass_g_cm2,a.radial_mass_g_cm2,
        b.longitudinal_mass_g_cm2,b.radial_mass_g_cm2};
    out[5]=sample_electron_joint_device(1,175,.2,c.data(),s,low,high).status==ElectronJointStatus::unsupported_section;
    out[6]=sample_electron_joint_device(0,154,.2,c.data(),s,low,high).status==ElectronJointStatus::energy_domain;
    out[7]=sample_electron_joint_device(0,186,.2,c.data(),s,low,high).status==ElectronJointStatus::energy_domain;
    out[8]=sample_electron_joint_device(0,175,-.1,c.data(),s,low,high).status==ElectronJointStatus::invalid_payload;
    c[35].count=0;
    out[9]=sample_electron_joint_device(0,175,.2,c.data(),s,low,high).status==ElectronJointStatus::invalid_payload;
    c[37].fraction=0;
    out[10]=sample_electron_joint_device(8,0,.2,c.data(),s,low,high).fraction;
    out[11]=sample_electron_joint_device(0,174.999,.2,c.data(),s,low,high).status==ElectronJointStatus::hit;
    return out;
}
static std::array<double,5> exercise_polyline() {
    // Two crossings and a return to the original z. The collapsed vector
    // lands at x=11mm; retaining the turns must return x=2mm instead.
    const std::array<double,3> v[]={{.1,0,.1},{1,0,1},{0,0,-1},{0,0,-.1}};
    const auto r=replay_mass_polyline({0,0,-1},v,4,{-1,-1,-2},{1,1,1},{20,2,4},
        [](const std::array<int,3>& c){return c[2]<2 ? 1. : 10.;});
    return {r.endpoint[0],r.endpoint[1],r.endpoint[2],double(r.invalid || r.escaped),double(r.completed_segments)};
}
static std::array<double,16> exercise() {
    std::array<double,16> out{};
    const float e[]{150,200,225}, f[]{0.08F,0.09F,0.1F}, l[]{20,25,30};
    float fraction=0,lambda=0;
    out[0]=schneider_longitudinal_lookup_device(175,e,f,l,3,fraction,lambda);
    out[1]=fraction; out[2]=lambda;
    out[3]=schneider_longitudinal_lookup_device(149,e,f,l,3,fraction,lambda);
    out[4]=fraction;
    out[5]=schneider_longitudinal_lookup_device(226,e,f,l,3,fraction,lambda);
    out[6]=schneider_longitudinal_lookup_device(
        std::numeric_limits<float>::quiet_NaN(),e,f,l,3,fraction,lambda);
    auto r=march_longitudinal_segments({0.4,0.25,0.25},{1,0,0},
        {0,0,0},{0.5,0.5,0.5},{4,1,1},0.5,
        [&](const std::array<int,3>& cell,double length) {
            out[7+cell[0]]+=length/0.5;return true;
        });
    out[9]=r.traversed_mm;
    r=march_longitudinal_segments({0.5,0.25,0.25},{-1,0,0},
        {0,0,0},{0.5,0.5,0.5},{4,1,1},1,
        [&](const std::array<int,3>& cell,double length) {
            out[10]+=length;out[11]=cell[0];return true;
        });
    out[12]=r.escaped;
    r=march_longitudinal_segments({0.4,0.25,0.25},{1,0,0},
        {0,0,0},{0.5,0.5,0.5},{4,1,1},1,
        [&](const std::array<int,3>& cell,double) {return cell[0]==0;});
    out[13]=r.blocked;out[14]=r.traversed_mm;
    r=march_longitudinal_segments({0.25,0.25,0.25},{0,0,0},
        {0,0,0},{0.5,0.5,0.5},{4,1,1},1,
        [](const std::array<int,3>&,double){return true;});
    out[15]=r.invalid;
    return out;
}
static std::array<double,12> exercise_mass() {
    std::array<double,12> out{};
    auto rho=[](const std::array<int,3>& c){return c[0]==0 ? 1.0 : 10.0;};
    auto r=march_longitudinal_mass_segments({0,0.25,0.25},{1,0,0},
        {0,0,0},{0.5,0.5,0.5},{2,1,1},0.15,rho,
        [&](const std::array<int,3>& c,double share){out[c[0]]+=share;});
    out[2]=r.traversed_mm;out[3]=r.invalid || r.blocked || r.escaped;
    r=march_longitudinal_mass_segments({1,0.25,0.25},{-1,0,0},
        {0,0,0},{0.5,0.5,0.5},{2,1,1},0.6,rho,
        [&](const std::array<int,3>& c,double share){out[4+c[0]]+=share;});
    out[6]=r.escaped;out[7]=r.traversed_mm;
    r=march_longitudinal_mass_segments({0,0.25,0.25},{1,0,0},
        {0,0,0},{0.5,0.5,0.5},{2,1,1},0.1,
        [](const std::array<int,3>&){return 0.0;},
        [&](const std::array<int,3>&,double){out[8]+=1;});
    out[9]=r.invalid;
    r=march_longitudinal_mass_segments({0,0.25,0.25},{1,0,0},
        {0,0,0},{0.5,0.5,0.5},{2,1,1},0.55,rho,
        [&](const std::array<int,3>&,double share){out[10]+=share;});
    out[11]=r.escaped || r.invalid || r.blocked;
    return out;
}
static void ordered_loader_fixtures() {
    const auto dir=std::filesystem::temp_directory_path()/("maigo-ordered-loader-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(dir);
    const auto csv=dir/"joint.csv",meta=dir/"joint.metadata.json",binary=dir/"paths.bin";
    {std::ofstream f(csv);f<<"section_id,energy_bin,nonlocal_fraction,cdf,longitudinal_mass_g_cm2,radial_mass_g_cm2\n0,35,0.25,1,0.1,0.1\n8,35,0.25,1,0.1,0.1\n";}
    auto write=[&](int fault) {
        {std::ofstream f(binary,std::ios::binary);f.write(fault==1?"BADMAGIC":"ELPATH01",8);
         const std::uint32_t header[]{1,2,2};f.write(reinterpret_cast<const char*>(header),sizeof(header));
         const ElectronPathRange ranges[]={{0,1},{1,static_cast<std::uint32_t>(fault==2?2:1)}};
         f.write(reinterpret_cast<const char*>(ranges),sizeof(ranges));
         const std::array<double,3> vectors[]={{.1,0,.1},{fault==3?.2:.1,0,.1}};
         f.write(reinterpret_cast<const char*>(vectors),sizeof(vectors));}
        std::ofstream f(meta);
        f<<"{\"schema_version\":2,\"status\":\"UNVALIDATED_INTERFACE_DIAGNOSTIC\",\"projectile\":[6,12],\"energy_bin_width_MeVu\":5,\"energy_max_MeVu\":185,\"charged_escape_allowed\":false,\"fixed_parent_inset_mm\":{\"air\":100,\"tissue\":5},\"sources\":[{}],\"data_filename\":\"joint.csv\",\"data_sha256\":\""<<compute_file_sha256_hex(csv)<<"\",\"data_size_bytes\":"<<std::filesystem::file_size(csv)
         <<",\"ordered_path_file\":\"paths.bin\",\"ordered_path_sha256\":\""<<compute_file_sha256_hex(binary)<<"\",\"ordered_path_size_bytes\":"<<std::filesystem::file_size(binary)<<",\"channels\":[";
        for(int sec:{0,8}) {if(sec)f<<',';
            f<<"{\"section_id\":"<<sec<<",\"energy_bin\":35,\"energy_min_MeVu\":175,\"energy_max_MeVu\":180,\"nonlocal_fraction\":0.25,\"parent_loss_MeV\":1,\"terminal_local_excess_MeV\":0,\"raw_kinetic_closure_residual_MeV\":0,\"redistribution_inactive\":false}";}
        f<<"]}";
    };
    auto load=[&]{return ElectronJointResponseTable::from_csv(csv,compute_file_sha256_hex(csv),compute_file_sha256_hex(meta));};
    write(0);require(load().path_vectors.size()==2,"ordered fixture");
    for(int fault:{1,2,3}) {write(fault);rejects(load);}
    write(0);std::filesystem::remove(binary);rejects(load);
    std::filesystem::remove_all(dir);
}
int main(int argc, char** argv) {
    try {
        ordered_loader_fixtures();
        // Optional external diagnostic payload: no dependency on unshipped data
        // in the ordinary regression suite.
        if(argc==2) {
            const std::filesystem::path file=argv[1];
            auto meta=file;meta.replace_extension(".metadata.json");
            const auto dp=compute_file_sha256_hex(file),mp=compute_file_sha256_hex(meta);
            const auto table=ElectronJointResponseTable::from_csv(file,dp,mp);
            require(!table.samples.empty(),"joint real payload load");
            rejects([&]{ElectronJointResponseTable::from_csv(file,std::string(64,'0'),mp);});
            rejects([&]{ElectronJointResponseTable::from_csv(file,dp,std::string(64,'0'));});
            for(unsigned sec:{0U,8U})for(double e:{165.,170.,175.,180.}) {
                auto d=sample_electron_joint_device(sec,e,.5,table.channels.data(),table.samples.data(),table.minimum,table.maximum);
                require(d.status==ElectronJointStatus::hit,"real joint coverage");
            }
        }
        const auto host=exercise();
        const auto mass=exercise_mass();
        require(std::abs(mass[0]-1.0/3)<1e-12 && std::abs(mass[1]-2.0/3)<1e-12,"mass interface split");
        require(std::abs(mass[2]-.6)<1e-12 && mass[3]==0,"partial terminal tissue cell");
        require(std::abs(mass[4]+mass[5]-11.0/12)<1e-12 && mass[6]==1 && mass[7]==1,"reverse mass escape closure");
        require(mass[8]==0 && mass[9]==1 && std::abs(mass[10]-1)<1e-12 && mass[11]==0,"invalid density/exact exit");
        validate_longitudinal_interface_grid({static_cast<float>(kLongitudinalReferenceDensityGPerCm3),1.0787997245788574F},{0,8});
        rejects([]{validate_longitudinal_interface_grid({1.0F},{8});});
        rejects([]{validate_longitudinal_interface_grid({1.0787997245788574F},{7});});
        reject_unvalidated_longitudinal_heterogeneity({.011F,.011F},{0,0});
        rejects([]{reject_unvalidated_longitudinal_heterogeneity({.011F,1.0788F},{0,8});});
        rejects([]{reject_unvalidated_longitudinal_heterogeneity({.011F,.012F},{0,0});});
        {TransportConfig bad;bad.ct_longitudinal_interface_mass_diagnostic=true;rejects([&]{bad.validate();});}
        require(host[0]==1 && std::abs(host[1]-0.085)<1e-7 && host[2]==22.5,"interpolation");
        require(host[3]==0 && host[4]==0 && host[5]==0 && host[6]==0,"domain rejects");
        require(std::abs(host[7]-0.2)<1e-12 && std::abs(host[8]-0.8)<1e-12,"20/80 split");
        require(host[9]==0.5 && host[10]==0.5 && host[11]==0 && host[12]==1,"negative boundary escape");
        require(host[13]==1 && std::abs(host[14]-0.1)<1e-12 && host[15]==1,"blocked/invalid");
        // Oblique, anisotropic grid; translating source and grid is invariant.
        for (float rho : {static_cast<float>(kLongitudinalReferenceDensityGPerCm3),
                          0.03932345286011696F, 0.06621015816926956F}) {
            require(longitudinal_probe_density({rho,rho}, {0,0})==rho,"probe density");
            rejects([&]{longitudinal_probe_density({rho,rho}, {0,1});});
            rejects([&]{longitudinal_probe_density({rho,rho*1.00001F}, {0,0});});
        }
        rejects([]{longitudinal_probe_density({},{});});
        rejects([]{longitudinal_probe_density({0.02F},{0});});
        rejects([]{longitudinal_probe_density({std::numeric_limits<float>::quiet_NaN()},{0});});
        {
            TransportConfig bad;
            bad.ct_longitudinal_homogeneous_density_diagnostic=true;
            rejects([&]{bad.validate();});
        }
        for (double shift : {0.0,0.123,100.0}) {
            double total=0;
            const auto r=march_longitudinal_segments(
                {shift+0.1,shift+0.2,shift+0.3},{0.6,0,0.8},
                {shift,shift,shift},{0.5,1,2},{20,20,20},7.3,
                [&](const std::array<int,3>& c,double len) {
                    require(c[0]>=0 && c[2]>=0 && len>0,"oblique segment");
                    total+=len;return true;
                });
            require(!r.invalid && !r.escaped && std::abs(total-7.3)<1e-12,"translation/closure");
        }
        // Unlike the old Python mirror, exercise the compiled device function.
#ifdef CARBON_HAS_SYCL
        sycl::queue queue(sycl::gpu_selector_v);
        const auto polyline=exercise_polyline();
        const std::array<double,5> expected_polyline{2,0,-1,0,4};
        for(int i=0;i<5;++i)require(std::abs(polyline[i]-expected_polyline[i])<1e-12,"ordered path interface identity");
        auto* poly_device=sycl::malloc_shared<std::array<double,5>>(1,queue);
        queue.single_task([=]{*poly_device=exercise_polyline();}).wait_and_throw();
        for(int i=0;i<5;++i)require(std::abs((*poly_device)[i]-polyline[i])<1e-12,"polyline host/device");
        sycl::free(poly_device,queue);
        const auto joint=exercise_joint();
        require(joint==std::array<double,12>{.25,-.01,.02,.03,.04,1,1,1,1,1,0,1},"joint sampling semantics");
        auto* joint_device=sycl::malloc_shared<std::array<double,12>>(1,queue);
        queue.single_task([=]{*joint_device=exercise_joint();}).wait_and_throw();
        require(*joint_device==joint,"joint host/device equivalence");
        sycl::free(joint_device,queue);
        auto* device=sycl::malloc_shared<std::array<double,16>>(1,queue);
        require(device!=nullptr,"allocation");
        queue.single_task([=]{*device=exercise();}).wait_and_throw();
        for(std::size_t i=0;i<host.size();++i)
            require(std::abs((*device)[i]-host[i])<1e-12,"host/device equivalence");
        sycl::free(device,queue);
        auto* mass_device=sycl::malloc_shared<std::array<double,12>>(1,queue);
        require(mass_device!=nullptr,"mass allocation");
        queue.single_task([=]{*mass_device=exercise_mass();}).wait_and_throw();
        for(std::size_t i=0;i<mass.size();++i)
            require(std::abs((*mass_device)[i]-mass[i])<1e-12,"mass host/device equivalence");
        sycl::free(mass_device,queue);
        std::cout<<"GPU host/device queries and rays passed\n";
#endif
        const std::filesystem::path file =
            std::filesystem::path(CARBON_SOURCE_DIR) /
            "data/schneider/schneider_section0_c12_delta_longitudinal_v1.csv";
        require(SchneiderLongitudinalTable::from_csv(file).energy_count()==3,"pinned data");
        const auto tmp=std::filesystem::temp_directory_path()/
            ("maigo-longitudinal-test-"+std::to_string(
             std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directory(tmp);
        const auto copy=tmp/file.filename();
        std::filesystem::copy_file(file,copy);
        rejects([&]{SchneiderLongitudinalTable::from_csv(copy);}); // no sidecars
        const auto meta=file.parent_path()/(file.stem().string()+".metadata.json");
        const auto manifest=file.parent_path()/(file.stem().string()+".candidate.json");
        std::filesystem::copy_file(meta,tmp/meta.filename());
        rejects([&]{SchneiderLongitudinalTable::from_csv(copy);}); // no manifest
        std::filesystem::copy_file(manifest,tmp/manifest.filename());
        require(SchneiderLongitudinalTable::from_csv(copy).energy_count()==3,"relocated pins");
        {std::ofstream bad(tmp/meta.filename(),std::ios::app);bad<<" ";}
        rejects([&]{SchneiderLongitudinalTable::from_csv(copy);});
        std::filesystem::copy_file(meta,tmp/meta.filename(),std::filesystem::copy_options::overwrite_existing);
        {std::ofstream bad(copy,std::ios::app);bad<<" ";}
        rejects([&]{SchneiderLongitudinalTable::from_csv(copy);});
        std::filesystem::remove_all(tmp);
        auto result=std::make_unique<TransportResult>();
        TransportConfig config;
        config.ct_schneider_delta_longitudinal_file=file;
        const auto quality=evaluate_run_quality(config,*result);
        bool blocked=false;
        for(const auto& f:quality.failures) if(f.code=="unvalidated_longitudinal_candidate") blocked=true;
        require(blocked && !quality.accepted,"candidate cannot pass quality");
        {
            TransportConfig joint;
            joint.ct_electron_joint_response_diagnostic_file="diagnostic.csv";
            const auto q=evaluate_run_quality(joint,*result);
            bool unvalidated=false;
            for(const auto& f:q.failures)if(f.code=="unvalidated_electron_joint_response")unvalidated=true;
            require(unvalidated && !q.accepted,"joint candidate cannot pass quality");
            for(auto mode:{RunMode::research,RunMode::production}) {
                joint.run_mode=mode;bool refused=false;
                try {joint.validate();}catch(const std::exception& e) {
                    refused=std::string(e.what()).find("Joint electron response requires pinned, isolated")!=std::string::npos;
                }
                require(refused,"joint patient/production scope rejection");
            }
        }
        config.enable_ct_grid=true;
        config.enable_voxel_scoring=true;
        config.ct_grid_file="diagnostic-grid-not-read";
        config.ct_schneider_delta_tail_file=file.parent_path()/
            "schneider_section0_c12_delta_tail_v1.csv";
        for (auto mode : {RunMode::research,RunMode::production,RunMode::smoke}) {
            config.run_mode=mode;
            config.ct_schneider_delta_longitudinal_scale=(mode==RunMode::smoke)?0.63:1;
            bool refused=false;
            try {config.validate();} catch(const std::exception& e) {
                refused=std::string(e.what()).find("Unvalidated longitudinal candidate")!=std::string::npos;
            }
            require(refused,"mode/scale contract");
        }
        std::cout<<"Longitudinal pinned data, domain, geometry and quality gates passed\n";
        return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
