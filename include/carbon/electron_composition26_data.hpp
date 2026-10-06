#pragma once
#include "carbon/electron_collision_composition26.hpp"
#include "carbon/electron_atomic26_data.hpp"
#include <array>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string_view>
#include <vector>
namespace carbon {
// Host component owner, with exactly26 directory entries. GPU upload must
// relocate every pointer; host views are never passed directly to a device.
// Photon laws, birth/rates and spatial joint are separate required components.
class ElectronComposition26Data {
    template<class T> static void read(std::ifstream& f,T& x){if(!f.read(reinterpret_cast<char*>(&x),sizeof(x)))throw std::runtime_error("Composition26 truncated");}
    template<class T> static void array(std::ifstream& f,std::vector<T>& x,std::size_t n){x.resize(n);if(n&&!f.read(reinterpret_cast<char*>(x.data()),n*sizeof(T)))throw std::runtime_error("Composition26 truncated array");}
    static void header(std::ifstream& f,std::string_view expected,unsigned expected_version=1){char magic[8]{};unsigned version,count;if(!f.read(magic,8))throw std::runtime_error("Composition26 missing header");read(f,version);read(f,count);if(std::string_view(magic,8)!=expected||version!=expected_version||count!=26)throw std::runtime_error("Composition26 invalid header/count");}
    static void end(std::ifstream& f){char extra;if(f.read(&extra,1))throw std::runtime_error("Composition26 trailing bytes");}
    static void finite(const std::vector<float>& v,bool positive=false){for(float x:v)if(!std::isfinite(x)||(positive&&!(x>0.f)))throw std::runtime_error("Composition26 invalid float");}
    static void increasing(const std::vector<float>& v){finite(v,true);for(unsigned i=1;i<v.size();++i)if(!(v[i]>v[i-1]))throw std::runtime_error("Composition26 unordered grid");}
public:
    struct Entry {
        std::vector<float> scalar_energy,scalar_coeff,cut_coeff,delta,brems_energy,brems_fraction,brems_integral;
        std::vector<unsigned> brems_z;std::vector<float> brems_elements;
        std::vector<electron_composition26::PenelopeOscillator> oscillators;
        unsigned scalar_count{},cut_pieces{},cut_count{};
        float log_low{},log_high{},reference_density{},density_low{},density_high{};
        ElectronContinuousScalarView scalar_view()const{return {scalar_energy.data(),scalar_coeff.data(),cut_coeff.data(),unsigned(scalar_energy.size()),scalar_count,cut_pieces,cut_count,log_low,log_high};}
        electron_composition26::PenelopeComposition penelope_view()const{return {oscillators.data(),unsigned(oscillators.size()),delta.data(),unsigned(delta.size())};}
        electron_composition26::BremsSpectrum26 brems_view()const{return {brems_energy.data(),brems_fraction.data(),brems_integral.data(),unsigned(brems_energy.size()),unsigned(brems_fraction.size()),brems_z.data(),brems_elements.data(),unsigned(brems_z.size())};}
    };
    std::array<Entry,26> entries;
    ElectronAtomic26Data atomic;
    explicit ElectronComposition26Data(const std::filesystem::path& directory):atomic(directory/"atomic26.bin"){
        {
            std::ifstream f(directory/"scalar26.bin",std::ios::binary);header(f,"MAIESC26");
            for(unsigned i=0;i<26;++i){auto& e=entries[i];unsigned section,n;read(f,section);read(f,n);read(f,e.scalar_count);read(f,e.cut_pieces);read(f,e.cut_count);read(f,e.log_low);read(f,e.log_high);
                if(section!=i||n<2||n>100000||!e.scalar_count||e.scalar_count>128||!e.cut_pieces||e.cut_pieces>128||!e.cut_count||e.cut_count>128||!std::isfinite(e.log_low)||!std::isfinite(e.log_high)||e.log_low>e.log_high)throw std::runtime_error("Composition26 scalar descriptor");
                array(f,e.scalar_energy,n);array(f,e.scalar_coeff,std::size_t(n)*e.scalar_count*3);array(f,e.cut_coeff,std::size_t(e.cut_pieces)*e.cut_count*2);
                increasing(e.scalar_energy);finite(e.scalar_coeff);finite(e.cut_coeff);
            }end(f);
        }
        {
            std::ifstream f(directory/"penelope26.bin",std::ios::binary);header(f,"MAIPEN26");
            for(unsigned i=0;i<26;++i){auto& e=entries[i];unsigned section,n,nd;read(f,section);read(f,n);read(f,nd);read(f,e.reference_density);read(f,e.density_low);read(f,e.density_high);
                if(section!=i||!n||n>128||nd!=34||!(e.density_low>0.f)||!std::isfinite(e.density_high)||e.density_high<e.density_low||!std::isfinite(e.reference_density)||e.reference_density<e.density_low||e.reference_density>e.density_high)throw std::runtime_error("Composition26 oscillator descriptor");
                array(f,e.oscillators,n);array(f,e.delta,nd);finite(e.delta);
                for(auto o:e.oscillators)if(!(o.strength>0.f)||!std::isfinite(o.strength)||!std::isfinite(o.ionization)||o.ionization<0.f||!(o.resonance>0.f)||!std::isfinite(o.resonance)||!(o.recoil_cut>0.f)||!std::isfinite(o.recoil_cut)||!std::isfinite(o.binding)||o.binding<0.f||o.z>104||o.shell_flag>30)throw std::runtime_error("Composition26 oscillator law");
            }end(f);
        }
        {
            std::ifstream f(directory/"brems26.bin",std::ios::binary);header(f,"MAISB026",2);
            for(unsigned i=0;i<26;++i){auto& e=entries[i];unsigned section,ne,nx,nz;read(f,section);read(f,ne);read(f,nx);read(f,nz);
                if(section!=i||ne<2||ne>4096||nx<2||nx>4096||!nz||nz>104)throw std::runtime_error("Composition26 SB descriptor");
                array(f,e.brems_energy,ne);array(f,e.brems_fraction,nx);array(f,e.brems_integral,std::size_t(ne)*nx);
                array(f,e.brems_z,nz);array(f,e.brems_elements,std::size_t(nz)*ne*nx);finite(e.brems_elements);
                for(unsigned k=0;k<nz;++k){if(!e.brems_z[k]||e.brems_z[k]>104||(k&&e.brems_z[k]<=e.brems_z[k-1]))throw std::runtime_error("Composition26 element identity");
                    for(unsigned j=0;j<ne;++j){const auto offset=(std::size_t(k)*ne+j)*nx;if(e.brems_elements[offset+nx-1]!=0.f)throw std::runtime_error("Composition26 element endpoint");for(unsigned x=0;x<nx;++x)if(e.brems_elements[offset+x]<0.f||(x&&e.brems_elements[offset+x]>e.brems_elements[offset+x-1]))throw std::runtime_error("Composition26 element spectrum");}}
                increasing(e.brems_energy);increasing(e.brems_fraction);finite(e.brems_integral);
                if(e.brems_fraction.back()!=1.f)throw std::runtime_error("Composition26 SB endpoint");
                for(unsigned j=0;j<ne;++j){if(e.brems_integral[j*nx+nx-1]!=0.f)throw std::runtime_error("Composition26 SB integral endpoint");for(unsigned k=0;k<nx;++k)if(e.brems_integral[j*nx+k]<0.f||(k&&e.brems_integral[j*nx+k]>e.brems_integral[j*nx+k-1]))throw std::runtime_error("Composition26 SB integral ordering");}
            }end(f);
        }
    }
    electron_composition26::CollisionComposition26 view(unsigned section,ElectronCutBirthComposition shells)const{
        if(section>=26)throw std::runtime_error("Composition26 section outside directory");
        const auto& e=entries[section];return {e.scalar_view(),e.penelope_view(),atomic.view(),e.brems_view(),shells};
    }
};
static_assert(sizeof(electron_composition26::PenelopeOscillator)==28);
}
