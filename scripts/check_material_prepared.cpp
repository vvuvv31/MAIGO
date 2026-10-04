// Compare a prepared artifact with freshly validated source-derived arrays.
#include "carbon/material_electron_prepared.hpp"
#include "carbon/electron_continuation_links.hpp"
#include "carbon/electron_continuation_crossings.hpp"
#include "carbon/electron_birth_index.hpp"
#include <iostream>
using namespace carbon;
template<class T> std::vector<T> read(MaterialPreparedReader& r,const std::string& name) {
    std::vector<T> v;std::size_t n=0;
    r.read<T>(name,[&](std::size_t count) {v.resize(count);return v.data();},n);return v;
}
template<class T> void equal(const std::vector<T>& a,const std::vector<T>& b) {
    if(a!=b)throw std::runtime_error("Prepared index mismatch");
}
void equal_step(const ElectronContinuationStep& a,const ElectronContinuationStep& b) {
#define FIELD(f) if(a.f!=b.f)throw std::runtime_error("Prepared decoded mismatch: " #f)
    FIELD(valid);FIELD(run);FIELD(event);FIELD(track);FIELD(parent);FIELD(pdg);FIELD(step);
    FIELD(parent_valid);FIELD(parent_step);FIELD(pre_position);FIELD(post_position);
    FIELD(pre_direction);FIELD(post_direction);FIELD(parent_direction);FIELD(pre_energy_MeV);
    FIELD(post_energy_MeV);FIELD(deposited_MeV);FIELD(length_mm);FIELD(birth_energy_MeV);
    FIELD(pre_density_g_cm3);FIELD(post_density_g_cm3);FIELD(pre_material);FIELD(post_material);
    FIELD(post_step_status);FIELD(track_status);
#undef FIELD
}
int main(int argc,char** argv) {
    try {
        if(argc!=6)throw std::runtime_error("Usage: check INDEX CACHE Z A SORTED_TABLE_ID");
        const auto index=MaterialElectronResponseIndex::load(argv[1],compute_file_sha256_hex(argv[1]),std::stoi(argv[3]),std::stoi(argv[4]));
        const auto id=std::stoul(argv[5]);const auto key=material_prepared_key(index,id);
        MaterialPreparedReader r(argv[2],key);const auto table=index.load_table(id);
        const auto channels=read<WaterElectronChannel>(r,"channels");
        if(channels.size()!=table.channels.size())throw std::runtime_error("Channel count mismatch");
        for(std::size_t i=0;i<channels.size();++i) {
            const auto& a=channels[i];const auto& b=table.channels[i];
            if(a.low!=b.low || a.high!=b.high || a.fraction!=b.fraction || a.unresolved!=b.unresolved || a.offset!=b.offset || a.count!=b.count)
                throw std::runtime_error("Channel mismatch");
        }
        constexpr std::size_t budget=96ULL<<30;
        const auto sources=index.load_state_sources(id,budget);
        ElectronBirthIndex births(table.channels,index.projectile_z,index.projectile_a);
        const int pdg=index.projectile_z==1?2212:1000000000+index.projectile_z*10000+index.projectile_a*10;
        std::size_t checked=0;
        for(std::size_t s=0;s<sources.size();++s) {
            const auto& raw=sources[s];const auto decoded=read<ElectronContinuationStep>(r,"decoded");
            if(decoded.size()!=raw.size()/kElectronStateV4RecordBytes)throw std::runtime_error("Decoded row count mismatch");
            for(std::size_t row=0;row<decoded.size();++row)equal_step(decoded[row],read_electron_continuation_step(raw.data(),raw.size(),row));
            const auto links=ElectronContinuationLinks::build(raw.data(),raw.size(),budget,pdg);
            equal(read<std::uint64_t>(r,"next"),links.next);
            equal(read<std::uint64_t>(r,"child_offsets"),links.child_offsets);
            equal(read<std::uint64_t>(r,"children"),links.child_rows);
            births.add(links.view(raw.data(),raw.size()),s);
            const auto c=ElectronEnergyCrossings::build(links.view(raw.data(),raw.size()),budget);
            equal(read<double>(r,"edges"),c.edges);equal(read<std::uint64_t>(r,"offsets"),c.offsets);
            const auto rows=read<std::uint32_t>(r,"crossings32");
            if(rows.size()!=c.rows.size())throw std::runtime_error("Crossing count mismatch");
            for(std::size_t i=0;i<rows.size();++i)if(rows[i]!=c.rows[i])throw std::runtime_error("Crossing mismatch");
            checked+=decoded.size();
        }
        births.finish(budget);const auto bc=read<ElectronBirthChannel>(r,"birth_channels");
        const auto bs=read<ElectronBirthSample>(r,"birth_samples");
        if(bc.size()!=births.channels.size() || bs.size()!=births.samples.size())throw std::runtime_error("Birth counts mismatch");
        for(std::size_t i=0;i<bc.size();++i) {
            const auto& a=bc[i];const auto& b=births.channels[i];
            if(a.low!=b.low || a.high!=b.high || a.fraction!=b.fraction || a.offset!=b.offset || a.count!=b.count)
                throw std::runtime_error("Birth channel mismatch");
        }
        for(std::size_t i=0;i<bs.size();++i)if(bs[i].cdf!=births.samples[i].cdf || bs[i].row!=births.samples[i].row || bs[i].source!=births.samples[i].source)
            throw std::runtime_error("Birth sample mismatch");
        r.finish();std::cout<<"PASS rows="<<checked<<" key="<<key<<std::endl;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
