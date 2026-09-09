#pragma once
#include "carbon/water_electron_response.hpp"
#include "carbon/electron_state_refs.hpp"
#include "carbon/electron_continuation_state.hpp"
#include "carbon/electron_continuation_replay.hpp"
#include "carbon/electron_continuation_sampling.hpp"
#include "carbon/electron_birth_sampling.hpp"

namespace carbon {
struct MaterialElectronTableFile {
    int section{-1};
    double density_g_cm3{};
    double reference_density_g_cm3{};
    std::filesystem::path path;
    std::string data_sha256,metadata_sha256;
    std::filesystem::path state_reference_file;
    std::string state_reference_sha256,state_reference_metadata_sha256;
};
// Load the small index first; callers select resident tables explicitly.
// Loading an index must not allocate the entire material bank on the GPU.
struct MaterialElectronResponseIndex {
    std::vector<MaterialElectronTableFile> files;
    static MaterialElectronResponseIndex load(const std::filesystem::path&,const std::string&);
    WaterElectronResponseTable load_table(std::size_t) const;
    std::vector<unsigned char> load_state_references(std::size_t,std::size_t byte_budget) const;
    std::vector<std::vector<unsigned char>> load_state_sources(std::size_t,std::size_t byte_budget) const;
    // Union of density brackets used by the actual geometry. No table from
    // another section can cover a missing material, even at equal density.
    std::vector<std::size_t> required_tables(const std::vector<std::pair<int,double>>&) const;
};
// Entries are sorted by (section,density). Section -1 denotes explicit
// Water_75eV composition, never a fallback for a Schneider section.
struct MaterialElectronStateSource {
    const unsigned char* data{};
    std::size_t bytes{};
    const std::uint64_t* next_rows{};
    const std::uint64_t* child_offsets{};
    const std::uint64_t* child_rows{};
    std::uint64_t child_count{};
    ElectronEnergyCrossingView crossings{};
    const double* short_range_lengths{}; // Optional immutable, raw-row indexed cache.
    ElectronContinuationView continuation() const {
        return {data,bytes,next_rows,child_offsets,child_rows,child_count};
    }
};
struct MaterialElectronResponseView {
    int section{-1};
    // Lookup nodes use the same float formula density as CCTG. Reference
    // density is the actual TOPAS material density used during extraction.
    double density_g_cm3{};
    double reference_density_g_cm3{};
    const WaterElectronChannel* channels{};
    const WaterElectronSample* samples{};
    const std::uint32_t* heads{};
    const WaterElectronPathNode* nodes{};
    const double* prefix_radius{};
    std::size_t channel_count{},node_count{};
    ElectronStateReferenceView state_references{};
    const MaterialElectronStateSource* state_sources{};
    std::size_t state_source_count{};
    const ElectronBirthChannel* birth_channels{};
    const ElectronBirthSample* birth_samples{};
    std::uint64_t birth_sample_count{};
    ElectronContinuationStep state_at(ElectronStateReference reference) const {
        if(!reference.valid || !state_sources || reference.source>=state_source_count)return {};
        const auto source=state_sources[reference.source];
        return read_electron_continuation_step(source.data,source.bytes,reference.row);
    }
    ElectronReplayStep replay_at(ElectronStateReference reference) const {
        if(!reference.valid || !state_sources || reference.source>=state_source_count)return {};
        return prepare_electron_replay_step(state_sources[reference.source].continuation(),reference.row);
    }
};
// Rejection sampling from energy-bucket proposals yields equal weight for each
// eligible track crossing, NOT each stored track step. A bounded fast proposal
// phase is followed by an exact bucket scan; its cap cannot create fake misses.
// Never replace missing coverage by absorption or a water table.
template<class Uniform>
inline ElectronEnergyCrossingDraw sample_material_electron_continuation(
    const MaterialElectronResponseView& table,int pdg,double energy,Uniform uniform,
    unsigned max_attempts=256) {
    ElectronEnergyCrossingDraw out;
    if(pdg!=11) {out.status=ElectronCrossingStatus::unsupported_particle;return out;}
    if(!table.state_sources || !table.state_source_count || !std::isfinite(energy) || energy<=0)return out;
    std::uint64_t total=0;
    for(std::size_t s=0;s<table.state_source_count;++s) {
        const auto view=table.state_sources[s].crossings;
        const auto bin=view.bin(energy);
        if(bin==view.bins) {out.status=ElectronCrossingStatus::energy_domain;return out;}
        if((view.count && !view.rows) || view.offsets[bin]>view.offsets[bin+1] || view.offsets[bin+1]>view.count)return out;
        const auto count=view.offsets[bin+1]-view.offsets[bin];
        if(count>std::numeric_limits<std::uint64_t>::max()-total)return out;
        total+=count;
    }
    if(!total) {out.status=ElectronCrossingStatus::empty;return out;}
    for(unsigned attempt=0;attempt<max_attempts;++attempt) {
        const double u=uniform();
        if(!std::isfinite(u) || u<0 || u>=1)return out;
        auto pick=static_cast<std::uint64_t>(u*total);
        if(pick>=total)pick=total-1; // floating multiplication endpoint only
        for(std::size_t s=0;s<table.state_source_count;++s) {
            const auto& source=table.state_sources[s];
            const auto view=source.crossings;const auto bin=view.bin(energy);
            const auto count=view.offsets[bin+1]-view.offsets[bin];
            if(pick>=count) {pick-=count;continue;}
            auto candidate=inspect_electron_energy_crossing(source.continuation(),view.rows[view.offsets[bin]+pick],energy);
            candidate.source=s;
            if(candidate.status!=ElectronCrossingStatus::rejected)return candidate;
            break;
        }
    }
    // Low acceptance of a coarse proposal bucket is not missing physics data.
    // Reservoir selection over all eligible crossings preserves the same
    // conditional distribution and has a finite, data-bounded running time.
    std::uint64_t seen=0;
    for(std::size_t s=0;s<table.state_source_count;++s) {
        const auto& source=table.state_sources[s];
        const auto view=source.crossings;const auto bin=view.bin(energy);
        for(auto i=view.offsets[bin];i<view.offsets[bin+1];++i) {
            auto candidate=inspect_electron_energy_crossing(source.continuation(),view.rows[i],energy);
            if(candidate.status==ElectronCrossingStatus::rejected)continue;
            if(candidate.status!=ElectronCrossingStatus::ready)return candidate;
            ++seen;
            const double u=uniform();
            if(!std::isfinite(u) || u<0 || u>=1)return {};
            if(u<1.0/double(seen)) {out=candidate;out.source=s;}
        }
    }
    if(!seen)out.status=ElectronCrossingStatus::empty;
    out.exact_bucket_scan=true;
    return out;
}
enum class MaterialElectronStatus { hit, missing_material, density_domain, energy_domain, invalid };
struct MaterialElectronDraw {
    MaterialElectronStatus status{MaterialElectronStatus::invalid};
    WaterElectronDraw response{};
    std::size_t table_index{};
    double reference_density_g_cm3{};
};
inline MaterialElectronDraw sample_material_electron_response(
    int section,double density,double energy,double density_draw,double cdf_draw,double along,
    const MaterialElectronResponseView* tables,std::size_t count) {
    MaterialElectronDraw out;
    if(!tables || !count || section< -1 || section>=25 || !std::isfinite(density) || density<=0 ||
       !std::isfinite(density_draw) || density_draw<0 || density_draw>1)return out;
    std::size_t first=0;
    while(first<count && tables[first].section<section)++first;
    if(first==count || tables[first].section!=section) {
        out.status=MaterialElectronStatus::missing_material;return out;
    }
    std::size_t end=first+1;
    while(end<count && tables[end].section==section)++end;
    if(density<tables[first].density_g_cm3 || density>tables[end-1].density_g_cm3) {
        out.status=MaterialElectronStatus::density_domain;return out;
    }
    std::size_t hi=first;
    while(hi+1<end && tables[hi].density_g_cm3<density)++hi;
    const std::size_t lo=tables[hi].density_g_cm3==density ? hi : hi-1;
    const double span=tables[hi].density_g_cm3-tables[lo].density_g_cm3;
    if(lo!=hi && !(span>0))return out;
    const double blend=lo==hi ? 0 : (density-tables[lo].density_g_cm3)/span;
    const auto draw=[&](std::size_t i) {
        const auto t=tables[i];
        return sample_water_electron_response(energy,cdf_draw,along,t.channels,t.samples,t.heads,t.channel_count);
    };
    const auto a=draw(lo),b=lo==hi ? a : draw(hi);
    if(!a.valid || !b.valid) {out.status=MaterialElectronStatus::energy_domain;return out;}
    const double left=(1-blend)*a.fraction,right=blend*b.fraction,total=left+right;
    // Interpolate energy fractions first; conditional deposition samples must
    // use fraction-weighted mixing, not simply the density interpolation weight.
    const bool upper=total>0 && right>0 && (left==0 || density_draw>=left/total);
    out.table_index=upper ? hi : lo;
    out.response=upper ? b : a;
    out.response.fraction=total;
    out.response.unresolved=(1-blend)*a.unresolved+blend*b.unresolved;
    out.reference_density_g_cm3=tables[out.table_index].reference_density_g_cm3;
    out.status=MaterialElectronStatus::hit;
    return out;
}
} // namespace carbon
