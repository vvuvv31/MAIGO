#pragma once
#include "carbon/material_electron_density.hpp"
#include <type_traits>

namespace carbon {
// Lossless queue representation: reorder all original reference fields to
// use their padding for the two deferred-query inputs. No packed/unaligned
// accesses, bit-field truncation, or changed task capacity.
struct QueuedElectronBirth {
    double fraction{}, energy{}, uniform{};
    std::uint64_t row{}, incoming_edge{std::numeric_limits<std::uint64_t>::max()};
    std::uint32_t source{};
    bool valid{}, reference_valid{}, post{}, independent{};
    // independent=true retains parent energy/CDF uniform for the new bank;
    // source identifies its density node. No old track row is sampled.
    QueuedElectronBirth() = default;
    QueuedElectronBirth(const ElectronBirthDraw& draw, double e, double u)
        : fraction(draw.fraction), energy(e), uniform(u), row(draw.reference.row),
          incoming_edge(draw.reference.incoming_edge), source(draw.reference.source),
          valid(draw.valid), reference_valid(draw.reference.valid), post(draw.reference.post) {}
    ElectronBirthDraw draw() const {
        ElectronBirthDraw out;
        out.valid=valid;out.fraction=fraction;
        out.reference.valid=reference_valid;out.reference.source=source;
        out.reference.row=row;out.reference.incoming_edge=incoming_edge;out.reference.post=post;
        return out;
    }
};
static_assert(sizeof(QueuedElectronBirth) == sizeof(ElectronBirthDraw));
static_assert(std::is_trivially_copyable_v<QueuedElectronBirth>);
// The original sampler's complete domain/support checks, without selecting
// a row. CDF traversal cannot change valid/fraction; it only selects reference.
inline ElectronBirthDraw prepare_electron_birth(double energy, double u,
    const ElectronBirthChannel* channels, std::size_t channel_count,
    const ElectronBirthSample* samples, std::uint64_t sample_count) {
    ElectronBirthDraw out;
    if (!channels || !channel_count || !std::isfinite(energy) ||
        !std::isfinite(u) || u < 0 || u >= 1 || energy < channels[0].low ||
        energy > channels[channel_count-1].high) return out;
    std::size_t lo=0, hi=channel_count;
    while (lo+1<hi) {const auto mid=lo+(hi-lo)/2;if(channels[mid].low<=energy)lo=mid;else hi=mid;}
    const auto c=channels[lo];
    if (energy>c.high || !std::isfinite(c.fraction) || c.fraction<0 || c.fraction>1 ||
        c.offset>sample_count || c.count>sample_count-c.offset) return out;
    out.fraction=c.fraction;
    if (!c.count) {out.valid=c.fraction==0;return out;}
    if (!samples || !(c.fraction>0)) return out;
    out.valid=true;
    return out;
}
inline MaterialElectronBirthDraw prepare_material_electron_birth(
    int section,double density,double energy,double mix_u,double birth_u,
    const MaterialElectronResponseView* tables,std::size_t count,
    MaterialElectronLookupView lookup={}) {
    MaterialElectronBirthDraw out;
    const auto b=material_electron_density_bracket(section,density,tables,count,lookup);
    if(!b.valid || !std::isfinite(mix_u) || mix_u<0 || mix_u>=1)return out;
    auto prepare=[&](std::size_t i) {
        const auto t=tables[i];return prepare_electron_birth(energy,birth_u,t.birth_channels,
            t.channel_count,t.birth_samples,t.birth_sample_count);
    };
    const auto low=prepare(b.low),high=b.low==b.high?low:prepare(b.high);
    if(!low.valid || !high.valid)return out;
    const double a=(1-b.upper_weight)*low.fraction,c=b.upper_weight*high.fraction,total=a+c;
    const bool upper=c>0 && (a==0 || mix_u>=a/total);
    out.table=upper?b.high:b.low;out.birth=upper?high:low;
    out.birth.fraction=total;
    return out;
}
inline ElectronBirthDraw resolve_prepared_electron_birth(
    ElectronBirthDraw prepared,const MaterialElectronResponseView& table,
    double energy,double birth_u) {
    if(!prepared.valid)return prepared;
    const auto resolved=sample_electron_birth(energy,birth_u,table.birth_channels,
        table.channel_count,table.birth_samples,table.birth_sample_count);
    prepared.valid=resolved.valid;
    prepared.reference=resolved.reference;
    return prepared;
}
} // namespace carbon
