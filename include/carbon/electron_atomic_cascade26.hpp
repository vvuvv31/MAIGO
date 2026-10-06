#pragma once
#include <cmath>
#include <cstdint>

namespace carbon::electron_composition26 {
// Density-independent EADL laws exported from the same Geant4 version as
// the native reference. IDs are native vacancy IDs, not density indices.
struct AtomicTransition26 {
    std::uint64_t cdf{};
    float energy{};
    unsigned target{},ejected{};
};
struct AtomicVacancy26 {
    const AtomicTransition26* radiative{};
    const AtomicTransition26* auger{};
    unsigned radiative_count{},auger_count{},z{},id{};
    float binding{},max_photon{},max_electron{};
};
struct AtomicBank26 {
    const AtomicVacancy26* vacancies{};unsigned count{};
    bool active{},auger{},cascade{};
};
inline const AtomicVacancy26* atomic_vacancy26(AtomicBank26 bank,unsigned z,unsigned id) {
    unsigned low=0,high=bank.count;
    const auto key=(z<<16)|id;
    while(low<high){const auto middle=low+(high-low)/2;
        const auto v=bank.vacancies[middle];
        if(((v.z<<16)|v.id)<key)low=middle+1;else high=middle;}
    if(low>=bank.count)return nullptr;
    const auto* v=bank.vacancies+low;return v->z==z&&v->id==id?v:nullptr;
}
inline const AtomicTransition26* atomic_transition26(const AtomicTransition26* t,unsigned n,std::uint64_t bits) {
    if(!n||!t||(t[n-1].cdf!=UINT64_MAX&&bits>=t[n-1].cdf))return nullptr;
    unsigned low=0,high=n-1;
    while(low<high){const auto middle=low+(high-low)/2;
        if(t[middle].cdf==UINT64_MAX||bits<t[middle].cdf)high=middle;else low=middle+1;}
    return t+low;
}
struct AtomicCascade26Result {bool valid{};float local{},photon{},electron{};unsigned photons{},electrons{};};
// The callback sees each accepted native secondary. It can use an energy
// weighted reservoir rather than enqueueing the entire cascade. Under-cut
// transitions stop their own vacancy, exactly as G4UAtomicDeexcitation.
// Breadth-first order preserves the native binding-budget acceptance order.
template<class Uniform,class Emit> inline AtomicCascade26Result sample_atomic_cascade26(
    AtomicBank26 bank,unsigned z,unsigned id,float budget,float electron_cut,float photon_cut,
    Uniform& uniform,Emit emit) {
    AtomicCascade26Result out;
    if(!std::isfinite(budget)||budget<0.f||!(electron_cut>0.f&&photon_cut>0.f))return out;
    out.local=budget;
    if(!bank.active||z<6||z>104||!id||budget==0.f){out.valid=true;return out;}
    if(!bank.vacancies||!bank.count)return out;
    unsigned pending[128];pending[0]=id;unsigned begin=0,end=1;
    while(begin<end){
        const auto* v=atomic_vacancy26(bank,z,pending[begin++]);
        // Outermost vacancies with no transition are a native local budget.
        if(!v)continue;
        if(v->max_photon<photon_cut&&(!bank.auger||v->max_electron<electron_cut))continue;
        const auto draw=uniform();
        const auto* t=atomic_transition26(v->radiative,v->radiative_count,draw.bits);
        int pdg=22;
        if(!t){if(!bank.auger)continue;t=atomic_transition26(v->auger,v->auger_count,uniform().bits);pdg=11;}
        if(!t)continue;
        const float cut=pdg==22?photon_cut:electron_cut;
        if(t->energy<cut)continue;
        if(!(t->energy>0.f)||!std::isfinite(t->energy))return {};
        // Native Penelope filters deexcitation products against its remaining
        // budget after the cascade, but does not undo the generated vacancies.
        if(t->energy<out.local){
            out.local-=t->energy;
            const float cosine=1.f-2.f*float(uniform());
            if(pdg==22){out.photon+=t->energy;++out.photons;}else{out.electron+=t->energy;++out.electrons;}
            emit(pdg,t->energy,cosine);
        }
        if(bank.cascade){
            const unsigned add=t->ejected?2:1;if(end+add>128)return {};
            if(t->target<=v->id||(t->ejected&&t->ejected<=v->id))return {};
            pending[end++]=t->target;if(t->ejected)pending[end++]=t->ejected;
        }else{
            if(end>=128||t->target<=v->id)return {};pending[end++]=t->target;
        }
    }
    out.valid=std::isfinite(out.local)&&out.local>=0.f;
    return out;
}
} // namespace carbon::electron_composition26
