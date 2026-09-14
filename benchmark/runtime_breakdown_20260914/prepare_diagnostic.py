from pathlib import Path
import shutil
root=Path('scratch/runtime_breakdown_20260914');p=root/'diagnostic_source';shutil.copytree(root/'source',p,dirs_exist_ok=True)
(p/'include/carbon/device_phase_probe.hpp').write_text('''#pragma once
#include <sycl/sycl.hpp>
#include <cstdint>
namespace carbon {
// Diagnostic sampled lane elapsed cycles, NOT wall-time or exclusive SM occupancy.
inline std::uint64_t phase_clock() {
    std::uint64_t t=0;
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t) :: "memory");
#endif
    return t;
}
struct DevicePhaseProbe {
    std::uint64_t* data; int slot; bool active; std::uint64_t last;
    DevicePhaseProbe(std::uint64_t* d,int s,bool a):data(d),slot(s),active(a),last(a?phase_clock():0){}
    void flush() {
        if(!active)return;
        const auto now=phase_clock();
        sycl::atomic_ref<std::uint64_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(data[slot]).fetch_add(now-last);
        sycl::atomic_ref<std::uint64_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(data[24+slot]).fetch_add(1);
    }
    void mark(int s){if(active){flush();slot=s;last=phase_clock();}}
    ~DevicePhaseProbe(){flush();}
};
}
''')
f=p/'include/carbon/unified_em_view.hpp';s=f.read_text();s='#include "carbon/device_phase_probe.hpp"\n'+s
# Only the diagnostic clone changes function signatures.
start=s.index('UnifiedEmLoss unified_em_explicit_loss(');end=s.index('\n}',start)+2
part=s[start:end];part=part.replace('Uniform& uniform){','Uniform& uniform,std::uint64_t* phase_data,int phase_offset,bool phase_sample){\n    DevicePhaseProbe loss_probe(phase_data,phase_offset,phase_sample);',1)
part=part.replace('    if(fluctuations && mean<t){','    loss_probe.mark(phase_offset+1);\n    if(fluctuations && mean<t){',1)
part=part.replace('    const bool selected=h>=distance;','    loss_probe.mark(phase_offset+2);\n    const bool selected=h>=distance;',1)
s=s[:start]+part+s[end:]
start=s.index('UnifiedEmLoss unified_em_loss(');end=s.index('\n}',start)+2;part=s[start:end]
part=part.replace('Uniform& uniform){','Uniform& uniform,std::uint64_t* phase_data=nullptr,int phase_offset=0,bool phase_sample=false){',1)
part=part.replace('pre,uniform);','pre,uniform,phase_data,phase_offset,phase_sample);')
s=s[:start]+part+s[end:];f.write_text(s)
f=p/'src/transport_sycl.cpp';s=f.read_text()
def replace(a,b):
 global s
 assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)
replace('    std::uint64_t* sec_step_profile_device=nullptr;','''    auto* phase_data=mem_tracker.allocate<std::uint64_t>(48);
    if(!phase_data)throw std::bad_alloc();
    queue.fill(phase_data,std::uint64_t{0},48).wait_and_throw();
    std::uint64_t* sec_step_profile_device=nullptr;''')
replace('while (energy_MeV > energy_cutoff_MeV && steps < max_primary_steps) {','''while (energy_MeV > energy_cutoff_MeV && steps < max_primary_steps) {
                    DevicePhaseProbe step_probe(phase_data,0,((rng_history ^ (std::uint64_t(steps)*0x9e3779b9u))&2047u)==0);''')
replace('sec_steps < kSecondaryMaxSteps) {','''sec_steps < kSecondaryMaxSteps) {
                            DevicePhaseProbe step_probe(phase_data,8,((frag.rng_stream ^ (std::uint64_t(sec_steps)*0x9e3779b9u))&2047u)==0);''')
replace('auto draw=unified_em_loss(unified_primary_state,','step_probe.mark(1);\n                        auto draw=unified_em_loss(unified_primary_state,')
replace('enable_energy_straggling,unified_primary_pre,unified_uniform);','enable_energy_straggling,unified_primary_pre,unified_uniform,phase_data,16,step_probe.active);')
replace('                    auto local_voxel_deposit_MeV = deposited_MeV;','                    step_probe.mark(2);\n                    auto local_voxel_deposit_MeV = deposited_MeV;')
replace('                    const float seg_dir_x = direction_x;','                    step_probe.mark(3);\n                    const float seg_dir_x = direction_x;')
replace('                    position_x_mm += seg_dir_x * step_mm;','                    step_probe.mark(4);\n                    position_x_mm += seg_dir_x * step_mm;')
replace('                    if ((use_ct_elastic || use_all_elastic) && ct_elastic_this_step &&','                    step_probe.mark(5);\n                    if ((use_ct_elastic || use_all_elastic) && ct_elastic_this_step &&')
replace('auto draw=unified_em_loss(unified_secondary_state,','step_probe.mark(9);\n                                auto draw=unified_em_loss(unified_secondary_state,')
replace('enable_secondary_energy_straggling,unified_secondary_pre,unified_secondary_uniform);','enable_secondary_energy_straggling,unified_secondary_pre,unified_secondary_uniform,phase_data,20,step_probe.active);')
replace('                            const auto post_em_e = sycl::fmax(0.0F, sec_e - dE);','                            step_probe.mark(10);\n                            const auto post_em_e = sycl::fmax(0.0F, sec_e - dE);')
replace('\n                            if (secondary_inelastic) {','\n                            step_probe.mark(12);\n                            if (secondary_inelastic) {')
replace('                            if (!secondary_inelastic) sec_e = post_em_e;','                            step_probe.mark(10);\n                            if (!secondary_inelastic) sec_e = post_em_e;')
replace('                            if (cinel02_should_apply_secondary_mcs(','                            step_probe.mark(11);\n                            if (cinel02_should_apply_secondary_mcs(')
replace('                            if(secondary_elastic && sec_e>energy_cutoff_MeV) {','                            step_probe.mark(13);\n                            if(secondary_elastic && sec_e>energy_cutoff_MeV) {')
replace('    runtime_steps.finish();','''    runtime_steps.finish();
    { std::uint64_t host_phase[48]{};queue.copy(phase_data,host_phase,48).wait_and_throw();
      for(int i=0;i<24;++i)std::cout<<"[phase-cycles] "<<i<<" "<<host_phase[i]<<" "<<host_phase[24+i]<<"\\n"; }
''')
f.write_text(s)
