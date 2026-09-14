from pathlib import Path
import shutil
repo=Path(__file__).resolve().parents[2];root=repo/'scratch/runtime_breakdown_20260914';out=root/'deep_source'
shutil.copytree(root/'fine_source',out)
p=out/'include/carbon/device_phase_probe.hpp';s=p.read_text().replace('data[24+slot]','data[40+slot]');p.write_text(s)
p=out/'src/transport_sycl.cpp';s=p.read_text()
s=s.replace('allocate<std::uint64_t>(48)','allocate<std::uint64_t>(80)').replace('queue.fill(phase_data,std::uint64_t{0},48)','queue.fill(phase_data,std::uint64_t{0},80)')
s=s.replace('host_phase[48]','host_phase[80]').replace('queue.copy(phase_data,host_phase,48)','queue.copy(phase_data,host_phase,80)').replace('i<24;++i)std::cout<<"[phase-cycles]"','i<40;++i)std::cout<<"[phase-cycles]"')
s=s.replace('for(int i=0;i<24;++i)','for(int i=0;i<40;++i)').replace('host_phase[24+i]','host_phase[40+i]')
def one(a,b):
 global s
 assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)
one('                    if ((enable_inelastic || enable_nuclear_elastic) &&','                    step_probe.mark(24);\n                    if ((enable_inelastic || enable_nuclear_elastic) &&')
one('                    // Diagnostic only: CT face clamping above guarantees the','                    step_probe.mark(25);\n                    // Diagnostic only: CT face clamping above guarantees the')
for name,slot,indent in [('primary',26,'                        '),('secondary',28,'                                ')]:
 a=indent+f'if constexpr(CARBON_EM_STEP_CACHE) unified_{name}_pre='
 assert s.count(a)==1;s=s.replace(a,indent+f'step_probe.mark({slot});\n'+a)
 a=indent+f'unified_{name}_rate=unified_{name}_clock.update'
 assert s.count(a)==1;s=s.replace(a,indent+f'step_probe.mark({slot+1});\n'+a)
# Nuclear hazard lookup inside the secondary step only; return to remaining pre-loss work.
start=s.index('step_probe.mark(15);')
a='                            if (schneider_ct_device_ctx.uses_cinel03() && (sec_in_ct || use_unified_water) &&'
pos=s.index(a,start);brace=s.index('{',pos);level=1;end=brace+1
while level:
 if s[end]=='{':level+=1
 elif s[end]=='}':level-=1
 end+=1
s=s[:end]+'\n                            step_probe.mark(31);'+s[end:]
s=s[:pos]+'                            step_probe.mark(30);\n'+s[pos:]
p.write_text(s)
