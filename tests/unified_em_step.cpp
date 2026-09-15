#include "carbon/unified_em_view.hpp"
#include <iostream>
int main() {
    carbon::UnifiedEmRecord r{};r.a=12;r.step_fraction=.2f;r.final_range=1.f;
    carbon::UnifiedEmState s;s.host_record=&r;s.density=1;s.valid=true;
    carbon::UnifiedEmStep pre;pre.lo.range=10;pre.hi.range=10;
    auto native=s.step(pre);
    auto check=[](bool x){if(!x)throw std::runtime_error("Step extension guard failed");};
    check(s.research_step(1200,pre,1)==native);
    check(s.research_step(1200,pre,1.25f)==native*1.25f);
    check(s.research_step(239,pre,1.25f)==native);
    s.density=.199f;check(s.research_step(1200,pre,1.25f)==native);s.density=1;
    pre.hi.range=4.99f;check(s.research_step(1200,pre,1.25f)==s.step(pre));
    pre.hi.range=10;pre.lo.range=4.99f;check(s.research_step(1200,pre,1.25f)==s.step(pre));
    std::cout<<"research step guards=6 passed\n";
}
