#include "carbon/unified_em_view.hpp"
#include <iostream>
int main(){
 carbon::UnifiedEmRecord r{};r.node_count=3;
 carbon::UnifiedEmNode n[3]{};n[0].energy=1;n[1].energy=2;n[2].energy=3;
 float packed[]={2,20,4,40,8,80};carbon::UnifiedEmPoint p{};p.record=&r;p.nodes=n;p.delta_means=packed;
 for(float e:{1.f,1.5f,2.f,2.5f,3.f}){float m=0,v=0;p.at(e,&m,&v);float expected=e<=2?2*e:4+4*(e-2);if(std::abs(m-expected)>1e-6 || std::abs(v-10*expected)>1e-5)return 1;}
 std::cout<<"interleaved moment lookup exact/interpolated/endpoints PASS\n";
}
