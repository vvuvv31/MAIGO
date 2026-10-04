#include "carbon/electron_elastic_batch.hpp"
#include <iostream>
int main(){
 namespace f=carbon::electron_fp32;namespace batch=carbon::electron_elastic_batch_research;
 namespace guard=carbon::electron_range_guard_research;
 std::uint64_t state=0x731cabcdef985ULL,points=0,old_mismatch=0,new_mismatch=0,eligible=0;
 auto uniform=[&](){return float(f::random_draw(state));};
 for(float shift:{-300.f,0.f,300.f})for(float spacing:{.02f,.1f,.5f}){
  float density[2]={1,1};std::uint8_t section[2]={6,6};
  const f::ElectronCtGeometry material{{1,1,2},{-20,-20,shift-20},{40,40,20},density,section};
  const unsigned bins=unsigned(std::round(40/spacing));
  const f::ElectronCtGeometry scoring{{bins,bins,bins},{-20,-20,shift-20},{spacing,spacing,spacing},nullptr,nullptr};
  for(unsigned i=0;i<65536;++i){
   f::Position p{};
   for(unsigned k=0;k<3;++k){
    const unsigned cell=std::min(bins-1,unsigned(uniform()*bins));
    const float face=scoring.origin[k]+cell*spacing;
    p[k]=face+spacing*uniform();
    if(i%4==1)p[k]=std::nextafter(face,INFINITY);
    if(i%4==2)p[k]=std::nextafter(face,-INFINITY);
    if(i%4==3)p[k]=face;
   }
   const auto m=f::next_charged_face(material,p,{0,0,1});if(!m.valid||m.escaped)continue;
   const auto box=batch::bounds(material,scoring,p,m.voxel);if(!box.valid)continue;
   const auto target=guard::scoring_index(scoring,p);
   float raw=INFINITY,old_margin=0;
   for(unsigned k=0;k<3;++k){raw=std::min(raw,std::min(p[k]-box.low[k],box.high[k]-p[k]));old_margin=std::max(old_margin,4*std::numeric_limits<float>::epsilon()*std::max(1.f,std::abs(p[k])));}
   const float old_radius=std::max(0.f,raw-old_margin);
   eligible+=box.safety>0;
   for(unsigned mode=0;mode<2;++mode){const float radius=mode?box.safety:old_radius;if(!(radius>0))continue;
    for(unsigned axis=0;axis<3;++axis)for(float sign:{-1.f,1.f}){
     auto q=p;q[axis]+=sign*radius;++points;
     const bool bad=guard::scoring_index(scoring,q)!=target||guard::scoring_index(material,q)!=m.voxel||!batch::contains(box,q);
     if(mode)new_mismatch+=bad;else old_mismatch+=bad;
    }
   }
  }
 }
 std::cout<<"{\"points\":"<<points<<",\"eligible_centers\":"<<eligible<<",\"old_guard_mismatch\":"<<old_mismatch<<",\"origin_aware_mismatch\":"<<new_mismatch<<"}\n";
 return new_mismatch||!old_mismatch||!eligible?1:0;
}
