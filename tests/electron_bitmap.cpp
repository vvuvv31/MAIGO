#include "carbon/electron_bitmap_builder.hpp"
#include <random>
#include <iostream>
#include <numeric>
using namespace carbon;
void need(bool b){if(!b)throw std::runtime_error("host test failed");}
int main(){std::mt19937_64 rng(75203);std::uint64_t tested=0;
 for(std::uint32_t rows:{1,2,63,64,65,511,512,513,1024,4097,65537,1000003})for(unsigned scenario=0;scenario<8;++scenario) {
  std::vector<std::uint32_t> candidates;
  for(std::uint32_t r=0;r<rows;++r)if(scenario==0 || (scenario==1?r%511==0:scenario==2?r>rows/2:scenario==3?(r/512)%2==1:scenario==4?r%64==63:scenario==5?rng()%10<3:scenario==6?r==rows-1:false))candidates.push_back(r);
  auto n=candidates.size();std::vector<std::uint64_t> offsets{0,n,2*n,2*n},starts{0,0,n};auto b=ElectronBitmapBuilder::build(rows,offsets,starts,candidates);auto v=b.view();
  for(std::uint64_t k=0;k<n;++k){need(v.select(0,k)==candidates[k]);need(v.select(1,k)==candidates[k]);tested+=2;}
 }
 auto x=electron_graph32({0,UINT32_MAX-1,UINT64_MAX},true);need(x[2]==UINT32_MAX&&x[1]==UINT32_MAX-1);
 for(auto bad:{std::uint64_t{UINT32_MAX},std::uint64_t{UINT32_MAX}+1}){bool rejected=false;try{electron_graph32({bad});}catch(...){rejected=true;}need(rejected);}
 for(auto bad:std::vector<std::vector<std::uint32_t>>{{2,1},{1,1},{1,9}}){bool rejected=false;try{ElectronBitmapBuilder::build(8,{0,2},{0},bad);}catch(...){rejected=true;}need(rejected);}
 std::cout<<"PASS exact rank/selects="<<tested<<" overflow/order rejection passed\n";
}
