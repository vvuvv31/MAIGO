#include "carbon/electron_penelope_lambda26.hpp"
#include <fstream>
#include <vector>
#include <iostream>
int main(int argc,char**argv){
 if(argc!=3)return 2;std::ifstream f(argv[1],std::ios::binary);std::ofstream out(argv[2],std::ios::binary);
 auto read=[&]<class T>(T& x){if(!f.read(reinterpret_cast<char*>(&x),sizeof(x)))throw std::runtime_error("fixture truncated");};
 unsigned count;read(count);
 for(unsigned i=0;i<count;++i){float cut;unsigned nr,nt,nq;read(cut);read(nr);read(nt);read(nq);
  if(!nr||!nt||nr>128||nt>128||nq>100000)return 3;
  std::vector<carbon::electron_composition26::PenelopeOscillator> reference(nr),target(nt);float rd[34],td[34];
  f.read(reinterpret_cast<char*>(reference.data()),nr*sizeof(reference[0]));f.read(reinterpret_cast<char*>(target.data()),nt*sizeof(target[0]));
  f.read(reinterpret_cast<char*>(rd),sizeof(rd));f.read(reinterpret_cast<char*>(td),sizeof(td));if(!f)return 4;
  const auto r=carbon::electron_composition26::prepare_penelope_lambda26({reference.data(),nr,rd,34},cut);
  const auto t=carbon::electron_composition26::prepare_penelope_lambda26({target.data(),nt,td,34},cut);if(!r.valid||!t.valid)return 5;
  for(unsigned q=0;q<nq;++q){float e;read(e);float ratio=1.f;
   if(e>2.f*cut){const float a=carbon::electron_composition26::penelope_lambda26(r,e),b=carbon::electron_composition26::penelope_lambda26(t,e);if(a>0.f)ratio=b/a;else if(b>0.f){std::cerr<<"new support without reference support\n";return 6;}}
   if(!std::isfinite(ratio)||ratio<0.f)return 7;out.write(reinterpret_cast<char*>(&ratio),sizeof(ratio));
  }
 }
 char extra;if(f.read(&extra,1)||!out)return 8;
}
