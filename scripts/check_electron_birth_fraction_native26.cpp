#include "carbon/electron_response26.hpp"
#include "carbon/unified_em_view.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
template<class T>void read(std::ifstream& f,T* p,std::size_t n=1){if(!f.read(reinterpret_cast<char*>(p),n*sizeof(T)))throw std::runtime_error("native yield fixture");}
int main(int argc,char** argv){
    if(argc!=3)return 2;const auto p=carbon::UnifiedEmPackage::load(argv[1]);
    if(p.materials.size()!=26)throw std::runtime_error("require26 ion model");
    carbon::UnifiedEmDevice v;
    v.materials=p.materials.data();v.species=p.species.data();v.records=p.records.data();v.nodes=p.nodes.data();
    v.segments=p.segments.data();v.material_count=26;v.continuous_descriptors=p.continuous_descriptors.data();
    v.continuous_scalars=p.continuous_scalars.data();v.continuous_ranges=p.continuous_ranges.data();
    v.continuous_degree=p.continuous_degree;v.continuous_scalar_count=p.continuous_scalar_count;
    v.continuous_lengths=p.continuous_lengths.data();std::ifstream f(argv[2],std::ios::binary);
    unsigned cases;read(f,&cases);std::cout.precision(12);
    for(unsigned c=0;c<cases;++c){
        int section;float rho;unsigned a,bins;read(f,&section);read(f,&rho);read(f,&a);read(f,&bins);
        const auto s=v.select(section==25?-1:section,rho,v.species_index(a==1?1:6,a));
        if(!s.valid)throw std::runtime_error("composition selection");
        for(unsigned b=0;b<bins;++b){double native,se;read(f,&native);read(f,&se);
            if(b*5<50)continue;double prediction=0;
            // Fine uniform parent-energy quadrature for the slowing bin.
            for(unsigned q=0;q<128;++q){
                const float kinetic=a*(5.f*b+(float(q)+.5f)*5.f/128.f);
                const auto prepared=s.prepare(kinetic);
                const float h=std::min(.001f,.01f*prepared.lo.range);
                const float restricted=s.mean(kinetic,h),delta=prepared.lo.delta_stopping*h;
                const float fraction=carbon::electron_response26::birth_fraction_from_em(restricted,delta);
                if(!std::isfinite(fraction))throw std::runtime_error("fraction domain");prediction+=fraction/128;
            }
            std::cout<<c<<' '<<section<<' '<<rho<<' '<<a<<' '<<b<<' '<<prediction<<' '<<native<<' '<<se<<'\n';
        }
    }
    if(f.peek()!=std::char_traits<char>::eof())throw std::runtime_error("yield fixture trailing bytes");
}
