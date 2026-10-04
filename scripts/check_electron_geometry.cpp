#include "carbon/electron_fp32.hpp"
#include <iostream>
#include <iomanip>
int main(){
    namespace f=carbon::electron_fp32;using C=f::Coordinate;
    unsigned cases=0,wrong=0;
    for(C origin:{C(0),C(-169.1015),C(300)})for(C spacing:{C(.5),C(.02)}){
        constexpr unsigned bins=505;
        for(unsigned i=1;i<bins;++i){const C face=origin+C(i)*spacing;
            for(C p:{std::nextafter(face,-std::numeric_limits<C>::infinity()),face,std::nextafter(face,std::numeric_limits<C>::infinity())})for(float direction:{-1.f,1.f}){
                const auto expected=p<face?i-1:p>face?i:direction<0?i-1:i;
                wrong+=f::directed_cell(p,direction,origin,spacing,bins)!=expected;++cases;
            }
        }
    }
    C p=300;constexpr float delta=1e-8f;
    for(unsigned i=0;i<4096;++i)p+=C(delta);
    const double expected=300+4096*double(delta),error=std::abs(double(p)-expected);
    const float density[1]={1};const std::uint8_t section[1]={12};
    const f::ElectronCtGeometry g{{1,1,1},{300,0,0},{1,1,1},density,section};
    const auto born=f::electron_ct_birth_position(g,{300.5,.5,.5},{1,0,0},.5f,std::nextafter(1.f,0.f));
    const auto material=f::electron_ct_point_material(g,born,{1,0,0});
    const auto narrowed=f::electron_ct_point_material(g,{C(float(born[0])),C(float(born[1])),C(float(born[2]))},{1,0,0});
    const bool pass=!wrong&&material.valid&&(sizeof(C)==4||(!narrowed.valid&&error<1e-9));
    std::cout<<std::setprecision(17)<<"{\"passed\":"<<(pass?"true":"false")<<",\"coordinate_bytes\":"<<sizeof(C)<<",\"face_cases\":"<<cases<<",\"wrong_ownership\":"<<wrong<<",\"birth_material_valid\":"<<material.valid<<",\"narrowed_birth_material_valid\":"<<narrowed.valid<<",\"translation_accumulation_error_mm\":"<<error<<"}\n";
    return pass?0:1;
}
