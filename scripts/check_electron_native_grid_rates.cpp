#include "carbon/electron_native_grid_rates.hpp"
#include <fstream>
#include <iostream>
#include <vector>
int main(int argc,char** argv){
    if(argc!=3)return 2;
    std::ifstream input(argv[1],std::ios::binary);std::ofstream output(argv[2],std::ios::binary);
    auto read=[&]<class T>(T& x){input.read(reinterpret_cast<char*>(&x),sizeof(x));if(!input)throw std::runtime_error("truncated fixture");};
    auto array=[&]<class T>(std::vector<T>& x,unsigned n){x.resize(n);input.read(reinterpret_cast<char*>(x.data()),n*sizeof(T));if(!input)throw std::runtime_error("truncated array");};
    unsigned models;read(models);
    for(unsigned i=0;i<models;++i){
        carbon::electron_composition26::NativeGridRateView v;unsigned coefficients,queries;
        read(v.pieces);read(v.degree);read(v.process);read(coefficients);read(queries);read(v.cut_low);read(v.cut_high);read(v.mass_majorant);
        std::vector<float> bounds,c;std::vector<std::uint32_t> bins,intervals,offsets,pole;
        array(bounds,v.pieces+1);array(bins,v.pieces);array(intervals,v.pieces);array(offsets,v.pieces+1);array(pole,v.pieces);array(c,coefficients);
        v.bounds=bounds.data();v.bins=bins.data();v.intervals=intervals.data();v.offsets=offsets.data();v.pole=pole.data();v.coefficients=c.data();
        if(offsets.back()!=coefficients)return 3;
        for(unsigned j=0;j<queries;++j){float density,cut,energy;read(density);read(cut);read(energy);
            const float value=carbon::electron_composition26::native_grid_rate(v,density,cut,energy);
            if(!std::isfinite(value)||value<0.f||value>density*v.mass_majorant){std::cerr<<"invalid grid-rate case="<<i<<" E="<<energy<<" cut="<<cut<<'\n';return 4;}
            output.write(reinterpret_cast<const char*>(&value),sizeof(value));
        }
    }
    char extra;if(input.read(&extra,1)||!output)return 5;
}
