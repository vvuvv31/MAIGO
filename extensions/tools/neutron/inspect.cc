#include "carbon/neutron_package.hpp"
#include <iostream>
int main(int argc,char** argv){try{if(argc!=2)return 2;auto p=carbon::NeutronPackage::load(argv[1],false);std::cout<<"materials="<<p.materials.size()<<" rates="<<p.rates.size()<<" events="<<p.events.size()<<" products="<<p.products.size()<<" macroscopic_rates="<<p.macroscopic_rates.size()<<" runtime_validated="<<p.runtime_validated<<'\n';return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
