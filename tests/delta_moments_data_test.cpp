#include "carbon/delta_moments_data.hpp"
#include <fstream>
#include <iostream>
#include <chrono>
#include <stdexcept>
int main(int argc,char** argv){
    if(argc!=2)throw std::runtime_error("Expected approved delta table path");
    constexpr std::size_t nodes=25872175;
    auto data=carbon::load_delta_moments(argv[1],carbon::delta_moments_source_sha256,nodes);
    if(data.size()!=2*nodes)throw std::runtime_error("Wrong vector length");
    auto root=std::filesystem::temp_directory_path()/std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    std::filesystem::create_directory(root);
    struct Cleanup{std::filesystem::path p;~Cleanup(){std::error_code e;std::filesystem::remove_all(p,e);}} cleanup{root};
    int checks=1;
    auto reject=[&](auto fn,const char* expected){try{fn();}catch(const std::runtime_error& e){if(std::string(e.what()).find(expected)==std::string::npos)throw;++checks;return;}throw std::runtime_error("Missing rejection");};
    reject([&]{carbon::load_delta_moments(argv[1],"wrong",nodes);},"source-package SHA");
    reject([&]{carbon::load_delta_moments(root/"absent",carbon::delta_moments_source_sha256,nodes);},"build_delta_moments.py");
    reject([&]{carbon::load_delta_moments(argv[1],carbon::delta_moments_source_sha256,nodes-1);},"size/node-count");
    auto corrupt=root/"corrupt.bin";{std::ofstream f(corrupt,std::ios::binary);f.seekp(16+8*nodes-1);f.put(0);}
    reject([&]{carbon::load_delta_moments(corrupt,carbon::delta_moments_source_sha256,nodes);},"SHA256 mismatch");
    std::cout<<"Delta loader checks="<<checks<<" PASS\n";
}
