#include "carbon/sha256.hpp"
#include <chrono>
#include <iostream>
int main(int argc,char** argv) {
    const auto dir=std::filesystem::temp_directory_path()/std::filesystem::path("carbon_sha256_"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(dir);const auto path=dir/"payload";
    unsigned checks=0;
    for(unsigned n:{0u,1u,55u,56u,63u,64u,65u,4095u,4096u,4097u,1048575u,1048576u,1048577u}) {
        std::vector<char> bytes(n);for(unsigned i=0;i<n;++i)bytes[i]=static_cast<char>((i*31u+i/17u)&255);
        {std::ofstream f(path,std::ios::binary);f.write(bytes.data(),bytes.size());}
        if(carbon::compute_file_sha256_hex(path)!=carbon::compute_sha256_hex(bytes.data(),bytes.size()))throw std::runtime_error("SHA mismatch");
        ++checks;
    }
    std::filesystem::remove_all(dir);
    bool rejected=false;try{carbon::compute_file_sha256_hex(dir/"missing");}catch(const std::runtime_error&){rejected=true;}
    if(!rejected)throw std::runtime_error("Missing file accepted");
    if(argc==3 && carbon::compute_file_sha256_hex(argv[1])!=argv[2])throw std::runtime_error("Package SHA mismatch");
    std::cout<<"file SHA256 boundary checks="<<checks<<" passed; missing file rejected\n";
}
