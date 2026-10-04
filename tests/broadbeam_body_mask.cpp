#include "carbon/broadbeam_body_mask.hpp"
#include <iostream>
#include <chrono>
using namespace carbon;
int main() {
 namespace fs=std::filesystem;
 const auto dir=fs::temp_directory_path()/("body_mask_test_"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));fs::create_directories(dir);
 struct Cleanup{fs::path p;~Cleanup(){fs::remove_all(p);}}cleanup{dir};
 const auto ct=dir/"ct.bin",mask=dir/"body.bin";std::ofstream(ct)<<"original CT";
 std::array<unsigned,3> dims{3,3,3};std::array<float,3> origin{-1,-2,0},spacing{.5,2,.5};
 auto write=[&](unsigned value=1){std::ofstream f(mask,std::ios::binary);f.write("MAIBODY1",8);f.write((char*)dims.data(),12);f.write((char*)origin.data(),12);f.write((char*)spacing.data(),12);const auto h=compute_file_sha256_hex(ct);f.write(h.data(),64);std::array<unsigned char,27> data{};data[13]=value;f.write((char*)data.data(),27);};
 auto check=[](bool x){if(!x)throw std::runtime_error("BODY mask test failed");};
 auto rejects=[&](auto op){bool threw=false;try{op();}catch(const std::runtime_error&){threw=true;}check(threw);};
 write();auto pin=compute_file_sha256_hex(mask);auto valid=load_broadbeam_body_mask(mask,pin,ct,dims,origin,spacing);check(valid[13]==1);
 rejects([&]{load_broadbeam_body_mask(mask,std::string(64,'0'),ct,dims,origin,spacing);});
 rejects([&]{load_broadbeam_body_mask(mask,pin,ct,{1,3,9},origin,spacing);});
 rejects([&]{load_broadbeam_body_mask(mask,pin,ct,dims,origin,{.5,1,.5});});
 std::ofstream(ct)<<"different patient CT";rejects([&]{load_broadbeam_body_mask(mask,pin,ct,dims,origin,spacing);});
 for(unsigned value:{0,2}){write(value);pin=compute_file_sha256_hex(mask);rejects([&]{load_broadbeam_body_mask(mask,pin,ct,dims,origin,spacing);});}
 std::cout<<"PASS BODY mask pins, shape/geometry, patient identity, empty/nonbinary rejection\n";
}
