#include "carbon/broadbeam_body_mask.hpp"
#include <iostream>
#include <chrono>
using namespace carbon;
int main() {
 namespace fs=std::filesystem;
 const auto dir=fs::temp_directory_path()/("body_mask_test_"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));fs::create_directories(dir);
 struct Cleanup{fs::path p;~Cleanup(){fs::remove_all(p);}}cleanup{dir};
 const auto mask=dir/"body.bin";
 std::array<unsigned,3> dims{3,3,3};std::array<float,3> origin{-1,-2,0},spacing{.5,2,.5};
 auto write=[&](unsigned value=1){std::ofstream f(mask,std::ios::binary);f.write("MAIBODY2",8);f.write((char*)dims.data(),12);f.write((char*)origin.data(),12);f.write((char*)spacing.data(),12);std::array<unsigned char,27> data{};data[13]=value;f.write((char*)data.data(),27);};
 auto check=[](bool x){if(!x)throw std::runtime_error("BODY mask test failed");};
 auto rejects=[&](auto op){bool threw=false;try{op();}catch(const std::runtime_error&){threw=true;}check(threw);};
 auto load=[&]{return load_broadbeam_body_mask(mask,dims,origin,spacing);};
 write();auto valid=load();check(valid.size()==27 && valid[13]==1);
 rejects([&]{load_broadbeam_body_mask(mask,{1,3,9},origin,spacing);});
 rejects([&]{load_broadbeam_body_mask(mask,dims,origin,{.5,1,.5});});
 rejects([&]{load_broadbeam_body_mask(mask,dims,{-2,-2,0},spacing);});
 for(unsigned value:{0,2}){write(value);rejects(load);}
 write();fs::resize_file(mask,fs::file_size(mask)-1);rejects(load);
 write();{std::ofstream f(mask,std::ios::binary|std::ios::app);f.put(0);}rejects(load);
 write();{std::fstream f(mask,std::ios::binary|std::ios::in|std::ios::out);f.write("MAIBODY1",8);}rejects(load);
 std::cout<<"PASS BODY mask shape/geometry, empty/nonbinary, truncated/trailing payload and old-format rejection\n";
}
