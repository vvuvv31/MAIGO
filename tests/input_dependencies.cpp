#include "carbon/input_dependencies.hpp"
#include <chrono>
#include <iostream>

int main() {
    using namespace carbon;
    namespace fs=std::filesystem;
    const auto dir=fs::temp_directory_path()/
        ("maigo_input_keys_"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(dir);
    struct Cleanup {fs::path path;~Cleanup(){fs::remove_all(path);}} cleanup{dir};
    auto write=[](const auto& p,const auto& s){std::ofstream(p)<<s;};
    auto check=[](bool ok){if(!ok)throw std::runtime_error("Input cache identity test failed");};
    auto rejected=[&](auto f){bool threw=false;try{f();}catch(const std::exception&){threw=true;}check(threw);};
    write(dir/"bank.bin","bank");write(dir/"rate.bin","rate");
    write(dir/"index.json",R"({"binary":"bank.bin","macroscopic":{"binary":"rate.bin"}})");
    auto key=[&]{return companion_input_key(dir/"index.json",{"binary"});};
    ImmutableInputCache cache;unsigned loads=0;
    auto get=[&]{return cache.get<unsigned>(key(),[&]{return ++loads;});};
    check(get()==1 && get()==1);
    write(dir/"rate.bin","changed rate");check(get()==2);
    write(dir/"bank.bin","changed bank");check(get()==3);
    fs::remove(dir/"bank.bin");rejected(get);
    write(dir/"index.json",R"({"binary":"../outside.bin"})");rejected(key);
    write(dir/"operator.bin","operator");
    write(dir/"electrons.txt","MAIECB02 1\n8 1.0 operator.bin\n");
    auto electron_key=[&]{return electron_bank_input_key(dir/"electrons.txt","MAIECB02");};
    const auto original_electrons=electron_key();
    // Same-sized payload replacement must also invalidate the resident bank.
    const auto old_time=fs::last_write_time(dir/"operator.bin");
    write(dir/"operator.bin","modified");
    fs::last_write_time(dir/"operator.bin",old_time+std::chrono::seconds(1));
    check(electron_key()!=original_electrons);
    fs::remove(dir/"operator.bin");rejected(electron_key);
    write(dir/"electrons.txt","MAIECB02 1\n8 1.0 ../outside.bin\n");rejected(electron_key);
    write(dir/"electrons.txt","MAIECB02 1\n8 -1.0 operator.bin\n");rejected(electron_key);
    write(dir/"electrons.txt","MAIECB02 1\n");rejected(electron_key);
    write(dir/"operator.bin","operator");
    write(dir/"electrons.txt","MAIECB02 1\n8 1.0 operator.bin\nextra\n");rejected(electron_key);
    write(dir/"electrons.txt","MAIPHB02 1\n8 1.0 operator.bin\n");rejected(electron_key);
    fs::create_directory(dir/"series");write(dir/"series/a.dcm","first");
    auto ct=[&]{return ct_input_key(dir/"series",{},"isocenter");};
    const auto one=ct();write(dir/"series/b.dcm","second");const auto two=ct();check(one!=two);
    write(dir/"series/a.dcm","modified first");const auto changed=ct();check(two!=changed);
    check(ct_input_key(dir/"series/a.dcm",{},"isocenter")==changed);
    check(ct_input_key(dir/"series",{},"center")!=changed);
    fs::remove(dir/"series/b.dcm");check(ct()!=changed);
    rejected([&]{ct_input_key(dir/"series/missing.dcm",{},"center");});
    write(dir/"mapping.txt","mapping");const auto mapped=ct_input_key(dir/"series",dir/"mapping.txt","center");
    write(dir/"mapping.txt","new mapping");check(ct_input_key(dir/"series",dir/"mapping.txt","center")!=mapped);
    {
        std::ofstream out(dir/"packed.bin",std::ios::binary);
        const auto magic=CtGrid::magic_value;out.write(reinterpret_cast<const char*>(&magic),sizeof(magic));
    }
    const auto packed=ct_input_key(dir/"packed.bin",{},"center");
    write(dir/"unrelated.txt","ignore");check(ct_input_key(dir/"packed.bin",{},"center")==packed);
    std::cout<<"PASS: companion payload invalidation, nested manifests, missing/traversal rejection, CT series changes/origin/material mapping, packed-file isolation\n";
}
