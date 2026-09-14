#pragma once
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <iomanip>
namespace carbon {
// Coarse host-stage wall times. Nested scopes must not be summed together.
class RuntimeScope {
    const char* name_;bool active_;
    std::chrono::steady_clock::time_point start_;
public:
    explicit RuntimeScope(const char* name):name_(name),active_(std::getenv("CARBON_RUNTIME_BREAKDOWN")!=nullptr),start_(std::chrono::steady_clock::now()){}
    void finish() {
        if(!active_)return;active_=false;
        const auto end=std::chrono::steady_clock::now();
        std::cerr<<"[runtime-stage] "<<name_<<" seconds="<<std::setprecision(12)
            <<std::chrono::duration<double>(end-start_).count()<<"\n";
    }
    ~RuntimeScope(){finish();}
};
template<class F> decltype(auto) runtime_call(const char* name,F&& call) {
    RuntimeScope scope(name);return call();
}
}
