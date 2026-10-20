#include "cli.hpp"
#include "run_session.hpp"
#include <cstdio>
#include <exception>
#include <iostream>
int main(int argc,char** argv) {
    std::setvbuf(stdout,nullptr,_IOLBF,0);
    try {
        const auto options=carbon::parse_cli(argc,argv);
        if(options.help){carbon::print_usage();return 0;}
        carbon::run_session(options);
        return 0;
    } catch(const std::exception& error) {
        std::cerr<<"carbon_mc: "<<error.what()<<'\n';return 1;
    }
}
