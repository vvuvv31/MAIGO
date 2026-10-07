#include "carbon/cli.hpp"
#include <charconv>
#include <iostream>
#include <stdexcept>
#include <string_view>
namespace carbon {
namespace {
std::uint64_t integer(std::string_view value) {
    std::uint64_t n{};
    const auto [end,error]=std::from_chars(value.data(),value.data()+value.size(),n);
    if(error!=std::errc{} || end!=value.data()+value.size())
        throw std::invalid_argument("Invalid unsigned integer: "+std::string(value));
    return n;
}
}
CliState parse_cli(int argc,char** argv) {
    CliState state;
    for(int i=1;i<argc;++i) {
        const std::string_view key=argv[i];
        if(key=="--help" || key=="-h"){state.help=true;continue;}
        if(i+1==argc)throw std::invalid_argument("Missing value for "+std::string(key));
        const std::string_view value=argv[++i];
        if(key=="--config")state.config_path=value;
        else if(key=="--batch-count")state.batch_count=integer(value);
        else if(key=="--histories-per-batch")state.histories_per_batch=integer(value);
        else if(key=="--batch-output")state.batch_output=value;
        else if(key=="--histories"){state.histories=integer(value);state.histories_overridden=true;}
        else if(key=="--seed"){state.seed=integer(value);state.seed_overridden=true;}
        else throw std::invalid_argument("Unknown argument: "+std::string(key));
    }
    const bool batches=state.batch_count || state.histories_per_batch || !state.batch_output.empty();
    if(batches && (!state.batch_count || !state.histories_per_batch || state.batch_output.empty() || state.histories_overridden))
        throw std::invalid_argument("Use --batch-count N --histories-per-batch H --batch-output NEW_DIR together; do not combine with --histories");
    if(state.histories_overridden && !state.histories)throw std::invalid_argument("Histories must be positive");
    return state;
}
void print_usage(){std::cout<<"carbon_mc --config config/<preset>.yaml [--seed N] [--histories N]\n"
    <<"carbon_mc --config config/<preset>.yaml --batch-count N --histories-per-batch H --batch-output NEW_DIR\n";}
}
