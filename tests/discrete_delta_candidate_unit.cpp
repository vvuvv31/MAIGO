#ifdef NDEBUG
#undef NDEBUG
#endif
#include "carbon/discrete_delta_candidate.hpp"
#include <cassert>
#include <cmath>
#include <random>

int main() {
    using namespace carbon;
    DiscreteDeltaClockCandidate<double> c;
    c.arm(2.);
    assert(c.distance(4.) == .5);
    c.consume(.1, 4., false); // competing nuclear event or voxel face
    assert(std::abs(c.remaining-1.6) < 1e-14);
    c.consume(10., 0., false); // zero-rate region preserves optical depth
    assert(std::abs(c.remaining-1.6) < 1e-14);
    assert(std::abs(c.distance(8.)-.2) < 1e-14);
    c.consume(.2, 8., true); // reset even if post-step proposal rejected
    assert(!c.active);

    auto zero=[](){return 0.;};
    assert(sample_spin_zero_delta_candidate(.1,.09,.5,0.,zero).status==DeltaDrawStatus::below_cut);
    auto bad=[](){return 1.;};
    assert(sample_spin_zero_delta_candidate(.1,1.,.5,0.,bad).status==DeltaDrawStatus::invalid);
    assert(sample_spin_zero_delta_candidate(.1,std::numeric_limits<double>::infinity(),.5,0.,zero).status==DeltaDrawStatus::invalid);
    int ff_calls=0;
    auto veto=[&](){return ++ff_calls==3 ? .9 : 0.;};
    assert(sample_spin_zero_delta_candidate(.1,1.,.5,10.,veto).status==DeltaDrawStatus::form_factor_null);
    int calls=0;
    auto rejected=[&](){return (++calls % 2) ? 0. : .99;};
    assert(sample_spin_zero_delta_candidate(.1,1.,.5,0.,rejected).status==DeltaDrawStatus::exhausted);

    // Independent integration in log-energy. Compare acceptance and moments,
    // including zero-energy form-factor vetoes, not just conditional means.
    for(double ff : {0.,.7}) {
        constexpr double cut=.057, tmax=.7, beta2=.4;
        double integrals[3]={},normalization=0;
        constexpr int bins=100000;
        const double dx=std::log(tmax/cut)/bins;
        for(int i=0;i<bins;++i) {
            const double t=cut*std::exp((i+.5)*dx);
            const double density=(1-beta2*t/tmax)/t;
            normalization+=density*dx;
            const double veto=ff*t>1e-6 ? 1/std::pow(1+ff*t,2) : 1;
            integrals[0]+=density*veto*dx;
            integrals[1]+=density*veto*t*dx;
            integrals[2]+=density*veto*t*t*dx;
        }
        std::mt19937_64 engine(13092026);
        auto uniform=[&](){return std::generate_canonical<double,53>(engine);};
        double sums[3]={};
        constexpr int n=1000000;
        for(int i=0;i<n;++i) {
            auto s=sample_spin_zero_delta_candidate(cut,tmax,beta2,ff,uniform);
            assert(s.status==DeltaDrawStatus::accepted || s.status==DeltaDrawStatus::form_factor_null);
            assert(s.energy_MeV==0 || (s.energy_MeV>=cut && s.energy_MeV<=tmax));
            sums[0]+=s.status==DeltaDrawStatus::accepted;
            sums[1]+=s.energy_MeV;sums[2]+=s.energy_MeV*s.energy_MeV;
        }
        for(int j=0;j<3;++j)
            assert(std::abs(sums[j]/n/(integrals[j]/normalization)-1)<.008);
    }
}
