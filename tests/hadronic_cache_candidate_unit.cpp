#include "carbon/hadronic_cache_candidate.hpp"
#include <cassert>

int main() {
    carbon::HadronicIncreasingCacheCandidate c;
    assert(c.update(250, 1) == 1);
    assert(c.update(220, 2) == 1);
    assert(c.update(200, 3) == 1); // strict Geant4 threshold
    assert(c.update(199, 4) == 4);
    assert(!c.accept(1, .5F));
    assert(!c.valid);
    assert(c.update(198, 2) == 2); // reset after rejected proposal
    assert(c.accept(3, .9F));
    assert(!c.valid);
    assert(c.update(197, 5) == 5);
    assert(c.accept(1, .2F)); // equality accepted
}
