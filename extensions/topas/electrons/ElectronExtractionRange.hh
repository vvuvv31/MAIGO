#pragma once
#include <cmath>
#include <cstdlib>
#include <stdexcept>

// Extraction domain only: no transport process, sampling law or production cut
// changes. Stay below pair-production threshold because this operator carries
// electrons and photons, not positrons.
inline double ElectronExtractionMaximumMeV() {
    const char* raw=std::getenv("MAIGO_ELECTRON_MAX_MEV");
    if(!raw)return 0.65;
    char* end=nullptr;
    const double value=std::strtod(raw,&end);
    if(end==raw || *end || !std::isfinite(value) || value<0.65 || value>1.0)
        throw std::invalid_argument("MAIGO_ELECTRON_MAX_MEV must be in [0.65, 1.0]");
    return value;
}
