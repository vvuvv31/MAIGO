#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <array>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace carbon {
struct SourcePlaneCovariance {
    double sigma_position_mm;
    double sigma_angle_rad;
    double correlation;
};

// Uniform-air Fermi-Eyges moments at the entrance are
// (A0,A1,A2) = (theta^2, L*theta^2/2, L^2*theta^2/3).
// Back-propagate these to the existing source plane: C_xa=-L*A0/2.
// Adding a positive correlation here would double-count the subsequent drift.
inline SourcePlaneCovariance add_upstream_air_covariance(
    SourcePlaneCovariance source, double length_mm, double theta_rms_rad) {
    if (!std::isfinite(length_mm) || !std::isfinite(theta_rms_rad) ||
        length_mm < 0 || theta_rms_rad < 0 ||
        !std::isfinite(source.sigma_position_mm) || !std::isfinite(source.sigma_angle_rad) ||
        !std::isfinite(source.correlation) || source.sigma_position_mm < 0 ||
        source.sigma_angle_rad < 0 || std::abs(source.correlation) > 1)
        throw std::invalid_argument("Invalid upstream air covariance");
    if (length_mm == 0 || theta_rms_rad == 0) return source;
    const double variance = theta_rms_rad * theta_rms_rad;
    const double xx = source.sigma_position_mm * source.sigma_position_mm +
        length_mm * length_mm * variance / 3.;
    const double aa = source.sigma_angle_rad * source.sigma_angle_rad + variance;
    const double xa = source.correlation * source.sigma_position_mm * source.sigma_angle_rad -
        length_mm * variance / 2.;
    const double sx = std::sqrt(xx), sa = std::sqrt(aa);
    return {sx, sa, std::clamp(xa / (sx * sa), -1., 1.)};
}
struct AirEntranceMoments { double xx, xa, aa; };

inline SourcePlaneCovariance add_measured_air_covariance(
    SourcePlaneCovariance source, double L, AirEntranceMoments m) {
    // Validate source through the zero-length identity operation.
    source=add_upstream_air_covariance(source,0.,0.);
    if (!std::isfinite(L) || L<0 || !std::isfinite(m.xx) || !std::isfinite(m.xa) ||
        !std::isfinite(m.aa) || m.xx<0 || m.aa<0 || m.xx*m.aa<m.xa*m.xa)
        throw std::invalid_argument("Invalid measured air covariance");
    const double xx=source.sigma_position_mm*source.sigma_position_mm+m.xx-2*L*m.xa+L*L*m.aa;
    const double aa=source.sigma_angle_rad*source.sigma_angle_rad+m.aa;
    const double xa=source.correlation*source.sigma_position_mm*source.sigma_angle_rad+m.xa-L*m.aa;
    if (aa==0 || xx==0) return source;
    return {std::sqrt(xx),std::sqrt(aa),std::clamp(xa/std::sqrt(xx*aa),-1.,1.)};
}

class AirMomentTable {
    std::vector<double> energies_, lengths_;
    std::vector<AirEntranceMoments> values_;
public:
    static AirMomentTable read(const std::filesystem::path& path) {
        std::ifstream f(path);std::string line;
        if (!std::getline(f,line)) throw std::invalid_argument("Missing air moment CSV");
        if (!line.empty() && line.back()=='\r')line.pop_back();
        if (line!="energy_MeVu,length_mm,position_variance_mm2,position_angle_cov_mm_rad,angle_variance_rad2")
            throw std::invalid_argument("Air moment schema mismatch");
        std::vector<std::array<double,5>> rows;AirMomentTable t;
        while (std::getline(f,line)) {
            if (line.empty()) continue;
            std::replace(line.begin(),line.end(),',',' ');std::istringstream s(line);
            std::array<double,5> r{};for(auto& v:r)if(!(s>>v)||!std::isfinite(v))throw std::invalid_argument("Invalid air moment row");
            std::string extra;if(s>>extra)throw std::invalid_argument("Extra air moment columns");
            if(r[0]<=0 || r[1]<=0 || r[2]<=0 || r[4]<=0 || r[2]*r[4]<=r[3]*r[3])throw std::invalid_argument("Nonphysical air moments");
            rows.push_back(r);t.energies_.push_back(r[0]);t.lengths_.push_back(r[1]);
        }
        auto unique=[](auto& v){std::sort(v.begin(),v.end());v.erase(std::unique(v.begin(),v.end()),v.end());};
        unique(t.energies_);unique(t.lengths_);
        if(t.energies_.size()<2 || t.lengths_.size()<2 || rows.size()!=t.energies_.size()*t.lengths_.size())throw std::invalid_argument("Incomplete air moment grid");
        std::sort(rows.begin(),rows.end());
        for(std::size_t i=0;i<t.energies_.size();++i)for(std::size_t j=0;j<t.lengths_.size();++j) {
            const auto& r=rows[i*t.lengths_.size()+j];
            if(r[0]!=t.energies_[i] || r[1]!=t.lengths_[j])throw std::invalid_argument("Duplicate/missing air moment node");
            t.values_.push_back({r[2],r[3],r[4]});
        }
        return t;
    }
    AirEntranceMoments sample(double E,double L) const {
        auto bracket=[](const auto& v,double q){
            if(!std::isfinite(q) || q<v.front() || q>v.back())throw std::out_of_range("Air moment lookup outside validated grid");
            auto it=std::upper_bound(v.begin(),v.end(),q);
            return std::min(static_cast<std::size_t>(it-v.begin()-1),v.size()-2);
        };
        const auto i=bracket(energies_,E),j=bracket(lengths_,L);
        const double u=(E-energies_[i])/(energies_[i+1]-energies_[i]);
        const double v=(L-lengths_[j])/(lengths_[j+1]-lengths_[j]);
        AirEntranceMoments out{0,0,0};
        for(std::size_t a=0;a<2;++a)for(std::size_t b=0;b<2;++b){
            const double w=(a?u:1-u)*(b?v:1-v);const auto& x=values_[(i+a)*lengths_.size()+j+b];
            out.xx+=w*x.xx;out.xa+=w*x.xa;out.aa+=w*x.aa;
        }
        return out;
    }
};
} // namespace carbon
