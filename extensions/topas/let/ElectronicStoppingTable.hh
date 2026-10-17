#pragma once
#include "G4Material.hh"
#include "G4SystemOfUnits.hh"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

// Immutable, offline native electronic stopping response. Reading/interpolating
// it must never call a Geant4 transport model or touch the random engine.
class ElectronicStoppingTable {
public:
    struct Curve {
        double density{}, excitation{};
        std::vector<double> z, fraction, energy, stopping;
        void CheckMaterial(const G4Material* material) const {
            const auto close = [](double a, double b) {
                return std::abs(a-b) <= 1e-10 * std::max(std::abs(a), std::abs(b));
            };
            if (!close(density, material->GetDensity()/(CLHEP::g/CLHEP::cm3)) ||
                !close(excitation, material->GetIonisation()->GetMeanExcitationEnergy()/CLHEP::eV) ||
                z.size() != material->GetNumberOfElements())
                throw std::runtime_error("LET response material/density/I mismatch: " + material->GetName());
            for (std::size_t i=0; i<z.size(); ++i)
                if (!close(z[i], (*material->GetElementVector())[i]->GetZ()) ||
                    !close(fraction[i], material->GetFractionVector()[i]))
                    throw std::runtime_error("LET response composition mismatch: " + material->GetName());
        }
        double At(double kineticMeV) const {
            if (!std::isfinite(kineticMeV) || kineticMeV < energy.front() || kineticMeV > energy.back())
                throw std::runtime_error("LET response energy outside extracted coverage");
            auto hi = std::upper_bound(energy.begin(), energy.end(), kineticMeV);
            const auto i = std::min(std::size_t(hi-energy.begin()-1), energy.size()-2);
            const auto w = (kineticMeV-energy[i])/(energy[i+1]-energy[i]);
            return stopping[i] + w*(stopping[i+1]-stopping[i]);
        }
    };
    using Key = std::tuple<std::string, std::int32_t, std::int32_t>;
    static std::shared_ptr<const ElectronicStoppingTable> Load(const std::string& path) {
        // Worker scorers share the immutable payload; their pointer caches are
        // local. Serial loading also prevents a partially published response.
        static std::mutex mutex;
        static std::map<std::string, std::weak_ptr<const ElectronicStoppingTable>> cache;
        std::lock_guard<std::mutex> guard(mutex);
        if (auto table=cache[path].lock()) return table;
        auto table=std::shared_ptr<ElectronicStoppingTable>(new ElectronicStoppingTable(path));
        cache[path]=table;
        return table;
    }
    const Curve& Find(const G4Material* material, int z, int a) const {
        auto it=curves_.find({material->GetName(), z, a});
        if (it==curves_.end())
            throw std::runtime_error("Missing LET response: " + material->GetName() +
                                     " Z=" + std::to_string(z) + " A=" + std::to_string(a));
        it->second.CheckMaterial(material);
        return it->second;
    }
private:
    std::map<Key, Curve> curves_;
    template<class T> static T Read(std::istream& file) {
        T value{};file.read(reinterpret_cast<char*>(&value),sizeof(value));
        if (!file) throw std::runtime_error("Truncated LET stopping response");
        return value;
    }
    explicit ElectronicStoppingTable(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        char magic[8];file.read(magic,8);
        if (!file || std::string(magic,8)!="ESLUT001" || Read<std::uint32_t>(file)!=1)
            throw std::runtime_error("Invalid LET stopping response: " + path);
        const auto count=Read<std::uint32_t>(file);
        if (!count || count>10000000) throw std::runtime_error("Invalid LET curve count");
        for (std::uint32_t k=0;k<count;++k) {
            const auto size=Read<std::uint32_t>(file);
            if (!size || size>4096) throw std::runtime_error("Invalid LET material name");
            std::string name(size,'\0');file.read(name.data(),size);
            const auto z=Read<std::int32_t>(file), a=Read<std::int32_t>(file);
            Curve curve;curve.density=Read<double>(file);curve.excitation=Read<double>(file);
            const auto elements=Read<std::uint32_t>(file);
            if (z<1 || a<z || a>300 || !std::isfinite(curve.density) || !(curve.density>0) ||
                !std::isfinite(curve.excitation) || !(curve.excitation>0) || !elements || elements>128)
                throw std::runtime_error("Invalid LET curve identity");
            curve.z.reserve(elements);
            curve.fraction.reserve(elements);
            for (std::uint32_t i=0;i<elements;++i) {
                const auto ez=Read<double>(file), fraction=Read<double>(file);
                if (!std::isfinite(ez) || !(ez>0) || !std::isfinite(fraction) || fraction<0 || fraction>1)
                    throw std::runtime_error("Invalid LET material composition");
                curve.z.push_back(ez);curve.fraction.push_back(fraction);
            }
            const auto nodes=Read<std::uint32_t>(file);
            if (nodes<2 || nodes>1000000) throw std::runtime_error("Invalid LET node count");
            curve.energy.reserve(nodes);
            curve.stopping.reserve(nodes);
            for (std::uint32_t i=0;i<nodes;++i) {
                const auto energy=Read<double>(file), stopping=Read<double>(file);
                if (!std::isfinite(energy) || !(energy>0) || (i && energy<=curve.energy.back()) ||
                    !std::isfinite(stopping) || stopping<0)
                    throw std::runtime_error("Invalid LET response node");
                curve.energy.push_back(energy);curve.stopping.push_back(stopping);
            }
            if (!curves_.emplace(Key{name,z,a},std::move(curve)).second)
                throw std::runtime_error("Duplicate LET response curve");
        }
        if (file.peek()!=std::char_traits<char>::eof())
            throw std::runtime_error("Trailing data in LET response");
    }
};
