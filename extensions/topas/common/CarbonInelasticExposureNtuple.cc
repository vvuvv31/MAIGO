// Scorer for CarbonInelasticExposureNtuple

#include "CarbonInelasticExposureNtuple.hh"

#include "CarbonInelasticCaptureProcess.hh"
#include "CarbonInelasticTargetIdentity.hh"

#include "G4Element.hh"
#include "G4Exception.hh"
#include "G4HadronicProcess.hh"
#include "G4HadronicProcessType.hh"
#include "G4Isotope.hh"
#include "G4Material.hh"
#include "G4ParticleDefinition.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4Threading.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"
#include "G4WrapperProcess.hh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

constexpr G4double kEnergyTolerance = 1.0e-9;

G4bool IsFiniteNonNegative(G4double value) {
    return std::isfinite(value) && value >= 0.0;
}

std::uint64_t Fnv1a(const std::string& value) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const auto character : value) {
        hash ^= static_cast<std::uint8_t>(character);
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::string CsvQuote(const G4String& value) {
    std::string result = "\"";
    for (const auto character : std::string(value)) {
        if (character == '"') {
            result += "\"\"";
        } else {
            result += character;
        }
    }
    result += '"';
    return result;
}

}  // namespace

G4int CarbonInelasticExposureNtuple::ReadRequiredInteger(
    TsParameterManager* parameter_manager,
    const G4String& parameter_name,
    const char* label) {
    if (!parameter_manager->ParameterExists(parameter_name)) {
        G4ExceptionDescription description;
        description << "CarbonInelasticExposureNtuple requires parameter "
                    << parameter_name << " (" << label << ").";
        G4Exception("CarbonInelasticExposureNtuple", "MissingParameter",
                    FatalException, description);
        return 0;
    }
    return parameter_manager->GetIntegerParameter(parameter_name);
}

G4double CarbonInelasticExposureNtuple::ReadRequiredDouble(
    TsParameterManager* parameter_manager,
    const G4String& parameter_name,
    const char* label) {
    if (!parameter_manager->ParameterExists(parameter_name)) {
        G4ExceptionDescription description;
        description << "CarbonInelasticExposureNtuple requires parameter "
                    << parameter_name << " (" << label << ").";
        G4Exception("CarbonInelasticExposureNtuple", "MissingParameter",
                    FatalException, description);
        return 0.0;
    }
    return parameter_manager->GetDoubleParameter(parameter_name, "Energy");
}

void CarbonInelasticExposureNtuple::Fatal(
    const char* code, const G4String& message) {
    G4ExceptionDescription description;
    description << message;
    G4Exception("CarbonInelasticExposureNtuple", code, FatalException,
                description);
}

CarbonInelasticExposureNtuple::CarbonInelasticExposureNtuple(
    TsParameterManager* parameter_manager,
    TsMaterialManager* material_manager,
    TsGeometryManager* geometry_manager,
    TsScoringManager* scoring_manager,
    TsExtensionManager* extension_manager,
    G4String scorer_name,
    G4String quantity,
    G4String output_file,
    G4bool is_sub_scorer)
    : TsVNtupleScorer(
          parameter_manager, material_manager, geometry_manager,
          scoring_manager, extension_manager, scorer_name, quantity,
          output_file, is_sub_scorer),
      projectile_z_(ReadRequiredInteger(
          parameter_manager, GetFullParmName("ProjectileZ"), "ProjectileZ")),
      projectile_a_(ReadRequiredInteger(
          parameter_manager, GetFullParmName("ProjectileA"), "ProjectileA")),
      energy_bin_min_mev_per_u_(ReadRequiredDouble(
          parameter_manager, GetFullParmName("EnergyBinMin"),
          "EnergyBinMin") /
                                MeV),
      energy_bin_width_mev_per_u_(ReadRequiredDouble(
          parameter_manager, GetFullParmName("EnergyBinWidth"),
          "EnergyBinWidth") /
                                  MeV),
      energy_bin_count_(ReadRequiredInteger(
          parameter_manager, GetFullParmName("EnergyBinCount"),
          "EnergyBinCount")),
      include_secondaries_(
          parameter_manager->ParameterExists(
              GetFullParmName("IncludeSecondaries")) &&
          parameter_manager->GetBooleanParameter(
              GetFullParmName("IncludeSecondaries"))) {
    if (parameter_manager->ParameterExists(
            GetFullParmName("RequireAuthoritativeCollisionState"))) {
        require_authoritative_collision_state_ =
            parameter_manager->GetBooleanParameter(
                GetFullParmName("RequireAuthoritativeCollisionState"));
    }

    const auto wildcard_projectile = projectile_z_ == 0 && projectile_a_ == 0;
    if ((!wildcard_projectile &&
         (projectile_z_ <= 0 || projectile_a_ < projectile_z_)) ||
        (wildcard_projectile && !include_secondaries_) ||
        (projectile_z_ < 0 || projectile_a_ < 0) ||
        energy_bin_min_mev_per_u_ < 0.0 ||
        !std::isfinite(energy_bin_min_mev_per_u_) ||
        energy_bin_width_mev_per_u_ <= 0.0 ||
        !std::isfinite(energy_bin_width_mev_per_u_) || energy_bin_count_ <= 0) {
        G4ExceptionDescription description;
        description << "Invalid exposure scorer configuration: projectile Z/A="
                    << projectile_z_ << "/" << projectile_a_
                    << ", energy grid min/width/count="
                    << energy_bin_min_mev_per_u_ << "/"
                    << energy_bin_width_mev_per_u_ << "/" << energy_bin_count_;
        G4Exception("CarbonInelasticExposureNtuple", "InvalidParameter",
                    FatalException, description);
    }

    compact_worker_id_ = G4Threading::G4GetThreadId();
    const auto worker_label = compact_worker_id_ < 0
                                  ? std::string("master")
                                  : std::to_string(compact_worker_id_);
    const auto* run_tag = std::getenv("CARBON_CINEL02_RUN_TAG");
    const auto tag = run_tag != nullptr && *run_tag != '\0'
                         ? std::string("_") + run_tag
                         : std::string{};
    compact_output_file_ = std::string(output_file) + ".worker_" + worker_label +
                           tag + ".compact.csv";
    compact_identity_ = Fnv1a(compact_output_file_);

    fNtuple->RegisterColumnS(&record_kind_, "record_kind");
    fNtuple->RegisterColumnI(&run_id_, "run_id");
    fNtuple->RegisterColumnI(&thread_id_, "thread_id");
    fNtuple->RegisterColumnI(&event_id_, "event_id");
    fNtuple->RegisterColumnI(&track_id_, "track_id");
    fNtuple->RegisterColumnI(&projectile_z_row_, "projectile_z");
    fNtuple->RegisterColumnI(&projectile_a_row_, "projectile_a");
    fNtuple->RegisterColumnI(&target_z_, "target_z");
    fNtuple->RegisterColumnI(&target_a_, "target_a");
    fNtuple->RegisterColumnI(&target_isotope_id_, "target_isotope_id");
    fNtuple->RegisterColumnS(&target_isotope_name_, "target_isotope_name");
    fNtuple->RegisterColumnS(&material_name_, "material_name");
    fNtuple->RegisterColumnI(&energy_bin_id_, "energy_bin_id");
    fNtuple->RegisterColumnF(
        &energy_low_mev_per_u_, "energy_low_MeV_per_u", "");
    fNtuple->RegisterColumnF(
        &energy_high_mev_per_u_, "energy_high_MeV_per_u", "");
    fNtuple->RegisterColumnF(&energy_mev_per_u_, "energy_MeV_per_u", "");
    fNtuple->RegisterColumnF(
        &collision_energy_mev_per_u_, "collision_energy_MeV_per_u", "");
    fNtuple->RegisterColumnF(&track_length_mm_, "track_length_mm", "");
    fNtuple->RegisterColumnF(&track_weight_, "track_weight", "");
    fNtuple->RegisterColumnD(
        &weighted_track_length_mm_, "weighted_track_length_mm", "");
    fNtuple->RegisterColumnD(
        &target_number_density_per_mm3_,
        "target_number_density_per_mm3", "");
    fNtuple->RegisterColumnD(
        &target_areal_density_weighted_per_mm2_,
        "target_areal_density_weighted_per_mm2", "");
    fNtuple->RegisterColumnI(&history_count_, "history_count");
    fNtuple->RegisterColumnD(
        &history_weight_sum_, "history_weight_sum", "");
    fNtuple->RegisterColumnD(
        &history_weight_squared_sum_, "history_weight_squared_sum", "");
    fNtuple->RegisterColumnI(
        &censored_history_count_, "censored_history_count");
    fNtuple->RegisterColumnD(
        &censored_history_weight_sum_, "censored_history_weight_sum", "");
    fNtuple->RegisterColumnD(
        &censored_history_weight_squared_sum_,
        "censored_history_weight_squared_sum", "");
    fNtuple->RegisterColumnI(&collision_count_, "collision_count");
    fNtuple->RegisterColumnD(
        &collision_weight_sum_, "collision_weight_sum", "");
    fNtuple->RegisterColumnD(
        &collision_weight_squared_sum_,
        "collision_weight_squared_sum", "");
    fNtuple->RegisterColumnF(
        &source_energy_mev_per_u_, "source_energy_MeV_per_u", "");
    fNtuple->RegisterColumnI(&step_index_, "step_index");
    fNtuple->RegisterColumnI(&collision_sequence_, "collision_sequence");
    fNtuple->RegisterColumnS(&collision_process_, "collision_process");
    fNtuple->RegisterColumnS(
        &collision_energy_source_, "collision_energy_source");
}

CarbonInelasticExposureNtuple::~CarbonInelasticExposureNtuple() {
    try {
        FlushCompactOutput();
    } catch (const std::exception& error) {
        G4ExceptionDescription description;
        description << error.what();
        G4Exception("CarbonInelasticExposureNtuple", "CompactOutputFailure",
                    FatalException, description);
    } catch (...) {
        G4ExceptionDescription description;
        description << "unknown compact exposure output failure";
        G4Exception("CarbonInelasticExposureNtuple", "CompactOutputFailure",
                    FatalException, description);
    }
}

G4bool CarbonInelasticExposureNtuple::IsConfiguredProjectile(
    const G4Track* track) const {
    if (track == nullptr || track->GetDefinition() == nullptr ||
        (!include_secondaries_ && track->GetParentID() != 0)) {
        return false;
    }
    const auto* definition = track->GetDefinition();
    // GenericIon carries the concrete isotope on the tracked dynamic ion;
    // its particle definition can expose zero for GetAtomicMass().  Match
    // the CINEL02 writer's identity fallback so the exposure denominator uses
    // the same source-primary population as the captured raw stream.
    const auto definition_a = definition->GetAtomicMass() > 0
                                  ? definition->GetAtomicMass()
                                  : definition->GetBaryonNumber();
    const auto definition_z = definition->GetAtomicNumber();
    if (definition_z <= 0 || definition_a < definition_z) {
        return false;
    }
    if (projectile_z_ == 0 && projectile_a_ == 0) {
        // V2 cascade rates are currently extracted for charged ions.  Neutral
        // hadrons use their separate transport/rate path and must not be
        // assigned an ion energy-per-u denominator here.
        return include_secondaries_;
    }
    return definition_z == projectile_z_ && definition_a == projectile_a_;
}

G4int CarbonInelasticExposureNtuple::CurrentThreadID() const {
    return std::max(0, G4Threading::G4GetThreadId());
}

CarbonInelasticExposureNtuple::HistoryKey
CarbonInelasticExposureNtuple::KeyFor(const G4Track* track) {
    return std::make_tuple(
        GetRunID(), CurrentThreadID(), GetEventID(), track->GetTrackID());
}

CarbonInelasticExposureNtuple::HistoryState&
CarbonInelasticExposureNtuple::GetOrCreateHistory(const G4Track* track) {
    const auto key = KeyFor(track);
    auto [iterator, inserted] = histories_.try_emplace(key);
    if (inserted) {
        iterator->second.run_id = std::get<0>(key);
        iterator->second.thread_id = std::get<1>(key);
        iterator->second.event_id = std::get<2>(key);
        iterator->second.track_id = std::get<3>(key);
        const auto* definition = track->GetParticleDefinition();
        iterator->second.projectile_z = definition->GetAtomicNumber();
        iterator->second.projectile_a = definition->GetAtomicMass() > 0
                                            ? definition->GetAtomicMass()
                                            : definition->GetBaryonNumber();
        if (iterator->second.projectile_z <= 0 ||
            iterator->second.projectile_a < iterator->second.projectile_z) {
            Fatal("InvalidProjectileIdentity",
                  "configured exposure track has no usable ion Z/A");
        }
    }
    const G4double weight = track->GetWeight();
    if (!std::isfinite(weight) || weight <= 0.0) {
        std::ostringstream message;
        message << "primary history has invalid weight " << weight;
        Fatal("InvalidHistoryWeight", message.str());
    }
    if (!iterator->second.have_weight) {
        iterator->second.history_weight = weight;
        iterator->second.source_energy_mev_per_u =
            track->GetVertexKineticEnergy() / MeV /
            iterator->second.projectile_a;
        iterator->second.have_weight = true;
    }
    return iterator->second;
}

std::vector<CarbonInelasticExposureNtuple::TargetSpec>
CarbonInelasticExposureNtuple::TargetsForMaterial(
    const G4Material* material) const {
    if (material == nullptr) {
        Fatal("MissingMaterial", "primary exposure step has no pre-step material");
        return {};
    }

    const auto* elements = material->GetElementVector();
    const auto* atom_densities = material->GetVecNbOfAtomsPerVolume();
    if (elements == nullptr || atom_densities == nullptr) {
        Fatal("MissingMaterialComposition",
              "primary exposure material has no element composition");
        return {};
    }

    std::map<std::pair<G4int, G4int>, TargetSpec> merged;
    for (std::size_t element_index = 0;
         element_index < material->GetNumberOfElements(); ++element_index) {
        const G4Element* element = (*elements)[element_index];
        const G4double element_density = atom_densities[element_index];
        if (element == nullptr || !IsFiniteNonNegative(element_density) ||
            element_density <= 0.0) {
            continue;
        }

        const auto add_target = [&](G4int z, G4int a, G4int isotope_id,
                                    const G4String& name, G4double fraction) {
            if (z <= 0 || a < z || !std::isfinite(fraction) || fraction <= 0.0) {
                return;
            }
            const G4double density =
                element_density * fraction / (1.0 / mm3);
            if (!std::isfinite(density) || density <= 0.0) {
                return;
            }
            auto& target = merged[std::make_pair(z, a)];
            if (target.z == 0) {
                target.z = z;
                target.a = a;
                target.isotope_id = isotope_id;
                target.name = name;
            }
            target.number_density_per_mm3 += density;
        };

        const auto isotope_count = element->GetNumberOfIsotopes();
        if (isotope_count > 0 && element->GetIsotopeVector() != nullptr &&
            element->GetRelativeAbundanceVector() != nullptr) {
            const auto* isotopes = element->GetIsotopeVector();
            const auto* abundances = element->GetRelativeAbundanceVector();
            for (std::size_t isotope_index = 0;
                 isotope_index < isotope_count; ++isotope_index) {
                const G4Isotope* isotope = (*isotopes)[isotope_index];
                if (isotope == nullptr) {
                    continue;
                }
                add_target(
                    isotope->GetZ(), isotope->GetN(),
                    static_cast<G4int>(isotope->GetIndex()), isotope->GetName(),
                    abundances[isotope_index]);
            }
        } else {
            // Directly constructed Geant4 elements can expose only an
            // effective A.  Preserve that identity instead of inventing a
            // natural-isotope split that the material did not declare.
            add_target(
                element->GetZasInt(),
                static_cast<G4int>(std::lround(element->GetN())), -1,
                element->GetName(), 1.0);
        }
    }

    std::vector<TargetSpec> result;
    result.reserve(merged.size());
    for (const auto& [key, target] : merged) {
        (void)key;
        if (target.number_density_per_mm3 > 0.0) {
            result.push_back(target);
        }
    }
    if (result.empty()) {
        Fatal("EmptyMaterialComposition",
              "primary exposure material contains no positive isotope density");
    }
    return result;
}

const CarbonInelasticExposureNtuple::TargetSpec*
CarbonInelasticExposureNtuple::FindTarget(
    const std::vector<TargetSpec>& targets, G4int z, G4int a) {
    for (const auto& target : targets) {
        if (target.z == z && target.a == a) {
            return &target;
        }
    }
    return nullptr;
}

G4int CarbonInelasticExposureNtuple::EnergyBin(
    G4double energy_mev_per_u) const {
    const G4double maximum =
        energy_bin_min_mev_per_u_ +
        energy_bin_width_mev_per_u_ * energy_bin_count_;
    const G4double tolerance = std::max(
        kEnergyTolerance,
        std::abs(maximum) * 1.0e-9 + energy_bin_width_mev_per_u_ * 1.0e-9);
    if (!std::isfinite(energy_mev_per_u) ||
        energy_mev_per_u < energy_bin_min_mev_per_u_ - tolerance ||
        energy_mev_per_u > maximum + tolerance) {
        std::ostringstream message;
        message << "primary energy " << energy_mev_per_u
                << " MeV/u is outside exposure grid ["
                << energy_bin_min_mev_per_u_ << ", " << maximum << "]";
        Fatal("EnergyOutsideGrid", message.str());
        return -1;
    }
    if (energy_mev_per_u >= maximum - tolerance) {
        return energy_bin_count_ - 1;
    }
    const auto index = static_cast<G4int>(std::floor(
        (std::max(energy_mev_per_u, energy_bin_min_mev_per_u_) -
         energy_bin_min_mev_per_u_) /
        energy_bin_width_mev_per_u_));
    return std::clamp(index, 0, energy_bin_count_ - 1);
}

std::vector<CarbonInelasticExposureNtuple::EnergySegment>
CarbonInelasticExposureNtuple::SplitStep(
    G4double pre_energy_mev_per_u,
    G4double post_energy_mev_per_u,
    G4double step_length_mm) const {
    if (!std::isfinite(step_length_mm) || step_length_mm < 0.0) {
        Fatal("InvalidStepLength", "primary exposure step length is invalid");
        return {};
    }
    if (step_length_mm == 0.0) {
        return {};
    }

    const G4int pre_bin = EnergyBin(pre_energy_mev_per_u);
    const G4int post_bin = EnergyBin(post_energy_mev_per_u);
    if (std::abs(post_energy_mev_per_u - pre_energy_mev_per_u) <
        kEnergyTolerance) {
        return {{pre_bin, step_length_mm}};
    }

    const G4double lower =
        std::min(pre_energy_mev_per_u, post_energy_mev_per_u);
    const G4double upper =
        std::max(pre_energy_mev_per_u, post_energy_mev_per_u);
    std::vector<G4double> boundaries;
    for (G4int index = 0; index <= energy_bin_count_; ++index) {
        const G4double boundary =
            energy_bin_min_mev_per_u_ + index * energy_bin_width_mev_per_u_;
        if (boundary > lower + kEnergyTolerance &&
            boundary < upper - kEnergyTolerance) {
            boundaries.push_back(boundary);
        }
    }
    if (post_energy_mev_per_u < pre_energy_mev_per_u) {
        std::sort(boundaries.begin(), boundaries.end(), std::greater<G4double>());
    }

    std::vector<G4double> points;
    points.reserve(boundaries.size() + 2);
    points.push_back(pre_energy_mev_per_u);
    points.insert(points.end(), boundaries.begin(), boundaries.end());
    points.push_back(post_energy_mev_per_u);

    const G4double total_energy_change =
        std::abs(post_energy_mev_per_u - pre_energy_mev_per_u);
    std::vector<EnergySegment> result;
    result.reserve(points.size() - 1);
    G4double remaining_length = step_length_mm;
    for (std::size_t index = 0; index + 1 < points.size(); ++index) {
        const G4double segment_energy_change =
            std::abs(points[index + 1] - points[index]);
        if (segment_energy_change <= kEnergyTolerance) {
            continue;
        }
        const G4double length =
            index + 2 == points.size()
                ? remaining_length
                : step_length_mm * segment_energy_change / total_energy_change;
        remaining_length -= length;
        const G4double midpoint =
            0.5 * (points[index] + points[index + 1]);
        result.push_back({EnergyBin(midpoint), length});
    }
    if (result.empty()) {
        result.push_back({pre_bin, step_length_mm});
    }
    (void)post_bin;
    return result;
}

CarbonInelasticExposureNtuple::CollisionInfo
CarbonInelasticExposureNtuple::CollisionForStep(const G4Step* step) {
    CollisionInfo result;
    if (step == nullptr || step->GetTrack() == nullptr ||
        step->GetPostStepPoint() == nullptr) {
        return result;
    }
    const G4VProcess* process =
        step->GetPostStepPoint()->GetProcessDefinedStep();
    if (process == nullptr || process->GetProcessType() != fHadronic ||
        process->GetProcessSubType() != fHadronInelastic) {
        return result;
    }
    result.is_inelastic = true;
    result.process_name = process->GetProcessName();
    result.material_name =
        step->GetPreStepPoint() != nullptr &&
                step->GetPreStepPoint()->GetMaterial() != nullptr
            ? step->GetPreStepPoint()->GetMaterial()->GetName()
            : "";

    const auto* capture =
        dynamic_cast<const CarbonInelasticCaptureProcess*>(process);
    if (capture != nullptr) {
        const auto* snapshot = capture->GetLastInputSnapshot();
        if (snapshot == nullptr) {
            // A hadronic process can be selected for a process-defined step
            // and then return an unchanged ParticleChange (for example when
            // the integral cross-section correction rejects the proposal).
            // The capture wrapper marks that call as non-authoritative, so it
            // is a normal non-collision rather than a missing collision state.
            return CollisionInfo{};
        }
        if (snapshot->event_id != static_cast<std::uint64_t>(GetEventID()) ||
            snapshot->track_id !=
                static_cast<std::uint32_t>(step->GetTrack()->GetTrackID())) {
            // A terminal zero-energy ion-inelastic definition step is a
            // Geant4 bookkeeping step, not a collision.  The capture wrapper
            // deliberately leaves its authoritative snapshot invalid for
            // this case, so do not manufacture a pre-step collision record.
            if (step->GetTrack()->GetKineticEnergy() <= 0.0) {
                return CollisionInfo{};
            }
            Fatal("MissingAuthoritativeCollisionState",
                  "CINEL02 collision has no matching pre-delegation track snapshot");
        } else {
            result.energy_mev_per_u = snapshot->collision_energy_MeV_per_u;
            result.energy_source = "capture_input_track";
        }
    } else {
        if (require_authoritative_collision_state_) {
            Fatal("NonAuthoritativeCollisionState",
                  "inelastic collision is not produced by the authoritative "
                  "CarbonInelasticCaptureProcess");
        }
        const auto* pre = step->GetPreStepPoint();
        if (pre == nullptr) {
            Fatal("MissingCollisionPreStep", "inelastic collision has no pre-step point");
        } else {
            const auto* definition = step->GetTrack()->GetParticleDefinition();
            const auto track_a = definition != nullptr && definition->GetAtomicMass() > 0
                                     ? definition->GetAtomicMass()
                                     : (definition != nullptr
                                            ? definition->GetBaryonNumber()
                                            : projectile_a_);
            if (track_a <= 0) {
                Fatal("InvalidProjectileIdentity",
                      "fallback inelastic collision has no usable mass number");
            }
            result.energy_mev_per_u =
                pre->GetKineticEnergy() / MeV / track_a;
            result.energy_source = "step_pre_point_fallback";
        }
    }

    const G4VProcess* selected_process = process;
    if (const auto* wrapper = dynamic_cast<const G4WrapperProcess*>(process)) {
        selected_process = wrapper->GetRegisteredProcess();
    }
    auto* hadronic = const_cast<G4HadronicProcess*>(
        dynamic_cast<const G4HadronicProcess*>(selected_process));
    if (hadronic == nullptr) {
        Fatal("MissingHadronicProcess",
              "inelastic scorer step has no registered G4HadronicProcess");
        return result;
    }
    // The capture wrapper freezes target identity immediately after the
    // delegated collision.  Geant4 may clear the mutable process target state
    // before this scorer is called, especially for secondary proton tracks;
    // re-querying it here would turn a valid collision into a false fatal.
    CarbonCinel02TargetIdentity target;
    if (capture != nullptr) {
        const auto* captured_target = capture->GetLastTargetIdentity();
        if (captured_target == nullptr) {
            Fatal("MissingTargetSnapshot",
                  "authoritative CINEL02 collision has no frozen target identity");
            return result;
        }
        target = *captured_target;
    } else {
        // ResolveCarbonCinel02Target first queries G4HadronicProcess::GetTargetIsotope
        // and only records G4Nucleus Z/A when that pointer is unavailable.
        target = ResolveCarbonCinel02Target(hadronic);
    }
    if (target.z <= 0 || target.a < target.z || target.name.empty()) {
        Fatal("MissingTargetIsotope",
              "inelastic process returned no usable target nucleus identity");
        return result;
    }
    result.target_z = target.z;
    result.target_a = target.a;
    result.target_isotope_id = target.isotope_id;
    result.target_name = target.name;
    if (!target.isotope_pointer_present) {
        ++target_isotope_pointer_missing_count_;
    }
    return result;
}

void CarbonInelasticExposureNtuple::SetCommonRow(
    const HistoryState& state,
    G4int target_z,
    G4int target_a,
    G4int target_isotope_id,
    const G4String& target_name,
    G4int energy_bin,
    G4double energy_mev_per_u) {
    record_kind_.clear();
    run_id_ = state.run_id;
    thread_id_ = state.thread_id;
    event_id_ = state.event_id;
    track_id_ = state.track_id;
    projectile_z_row_ = state.projectile_z;
    projectile_a_row_ = state.projectile_a;
    target_z_ = target_z;
    target_a_ = target_a;
    target_isotope_id_ = target_isotope_id;
    target_isotope_name_ = target_name;
    material_name_ = "none";
    energy_bin_id_ = energy_bin;
    energy_low_mev_per_u_ = static_cast<G4float>(
        energy_bin_min_mev_per_u_ + energy_bin * energy_bin_width_mev_per_u_);
    energy_high_mev_per_u_ = static_cast<G4float>(
        energy_bin_min_mev_per_u_ +
        (energy_bin + 1) * energy_bin_width_mev_per_u_);
    energy_mev_per_u_ = static_cast<G4float>(energy_mev_per_u);
    collision_energy_mev_per_u_ = 0.0F;
    track_length_mm_ = 0.0F;
    track_weight_ = static_cast<G4float>(state.history_weight);
    weighted_track_length_mm_ = 0.0;
    target_number_density_per_mm3_ = 0.0;
    target_areal_density_weighted_per_mm2_ = 0.0;
    history_count_ = 0;
    history_weight_sum_ = 0.0;
    history_weight_squared_sum_ = 0.0;
    censored_history_count_ = 0;
    censored_history_weight_sum_ = 0.0;
    censored_history_weight_squared_sum_ = 0.0;
    collision_count_ = 0;
    collision_weight_sum_ = 0.0;
    collision_weight_squared_sum_ = 0.0;
    source_energy_mev_per_u_ =
        static_cast<G4float>(state.source_energy_mev_per_u);
    step_index_ = 0;
    collision_sequence_ = -1;
    // TOPAS ASCII ntuples are whitespace separated. Keep every string column
    // tokenized even on exposure/history rows; an empty trailing string would
    // otherwise disappear when the row is parsed offline.
    collision_process_ = "none";
    collision_energy_source_ = "none";
}

CarbonInelasticExposureNtuple::CompactCell&
CarbonInelasticExposureNtuple::GetCompactCell(
    G4int projectile_z, G4int projectile_a,
    G4int target_z, G4int target_a, G4int target_isotope_id,
    const G4String& target_name, const G4String& material_name,
    G4int energy_bin, G4double energy_mev_per_u) {
    const CompactCellKey key = std::make_tuple(
        projectile_z, projectile_a, target_z, target_a, energy_bin);
    auto& cell = compact_cells_[key];
    if (cell.energy_bin < 0) {
        cell.projectile_z = projectile_z;
        cell.projectile_a = projectile_a;
        cell.target_z = target_z;
        cell.target_a = target_a;
        cell.target_isotope_id = target_isotope_id;
        cell.target_name = target_name;
        cell.material_name = material_name;
        cell.energy_bin = energy_bin;
        cell.energy_low = energy_bin_min_mev_per_u_ +
                          energy_bin * energy_bin_width_mev_per_u_;
        cell.energy_high = cell.energy_low + energy_bin_width_mev_per_u_;
        cell.energy = energy_mev_per_u;
    }
    return cell;
}

void CarbonInelasticExposureNtuple::FillExposureRow(
    const HistoryState& state,
    const TargetSpec& target,
    G4int energy_bin,
    G4double length_mm,
    G4double track_weight,
    G4int step_index,
    const G4String& material_name) {
    if (length_mm <= 0.0) {
        return;
    }
    const G4double weighted_length = length_mm * track_weight;
    auto& cell = GetCompactCell(
        state.projectile_z, state.projectile_a, target.z, target.a,
        target.isotope_id, target.name, material_name,
        energy_bin, energy_bin_min_mev_per_u_ +
                         (energy_bin + 0.5) * energy_bin_width_mev_per_u_);
    cell.weighted_track_length_mm += weighted_length;
    cell.target_areal_density_weighted_per_mm2 +=
        weighted_length * target.number_density_per_mm3;
    cell.density_sum_per_mm3 += target.number_density_per_mm3;
    ++cell.density_samples;
    (void)state;
    (void)step_index;
}

void CarbonInelasticExposureNtuple::FillCollisionRow(
    const HistoryState& state,
    const CollisionInfo& collision,
    G4int energy_bin,
    G4double collision_weight,
    G4int step_index,
    G4int collision_sequence) {
    auto& cell = GetCompactCell(
        state.projectile_z, state.projectile_a,
        collision.target_z, collision.target_a, collision.target_isotope_id,
        collision.target_name, collision.material_name, energy_bin,
        energy_bin_min_mev_per_u_ +
            (energy_bin + 0.5) * energy_bin_width_mev_per_u_);
    cell.collision_count += 1;
    cell.collision_weight_sum += collision_weight;
    cell.collision_weight_squared_sum += collision_weight * collision_weight;
    cell.collision_energy_weighted_sum += collision.energy_mev_per_u * collision_weight;
    (void)state;
    (void)step_index;
    (void)collision_sequence;
}

void CarbonInelasticExposureNtuple::FillHistoryOutcomeRow(
    const HistoryState& state,
    const TargetSpec& target,
    G4int energy_bin,
    G4bool censored) {
    auto& cell = GetCompactCell(
        state.projectile_z, state.projectile_a, target.z, target.a,
        target.isotope_id, target.name, "none", energy_bin,
        energy_bin_min_mev_per_u_ +
            (energy_bin + 0.5) * energy_bin_width_mev_per_u_);
    cell.history_count += 1;
    cell.history_weight_sum += state.history_weight;
    cell.history_weight_squared_sum += state.history_weight * state.history_weight;
    if (censored) {
        cell.censored_history_count += 1;
        cell.censored_history_weight_sum += state.history_weight;
        cell.censored_history_weight_squared_sum +=
            state.history_weight * state.history_weight;
    }
}

void CarbonInelasticExposureNtuple::FinalizeHistory(HistoryState& state) {
    if (!state.have_weight) {
        return;
    }
    for (const auto& [cell, target] : state.touched_cells) {
        const auto target_key = std::make_pair(target.z, target.a);
        const G4bool censored =
            state.collision_targets.find(target_key) ==
            state.collision_targets.end();
        FillHistoryOutcomeRow(state, target, std::get<0>(cell), censored);
    }
}

void CarbonInelasticExposureNtuple::FinalizeHistoriesForEvent(
    G4int run_id, G4int thread_id, G4int event_id) {
    std::vector<HistoryKey> keys;
    for (const auto& [key, state] : histories_) {
        if (state.run_id == run_id && state.thread_id == thread_id &&
            state.event_id == event_id) {
            keys.push_back(key);
        }
    }
    for (const auto& key : keys) {
        auto iterator = histories_.find(key);
        if (iterator != histories_.end()) {
            FinalizeHistory(iterator->second);
            histories_.erase(iterator);
        }
    }
}

G4bool CarbonInelasticExposureNtuple::ProcessHits(
    G4Step* step, G4TouchableHistory*) {
    if (!fIsActive) {
        ++fSkippedWhileInactive;
        return false;
    }
    if (step == nullptr || !IsConfiguredProjectile(step->GetTrack()) ||
        step->GetPreStepPoint() == nullptr ||
        step->GetPostStepPoint() == nullptr) {
        return false;
    }

    const G4Track* track = step->GetTrack();
    auto& state = GetOrCreateHistory(track);
    const G4Material* material = step->GetPreStepPoint()->GetMaterial();
    const auto targets = TargetsForMaterial(material);
    const auto collision = CollisionForStep(step);
    const G4double pre_energy =
        step->GetPreStepPoint()->GetKineticEnergy() / MeV /
        state.projectile_a;
    const G4double endpoint_energy =
        collision.is_inelastic
            ? collision.energy_mev_per_u
            : step->GetPostStepPoint()->GetKineticEnergy() / MeV /
                  state.projectile_a;
    const auto segments =
        SplitStep(pre_energy, endpoint_energy, step->GetStepLength() / mm);

    for (const auto& segment : segments) {
        if (segment.length_mm <= 0.0) {
            continue;
        }
        for (const auto& target : targets) {
            state.touched_cells.emplace(
                std::make_tuple(segment.bin, target.z, target.a), target);
            FillExposureRow(
                state, target, segment.bin, segment.length_mm,
                track->GetWeight(), track->GetCurrentStepNumber(),
                material != nullptr ? material->GetName() : "");
        }
    }

    if (collision.is_inelastic) {
        const auto* target =
            FindTarget(targets, collision.target_z, collision.target_a);
        if (target == nullptr) {
            std::ostringstream message;
            message << "selected target isotope Z=" << collision.target_z
                    << " A=" << collision.target_a
                    << " is absent from the pre-step material composition";
            Fatal("TargetCompositionMismatch", message.str());
        } else {
            auto collision_with_density = collision;
            collision_with_density.target_number_density_per_mm3 =
                target->number_density_per_mm3;
            const G4int collision_bin = EnergyBin(collision.energy_mev_per_u);
            // A zero-length process step is still a collision observation. It
            // deliberately leaves an exposure gap for the offline compiler to
            // reject rather than fabricating denominator length.
            state.touched_cells.emplace(
                std::make_tuple(collision_bin, target->z, target->a), *target);
            state.had_inelastic_collision = true;
            state.collision_targets.emplace(
                collision.target_z, collision.target_a);
            FillCollisionRow(
                state, collision_with_density, collision_bin,
                track->GetWeight(), track->GetCurrentStepNumber(),
                state.next_collision_sequence++);
        }
    }
    return true;
}

void CarbonInelasticExposureNtuple::UserHookForEndOfTrack(
    const G4Track* track) {
    if (!IsConfiguredProjectile(track)) {
        return;
    }
    const auto key = KeyFor(track);
    auto iterator = histories_.find(key);
    if (iterator == histories_.end()) {
        return;
    }
    FinalizeHistory(iterator->second);
    histories_.erase(iterator);
}

void CarbonInelasticExposureNtuple::UserHookForEndOfEvent() {
    FinalizeHistoriesForEvent(GetRunID(), CurrentThreadID(), GetEventID());
}

void CarbonInelasticExposureNtuple::UserHookForEndOfRun() {
    // TOPAS MT keeps worker scorers alive past the normal run output, so a
    // destructor-only flush can silently lose the worker-local sufficient
    // statistics.  End-of-run is the first lifecycle point at which every
    // primary history has been finalized and the worker can publish exactly
    // its remaining compact cells.
    try {
        FlushCompactOutput();
    } catch (const std::exception& error) {
        G4ExceptionDescription description;
        description << error.what();
        G4Exception("CarbonInelasticExposureNtuple", "CompactOutputFailure",
                    FatalException, description);
    } catch (...) {
        G4ExceptionDescription description;
        description << "unknown compact exposure output failure";
        G4Exception("CarbonInelasticExposureNtuple", "CompactOutputFailure",
                    FatalException, description);
    }
}

void CarbonInelasticExposureNtuple::AbsorbResultsFromWorkerScorer(
    TsVScorer* worker_scorer) {
    auto* worker = dynamic_cast<CarbonInelasticExposureNtuple*>(worker_scorer);
    if (worker == nullptr) {
        G4ExceptionDescription description;
        description << "CINEL02 exposure scorer received an incompatible worker scorer";
        G4Exception("CarbonInelasticExposureNtuple", "CompactOutputFailure",
                    FatalException, description);
        return;
    }
    // TsScoringManager invokes this callback on the master for every worker,
    // but does not invoke the worker's own end-of-run hook.  Publish the
    // worker-local sufficient statistics before the normal ntuple absorb.
    worker->FlushCompactOutput();
    TsVNtupleScorer::AbsorbResultsFromWorkerScorer(worker_scorer);
}

void CarbonInelasticExposureNtuple::FlushCompactOutput() {
    if (compact_cells_.empty()) {
        return;
    }
    const auto mode = compact_output_initialized_
                          ? (std::ios::out | std::ios::app)
                          : (std::ios::out | std::ios::trunc);
    std::ofstream output(compact_output_file_, mode);
    if (!output) {
        throw std::runtime_error(
            "cannot open compact exposure output " + compact_output_file_);
    }
    if (!compact_output_initialized_) {
        output << "record_kind,run_id,thread_id,event_id,track_id,projectile_z,"
                  "projectile_a,target_z,target_a,target_isotope_id,target_isotope_name,"
                  "material_name,energy_bin_id,energy_low_MeV_per_u,"
                  "energy_high_MeV_per_u,energy_MeV_per_u,collision_energy_MeV_per_u,"
                  "track_length_mm,track_weight,weighted_track_length_mm,"
                  "target_number_density_per_mm3,target_areal_density_weighted_per_mm2,"
                  "history_count,history_weight_sum,history_weight_squared_sum,"
                  "censored_history_count,censored_history_weight_sum,"
                  "censored_history_weight_squared_sum,collision_count,"
                  "collision_weight_sum,collision_weight_squared_sum,"
                  "source_energy_MeV_per_u,step_index,collision_sequence,"
                  "collision_process,collision_energy_source\n";
        compact_output_initialized_ = true;
    }

    for (const auto& [key, cell] : compact_cells_) {
        (void)key;
        const auto event_id = compact_identity_;
        const auto track_id = compact_row_id_++;
        const auto density = cell.density_samples > 0
                                 ? cell.density_sum_per_mm3 /
                                       cell.density_samples
                                 : 0.0;
        const auto collision_energy = cell.collision_weight_sum > 0.0
                                          ? cell.collision_energy_weighted_sum /
                                                cell.collision_weight_sum
                                          : 0.0;
        const auto write_row = [&](const char* kind, G4double track_length,
                                   G4double track_weight,
                                   G4double weighted_length,
                                   G4double areal_density, G4int histories,
                                   G4double history_sum,
                                   G4double history_squared,
                                   G4int censored_histories,
                                   G4double censored_sum,
                                   G4double censored_squared,
                                   G4int collisions, G4double collision_sum,
                                   G4double collision_squared,
                                   G4double collision_energy_value,
                                   const char* process,
                                   const char* energy_source) {
            output << kind << ",0," << compact_worker_id_ << "," << event_id
                   << "," << track_id << "," << cell.projectile_z << ","
                   << cell.projectile_a << "," << cell.target_z << ","
                   << cell.target_a << "," << cell.target_isotope_id << ","
                   << CsvQuote(cell.target_name) << ","
                   << CsvQuote(cell.material_name) << "," << cell.energy_bin
                   << "," << std::setprecision(12) << cell.energy_low << ","
                   << cell.energy_high << "," << cell.energy << ","
                   << collision_energy_value << "," << track_length << ","
                   << track_weight << "," << weighted_length << "," << density
                   << "," << areal_density << "," << histories << ","
                   << history_sum << "," << history_squared << ","
                   << censored_histories << "," << censored_sum << ","
                   << censored_squared << "," << collisions << ","
                   << collision_sum << "," << collision_squared << ",0,0,0,"
                   << process << "," << energy_source << "\n";
        };

        if (cell.weighted_track_length_mm > 0.0) {
            // The compact row stores the already weighted length with unit
            // representative weight.  This preserves the exact sufficient
            // statistic while satisfying the row-level contract
            // weighted_length = track_length * track_weight.
            write_row(
                "exposure", cell.weighted_track_length_mm, 1.0,
                cell.weighted_track_length_mm,
                cell.target_areal_density_weighted_per_mm2, 0, 0.0, 0.0, 0,
                0.0, 0.0, 0, 0.0, 0.0, 0.0, "none", "none");
        }
        if (cell.history_count > 0) {
            record_kind_ = "history_outcome";
            write_row(
                "history_outcome", 0.0, 1.0, 0.0, 0.0, cell.history_count,
                cell.history_weight_sum, cell.history_weight_squared_sum,
                cell.censored_history_count,
                cell.censored_history_weight_sum,
                cell.censored_history_weight_squared_sum, 0, 0.0, 0.0, 0.0,
                "none", "none");
        }
        if (cell.collision_count > 0) {
            record_kind_ = "collision";
            write_row(
                "collision", 0.0, 1.0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0,
                0.0, cell.collision_count, cell.collision_weight_sum,
                cell.collision_weight_squared_sum, collision_energy,
                "CINEL02_aggregated", "capture_input_track_aggregated");
        }
    }
    output.flush();
    if (!output) {
        throw std::runtime_error(
            "cannot flush compact exposure output " + compact_output_file_);
    }
    {
        std::ofstream ledger(compact_output_file_ + ".ledger.json",
                             std::ios::out | std::ios::trunc);
        if (!ledger) {
            throw std::runtime_error(
                "cannot open compact exposure ledger " + compact_output_file_);
        }
        ledger << "{\n"
               << "  \"schema\": \"CINEL02_EXPOSURE_TARGET_LEDGER_V1\",\n"
               << "  \"target_isotope_pointer_missing_count\": "
               << target_isotope_pointer_missing_count_ << ",\n"
               << "  \"policy\": \"G4Nucleus Z/A retained; isotope pointer absence is not inferred or dropped\"\n"
               << "}\n";
        ledger.flush();
        if (!ledger) {
            throw std::runtime_error(
                "cannot flush compact exposure ledger " + compact_output_file_);
        }
    }
    compact_cells_.clear();
}

void CarbonInelasticExposureNtuple::Clear() {
    histories_.clear();
    TsVNtupleScorer::Clear();
}
