// Scorer for CarbonSchneiderThinSlabValidationScorer

#include "CarbonSchneiderThinSlabValidationScorer.hh"

#include "G4Event.hh"
#include "G4HadronicProcess.hh"
#include "G4RunManager.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4Threading.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

CarbonSchneiderThinSlabValidationScorer::CarbonSchneiderThinSlabValidationScorer(
    TsParameterManager* parameter_manager,
    TsMaterialManager* material_manager,
    TsGeometryManager* geometry_manager,
    TsScoringManager* scoring_manager,
    TsExtensionManager* extension_manager,
    G4String scorer_name,
    G4String quantity,
    G4String output_file,
    G4bool is_sub_scorer)
    : TsVScorer(
          parameter_manager,
          material_manager,
          geometry_manager,
          scoring_manager,
          extension_manager,
          scorer_name,
          quantity,
          output_file,
          is_sub_scorer) {
    if (fPm->ParameterExists(GetFullParmName("SectionId"))) {
        section_id_ = fPm->GetIntegerParameter(GetFullParmName("SectionId"));
    }
    if (fPm->ParameterExists(GetFullParmName("MaterialName"))) {
        material_name_ = fPm->GetStringParameter(GetFullParmName("MaterialName"));
    }
    if (fPm->ParameterExists(GetFullParmName("NominalEnergyMeVPerU"))) {
        if (fPm->GetTypeOfParameter(GetFullParmName("NominalEnergyMeVPerU")) == "d") {
            nominal_energy_mevu_ = fPm->GetDoubleParameter(GetFullParmName("NominalEnergyMeVPerU"), "Energy") / MeV;
        } else {
            nominal_energy_mevu_ = fPm->GetUnitlessParameter(GetFullParmName("NominalEnergyMeVPerU"));
        }
    }
    if (fPm->ParameterExists(GetFullParmName("SlabTransZ"))) {
        slab_trans_z_mm_ = fPm->GetDoubleParameter(GetFullParmName("SlabTransZ"), "Length") / mm;
    }
    if (fPm->ParameterExists(GetFullParmName("SlabThickness"))) {
        slab_thickness_mm_ = fPm->GetDoubleParameter(GetFullParmName("SlabThickness"), "Length") / mm;
    }
    if (fPm->ParameterExists(GetFullParmName("NumberOfDepthBins"))) {
        num_depth_bins_ = fPm->GetIntegerParameter(GetFullParmName("NumberOfDepthBins"));
    }
    if (fPm->ParameterExists(GetFullParmName("OutputJsonPath"))) {
        output_json_path_ = fPm->GetStringParameter(GetFullParmName("OutputJsonPath"));
    } else {
        output_json_path_ = output_file + ".json";
    }

    if (num_depth_bins_ <= 0) num_depth_bins_ = 5;
    checkpoint_depths_mm_.resize(num_depth_bins_);
    bin_surviving_counts_.assign(num_depth_bins_, 0);
    bin_energy_sums_mevu_.assign(num_depth_bins_, 0.0);
    bin_energy_sq_sums_mevu_.assign(num_depth_bins_, 0.0);
    bin_energy_sample_counts_.assign(num_depth_bins_, 0);

    const G4double step_mm = slab_thickness_mm_ / static_cast<G4double>(num_depth_bins_);
    for (G4int k = 0; k < num_depth_bins_; ++k) {
        checkpoint_depths_mm_[k] = (k + 1) * step_mm;
    }
}

CarbonSchneiderThinSlabValidationScorer::~CarbonSchneiderThinSlabValidationScorer() = default;

G4bool CarbonSchneiderThinSlabValidationScorer::ProcessHits(G4Step* step, G4TouchableHistory*) {
    if (!fIsActive || step == nullptr || step->GetTrack() == nullptr) {
        return false;
    }

    G4Track* track = step->GetTrack();
    // Rule: strictly original primary track (TrackID == 1, ParentID == 0)
    if (track->GetTrackID() != 1 || track->GetParentID() != 0) {
        return false;
    }

    const G4ParticleDefinition* def = track->GetDefinition();
    if (def == nullptr || def->GetAtomicNumber() != 6 || def->GetAtomicMass() != 12) {
        return false;
    }

    const G4int run_id = GetRunID();
    const G4int event_id = GetEventID();

    if (run_id != current_run_id_ || event_id != current_event_id_) {
        current_run_id_ = run_id;
        current_event_id_ = event_id;
        primary_entered_ = false;
        primary_reacted_ = false;
        crossed_checkpoints_.clear();
    }

    const G4double entrance_z = slab_trans_z_mm_ - slab_thickness_mm_ * 0.5;
    const G4double z_pre = (step->GetPreStepPoint()->GetPosition().z() / mm) - entrance_z;
    const G4double z_post = (step->GetPostStepPoint()->GetPosition().z() / mm) - entrance_z;

    if (!primary_entered_) {
        primary_entered_ = true;
        total_entering_primaries_++;
    }

    // Check process defined step
    const G4VProcess* proc = step->GetPostStepPoint()->GetProcessDefinedStep();
    const G4String proc_name = proc ? proc->GetProcessName() : "None";

    bool is_hadronic_inelastic = false;
    if (proc != nullptr) {
        const auto* hp = dynamic_cast<const G4HadronicProcess*>(proc);
        if (hp != nullptr && hp->GetProcessSubType() == fHadronInelastic) {
            is_hadronic_inelastic = true;
        } else if (proc_name == "ionInelastic") {
            is_hadronic_inelastic = true;
        }
    }

    // Process checkpoint crossings for surviving primary:
    // If the particle reacted on this step, it was still unreacted for all checkpoints z_k <= z_post
    if (!primary_reacted_) {
        for (G4int k = 0; k < num_depth_bins_; ++k) {
            const G4double target_depth = checkpoint_depths_mm_[k];
            if (z_pre < target_depth && z_post >= target_depth) {
                if (crossed_checkpoints_.find(k) == crossed_checkpoints_.end()) {
                    crossed_checkpoints_.insert(k);
                    bin_surviving_counts_[k]++;

                    // Only sample continuous energy from steps without hadronic reactions
                    // to prevent post-reaction fragments / destroyed projectile states from contaminating mean_e
                    if (!is_hadronic_inelastic) {
                        const G4double frac = (z_post > z_pre) ? (target_depth - z_pre) / (z_post - z_pre) : 0.0;
                        const G4double pre_e = step->GetPreStepPoint()->GetKineticEnergy() / MeV;
                        const G4double post_e = step->GetPostStepPoint()->GetKineticEnergy() / MeV;
                        const G4double e_cross = pre_e - frac * (pre_e - post_e);
                        const G4double e_mevu = e_cross / 12.0;

                        bin_energy_sums_mevu_[k] += e_mevu;
                        bin_energy_sq_sums_mevu_[k] += (e_mevu * e_mevu);
                        bin_energy_sample_counts_[k]++;
                    }
                }
            }
        }
    }

    // After checkpoint recording, mark reaction state if this step was an inelastic reaction
    if (is_hadronic_inelastic) {
        if (!primary_reacted_) {
            primary_reacted_ = true;
            total_first_inelastic_count_++;
            process_counts_[proc_name]++;

            if (proc_name != "ionInelastic") {
                total_contamination_count_++;
            }

            if (first_interactions_.size() < max_detailed_interactions_) {
                FirstInteractionRecord rec;
                rec.event_id = event_id;
                rec.depth_mm = z_post; // Hadronic reaction occurs at post-step point
                rec.step_pre_energy_mevu = (step->GetPreStepPoint()->GetKineticEnergy() / MeV) / 12.0;
                rec.process_name = proc_name;
                rec.track_id = 1;
                rec.parent_id = 0;
                rec.is_hadronic_inelastic = true;
                first_interactions_.push_back(rec);
            } else {
                first_interaction_sample_overflow_count_++;
            }
        }
    }

    return true;
}

void CarbonSchneiderThinSlabValidationScorer::UserHookForEndOfEvent() {
    primary_entered_ = false;
    primary_reacted_ = false;
    crossed_checkpoints_.clear();
    current_event_id_ = -1;
}

void CarbonSchneiderThinSlabValidationScorer::AbsorbResultsFromWorkerScorer(TsVScorer* worker_scorer) {
    auto* worker = dynamic_cast<CarbonSchneiderThinSlabValidationScorer*>(worker_scorer);
    if (worker == nullptr) return;

    total_entering_primaries_ += worker->total_entering_primaries_;
    total_first_inelastic_count_ += worker->total_first_inelastic_count_;
    total_contamination_count_ += worker->total_contamination_count_;

    for (G4int k = 0; k < num_depth_bins_; ++k) {
        bin_surviving_counts_[k] += worker->bin_surviving_counts_[k];
        bin_energy_sums_mevu_[k] += worker->bin_energy_sums_mevu_[k];
        bin_energy_sq_sums_mevu_[k] += worker->bin_energy_sq_sums_mevu_[k];
        bin_energy_sample_counts_[k] += worker->bin_energy_sample_counts_[k];
    }

    for (const auto& kv : worker->process_counts_) {
        process_counts_[kv.first] += kv.second;
    }

    first_interaction_sample_overflow_count_ += worker->first_interaction_sample_overflow_count_;
    for (const auto& rec : worker->first_interactions_) {
        if (first_interactions_.size() < max_detailed_interactions_) {
            first_interactions_.push_back(rec);
        } else {
            first_interaction_sample_overflow_count_++;
        }
    }
}

void CarbonSchneiderThinSlabValidationScorer::UserHookForEndOfRun() {
    if (G4Threading::IsMasterThread()) {
        WriteResultsToJson();
    }
}

void CarbonSchneiderThinSlabValidationScorer::Output() {
    if (G4Threading::IsMasterThread()) {
        WriteResultsToJson();
    }
}

void CarbonSchneiderThinSlabValidationScorer::WriteResultsToJson() {
    if (output_json_path_.empty() || total_entering_primaries_ <= 0) {
        return;
    }

    std::ofstream out(output_json_path_.c_str());
    if (!out.is_open()) {
        std::cerr << "CarbonSchneiderThinSlabValidationScorer: Failed to open " << output_json_path_ << std::endl;
        return;
    }

    out << std::setprecision(10);
    out << "{\n";
    out << "  \"schema_version\": 1,\n";
    out << "  \"scorer\": \"CarbonSchneiderThinSlabValidationScorer\",\n";
    out << "  \"section_id\": " << section_id_ << ",\n";
    out << "  \"material_name\": \"" << material_name_ << "\",\n";
    out << "  \"nominal_energy_mevu\": " << nominal_energy_mevu_ << ",\n";
    out << "  \"slab_thickness_mm\": " << slab_thickness_mm_ << ",\n";
    out << "  \"num_depth_bins\": " << num_depth_bins_ << ",\n";
    out << "  \"entering_primaries\": " << total_entering_primaries_ << ",\n";
    out << "  \"total_first_inelastic_count\": " << total_first_inelastic_count_ << ",\n";
    out << "  \"contamination_count\": " << total_contamination_count_ << ",\n";
    out << "  \"first_interaction_sample_overflow_count\": " << first_interaction_sample_overflow_count_ << ",\n";

    out << "  \"process_breakdown\": {\n";
    bool first_proc = true;
    for (const auto& kv : process_counts_) {
        if (!first_proc) out << ",\n";
        first_proc = false;
        out << "    \"" << kv.first << "\": " << kv.second;
    }
    out << "\n  },\n";

    out << "  \"depth_checkpoints\": [\n";
    for (G4int k = 0; k < num_depth_bins_; ++k) {
        const G4long n_surv = bin_surviving_counts_[k];
        const G4double s_mc = static_cast<G4double>(n_surv) / static_cast<G4double>(total_entering_primaries_);
        const G4double s_err = std::sqrt((s_mc * (1.0 - s_mc)) / static_cast<G4double>(total_entering_primaries_));

        const G4long n_samples = bin_energy_sample_counts_[k];
        const G4double mean_e = (n_samples > 0) ? (bin_energy_sums_mevu_[k] / n_samples) : 0.0;
        G4double std_e = 0.0;
        if (n_samples > 1) {
            const G4double var = (bin_energy_sq_sums_mevu_[k] - (bin_energy_sums_mevu_[k] * bin_energy_sums_mevu_[k] / n_samples)) / (n_samples - 1);
            if (var > 0.0) std_e = std::sqrt(var);
        }

        out << "    {\n";
        out << "      \"checkpoint_index\": " << k << ",\n";
        out << "      \"depth_mm\": " << checkpoint_depths_mm_[k] << ",\n";
        out << "      \"unreacted_count\": " << n_surv << ",\n";
        out << "      \"survival_fraction\": " << s_mc << ",\n";
        out << "      \"survival_std_err\": " << s_err << ",\n";
        out << "      \"energy_sample_count\": " << n_samples << ",\n";
        out << "      \"mean_energy_mevu\": " << mean_e << ",\n";
        out << "      \"std_energy_mevu\": " << std_e << "\n";
        out << "    }" << (k + 1 < num_depth_bins_ ? ",\n" : "\n");
    }
    out << "  ],\n";

    out << "  \"first_interactions_sample\": [\n";
    for (size_t i = 0; i < first_interactions_.size(); ++i) {
        const auto& rec = first_interactions_[i];
        out << "    {\n";
        out << "      \"event_id\": " << rec.event_id << ",\n";
        out << "      \"depth_mm\": " << rec.depth_mm << ",\n";
        out << "      \"step_pre_energy_mevu\": " << rec.step_pre_energy_mevu << ",\n";
        out << "      \"process_name\": \"" << rec.process_name << "\",\n";
        out << "      \"track_id\": " << rec.track_id << ",\n";
        out << "      \"parent_id\": " << rec.parent_id << ",\n";
        out << "      \"is_hadronic_inelastic\": " << (rec.is_hadronic_inelastic ? "true" : "false") << "\n";
        out << "    }" << (i + 1 < first_interactions_.size() ? ",\n" : "\n");
    }
    out << "  ]\n";
    out << "}\n";

    out.close();
    std::cout << "CarbonSchneiderThinSlabValidationScorer: Successfully wrote " << output_json_path_ << std::endl;
}
