from pathlib import Path
import subprocess,shlex
root=Path('/mnt/sdb/wuwei/MAIGO_review_fix_20260923');out=root/'evidence/review_fix_20260923'
s=(root/'evidence/urban_d51e599_20260923/urban_lifecycle_oracle_r1.cc').read_text()
s=s.replace('#include "G4Box.hh"','#include "G4Box.hh"\n#include "G4UserStackingAction.hh"\n#include "G4StepLimiterPhysics.hh"\n#include "G4UserLimits.hh"')
s=s.replace('constexpr int kCaseCount = 9 * kMaterialCount;','constexpr int kCaseCount = kLossGridNodes * kMaterialCount;')
s=s.replace('  double energy_mev = 0.0;','  double energy_mev = 0.0;\n  double range_mm = 0.0, dedx = 0.0, inverse_mev = 0.0, true_step_mm = 0.0;')
s=s.replace('      new G4PVPlacement(nullptr, G4ThreeVector(x_mm * mm, 0, 0), logical,','      logical->SetUserLimits(new G4UserLimits(0.05*mm));\n      new G4PVPlacement(nullptr, G4ThreeVector(x_mm * mm, 0, 0), logical,')
s=s.replace('PhysicsList() { RegisterPhysics(new G4EmStandardPhysics_option4(0)); }','PhysicsList() { RegisterPhysics(new G4EmStandardPhysics_option4(0)); RegisterPhysics(new G4StepLimiterPhysics()); }')
s=s.replace('    gun_.SetParticleEnergy(kEnergyMeV[energy_index] * MeV);','    const double energy = kLossGridMinMeV * std::pow(kLossGridMaxMeV/kLossGridMinMeV, double(energy_index)/(kLossGridNodes-1));\n    gun_.SetParticleEnergy(energy * MeV);')
a=s.index('    const auto duplicate = std::find_if(');b=s.index('    auto* particle = track->GetDefinition();',a)
s=s[:a]+'    if (track->GetCurrentStepNumber() != 1) return;\n'+s[b:]
a=s.index('    g_msc = msc;');b=s.index('\n  }\n};',a)
s=s[:a]+'''    const double e = pre->GetKineticEnergy();
    const double r = urban->GetRange(particle, e, couple);
    const double d = urban->GetDEDX(particle, e, couple);
    const double inv = urban->GetEnergy(particle, r, couple);
    const double lr = loss->GetRange(e, couple);
    const double ld = loss->GetDEDX(e, couple);
    if (r != lr || d != ld || !(r > 0) || !(d > 0) ||
        !std::isfinite(r+d+inv) || std::abs(inv/e-1) > 2e-5) {
      throw std::runtime_error("actual-step Urban/loss context mismatch");
    }
    if (couple != g_contexts[context_id].logical->GetMaterialCutsCouple())
      throw std::runtime_error("actual step used wrong material-cuts couple");
    g_msc = msc; g_urban = urban; g_bound_loss = loss;
    g_trace.push_back({event_id, track->GetTrackID(), context_id,
        e/MeV, r/mm, d/(MeV/mm), inv/MeV, step->GetStepLength()/mm,
        particle, couple});
    // All observations above follow a real first step. Stop only after they
    // have been captured; no sampler/limiter is called a second time.
    const_cast<G4Track*>(track)->SetTrackStatus(fStopAndKill);
''' +s[b:]
# Retain material/cuts metadata writer; replace every after-run physics lookup.
s=s.replace('VALID_ACTIVE_C12_PROCESS_TRACKING','VALID_ACTIVE_C12_STEP_CONTEXT')
s=s.replace('      << "# physics_list G4EmStandardPhysics_option4\\n"','      << "# physics_list G4EmStandardPhysics_option4\\n"\n      << "# extraction_stage actual_C12_first_step_before_EndTracking\\n"')
a=s.index('    const double r_msc = urban->GetRange(');b=s.index('    if (!(r_msc > 0.0)',a)
s=s[:a]+'''    const auto& row = g_trace.at(i*kMaterialCount +
        (context.name == "water_cut_0p10mm" ? 0 : context.name == "water_cut_0p05mm" ? 1 : 2));
    if (std::abs(row.energy_mev/energy_mev-1) > 1e-12)
      throw std::runtime_error("captured grid energy mismatch");
    const double r_msc = row.range_mm, r_loss = row.range_mm;
    const double d_msc = row.dedx, d_loss = row.dedx;
    const double e_msc = row.inverse_mev, e_loss = row.inverse_mev;
''' +s[b:]
# Avoid old midpoint checks against the inactive process. Monotonic table and
# float round trips are checked later against independently captured steps.
a=s.index('  for (int i = 0; i + 1 < kLossGridNodes; ++i)');b=s.index('  if (max_range_rel >',a)
s=s[:a]+s[b:]
a=s.index('int main(int argc, char** argv)')
s=s[:a]+'''class KillSecondary final : public G4UserStackingAction {
 public:
  G4ClassificationOfNewTrack ClassifyNewTrack(const G4Track* t) override {
    return t->GetParentID() == 0 ? fUrgent : fKill;
  }
};
int main(int argc, char** argv) {
  if (argc != 5) return 2;
  try {
    G4RunManager run;
    run.SetUserInitialization(new Detector());
    run.SetUserInitialization(new PhysicsList());
    run.SetUserAction(new PrimaryGenerator());
    run.SetUserAction(new PassiveTrace());
    run.SetUserAction(new KillSecondary());
    run.Initialize();
    run.BeamOn(kCaseCount);
    if (g_trace.size() != kCaseCount) throw std::runtime_error("incomplete grid tracking");
    for (auto& c : g_contexts) c.couple = c.logical->GetMaterialCutsCouple();
    auto* c12 = G4IonTable::GetIonTable()->GetIon(6,12,0.0);
    for (int i=0;i<3;++i) extract_active_loss_table(argv[i+1],g_contexts[i],c12,g_urban,g_bound_loss);
    std::ofstream trace(argv[4]);
    trace << "event_id,context,E_total_MeV,range_mm,dedx_MeV_mm,inverse_MeV,true_step_mm\\n";
    trace << std::setprecision(17);
    for (const auto& r:g_trace) trace << r.event_id << ',' << r.context_id << ',' << r.energy_mev << ',' << r.range_mm << ',' << r.dedx << ',' << r.inverse_mev << ',' << r.true_step_mm << '\\n';
    std::cout << "ACTIVE_STEP_CONTEXT=PASS tracked_rows=" << g_trace.size() << "\\n";
  } catch (const std::exception& e) { std::cerr << e.what() << '\\n'; return 1; }
}
'''
path=out/'oracle_dense_active_step.cc';path.write_text(s)
flags=shlex.split(subprocess.check_output(['/software/geant4-11.3.2/bin/geant4-config','--cflags','--libs']).decode())
with (out/'oracle_dense_build.log').open('w') as f:
 subprocess.run(['g++',str(path),'-O2','-o',str(out/'oracle_dense_active_step')]+flags,stdout=f,stderr=subprocess.STDOUT,check=True)
print('dense actual-step oracle built')
