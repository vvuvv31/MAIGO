// Diagnostic export only: no change to TOPAS physics tables or process settings.
// Expose process state for read-only provenance; access labels do not change layout.
#include <bits/stdc++.h>
#define private public
#define protected public
#include "G4VEnergyLossProcess.hh"
#undef protected
#undef private
#include "CarbonStoppingPowerNtuple.hh"
#include "G4EmCalculator.hh"
#include "G4Material.hh"
#include "G4MaterialCutsCouple.hh"
#include "G4ProductionCutsTable.hh"
#include "G4VEmModel.hh"
#include "G4VEmFluctuationModel.hh"
#include "G4Step.hh"
#include "G4ParticleChangeForLoss.hh"
#include "G4Track.hh"
#include "G4DynamicParticle.hh"
#include "G4SystemOfUnits.hh"
#include "G4LossTableManager.hh"
#include "G4PhysicsTable.hh"
#include "G4PhysicsVector.hh"
#include "G4NistManager.hh"
#include "G4IonisParamMat.hh"
CarbonStoppingPowerNtuple::CarbonStoppingPowerNtuple(
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
          parameter_manager,
          material_manager,
          geometry_manager,
          scoring_manager,
          extension_manager,
          scorer_name,
          quantity,
          output_file,
          is_sub_scorer) {
    fNtuple->RegisterColumnF(&energy_mev_per_u_, "Energy (MeV/u)", "");
    fNtuple->RegisterColumnF(&total_energy_mev_, "Total Kinetic Energy (MeV)", "");
    fNtuple->RegisterColumnD(
        &electronic_dedx_mev_per_mm_, "Electronic dE/dx (MeV/mm)", "");
    fNtuple->RegisterColumnD(&total_dedx_mev_per_mm_, "Table restricted dE/dx (MeV/mm)", "");
    fNtuple->RegisterColumnD(&csda_range_mm_, "Table restricted range (mm)", "");
}

CarbonStoppingPowerNtuple::~CarbonStoppingPowerNtuple() = default;

G4bool CarbonStoppingPowerNtuple::ProcessHits(G4Step* step, G4TouchableHistory*) {
    if (!fIsActive || filled_ || step->GetTrack()->GetParentID()!=0) return false;
    auto* track=step->GetTrack();
    const auto* particle=track->GetDefinition();
    const int z=particle->GetAtomicNumber(),a=particle->GetAtomicMass();
    if(z<=0 || a<=0)throw std::runtime_error("Invalid projectile");
    auto* proc=G4LossTableManager::Instance()->GetEnergyLossProcess(particle);
    if(!proc)throw std::runtime_error("Missing energy loss process");
    // Calculator calls can change static hIoni/He3 charge and mass scaling.
    // Preserve the state used by actual tracking before any calculator query.
    const double native_mass_ratio=proc->massRatio;
    const double native_charge_square_ratio=proc->chargeSqRatio;
    auto* cuts=G4ProductionCutsTable::GetProductionCutsTable();
    std::ofstream catalog("catalog.csv");catalog<<std::setprecision(17);
    catalog<<"material,z,a,mass_MeV,density_g_cm3,cut_MeV,I_MeV,e0_MeV,spin,step_fraction,final_range_mm,linear_loss_limit,is_ion,mass_ratio,process,form_factor_per_MeV,magnetic_moment_square_minus_one,lowest_kinetic_MeV\n";
    for(std::size_t ci=0;ci<cuts->GetTableSize();++ci) {
        const auto* couple=cuts->GetMaterialCutsCouple(ci);
        const auto* material=couple->GetMaterial();
        const std::string name=material->GetName()=="Water_75eV" ? "JEM_W" : std::string(material->GetName());
        if(name.rfind("JEM_",0)!=0)continue;
        std::filesystem::create_directory(name);
        step->GetPreStepPoint()->SetMaterialCutsCouple(couple);
        step->GetPreStepPoint()->SetMaterial(const_cast<G4Material*>(material));
        const double cut=(*cuts->GetEnergyCutsVector(1))[ci];
        if(std::getenv("EM_DISTRIBUTION_PROBE")) {
            if(name!="JEM_W" && name!="JEM_S00D1" && name!="JEM_S12D1" && name!="JEM_S24D0")continue;
            std::ofstream out(name+"/distribution_probe.csv");out<<std::setprecision(17);
            out<<"energy_MeVu,step_mm,mean_loss_MeV,n,mean,moment2,moment4\n";
            for(double eu:{.1,2.,10.,30.,60.,200.}) {
                const double kinetic=eu*a*MeV;
                track->SetKineticEnergy(kinetic);proc->StartTracking(track);
                G4ForceCondition condition=NotForced;proc->PostStepGetPhysicalInteractionLength(*track,0.,&condition);
                G4GPILSelection selection=CandidateForSelection;double safety=0;
                const double native=proc->AlongStepGetPhysicalInteractionLength(*track,0.,DBL_MAX,safety,&selection);
                const double h=std::min(proc->GetRange(kinetic,couple)*.1,native*.2);
                step->SetStepLength(h);track->SetStepLength(h);
                bool old=proc->lossFluctuationFlag;proc->lossFluctuationFlag=false;
                auto* change=static_cast<G4ParticleChangeForLoss*>(proc->AlongStepDoIt(*track,*step));
                const double mean=kinetic-change->GetProposedKineticEnergy();proc->lossFluctuationFlag=old;
                auto* model=proc->currentModel;auto* fluct=model->GetModelOfFluctuations();auto* dp=track->GetDynamicParticle();
                if(proc->isIon)fluct->SetParticleAndCharge(particle,model->GetChargeSquareRatio(particle,material,kinetic));
                const double tmax=model->MaxSecondaryKinEnergy(dp),tcut=std::min(cut,tmax);
                double s1=0,s2=0,s4=0;constexpr int n=50000;
                for(int i=0;i<n;++i){double x=fluct->SampleFluctuations(couple,dp,tcut,tmax,h,mean)/MeV;s1+=x;s2+=x*x;s4+=x*x*x*x;}
                out<<eu<<','<<h/mm<<','<<mean/MeV<<','<<n<<','<<s1/n<<','<<s2/n<<','<<s4/n<<'\n';
            }
            continue;
        }
        if(std::getenv("EM_FINITE_PROBE")) {
            std::ofstream probe(name+"/mean_probe.csv");probe<<std::setprecision(17);
            probe<<"energy_MeVu,step_mm,mean_loss_MeV,native_step_mm,range_mm\n";
            for(double eu:{.01,.1,1.,2.,10.,50.,60.,100.,200.,400.}) {
                const double kinetic=eu*a*MeV;if(kinetic>6000*MeV)continue;
                for(double fraction:{.1,1.,2.}) {
                    track->SetKineticEnergy(kinetic);proc->StartTracking(track);
                    G4ForceCondition condition=NotForced;proc->PostStepGetPhysicalInteractionLength(*track,0.,&condition);
                    G4GPILSelection selection=CandidateForSelection;double safety=0;
                    const double native=proc->AlongStepGetPhysicalInteractionLength(*track,0.,DBL_MAX,safety,&selection);
                    const double range=proc->GetRange(kinetic,couple),h=std::min(range*.9,native*fraction);
                    step->SetStepLength(h);track->SetStepLength(h);
                    bool old=proc->lossFluctuationFlag;proc->lossFluctuationFlag=false;
                    auto* change=static_cast<G4ParticleChangeForLoss*>(proc->AlongStepDoIt(*track,*step));
                    const double loss=kinetic-change->GetProposedKineticEnergy();
                    proc->lossFluctuationFlag=old;
                    probe<<eu<<','<<h/mm<<','<<loss/MeV<<','<<native/mm<<','<<range/mm<<'\n';
                }
            }
            probe.close();if(!probe)throw std::runtime_error("Mean probe write failed");
            continue;
        }
        std::ofstream out(name+"/nodes.csv");out<<std::setprecision(17);
        out<<"energy_MeVu,full_dedx,restricted_dedx,table_dedx,range_mm,lambda_per_mm,factor,correction_per_mm,dispersion_per_mm,universal_dispersion_per_mm,charge_square,model_id,fluctuation_id\n";
        std::ofstream models(name+"/models.csv");models<<"id,model,fluctuation\n";
        std::map<std::string,int> model_ids;
        const double emax=6000.0/a;
        std::vector<double> grid;
        for(int i=0;i<=8192;++i)grid.push_back(std::exp(std::log(1e-6)+(std::log(emax)-std::log(1e-6))*i/8192));
        // Include reference beam energies and model boundary neighbourhoods.
        for(double e:{.01,.1,1.,2.,10.,50.,59.6,60.,100.,150.,200.,250.,300.,350.,400.,500.,600.,800.,1000.})if(e<=emax)grid.push_back(e);
        // Keep the native LS/ICRU correction boundary on both sides.
        // Interpolating across this boundary mixes distinct correction laws.
        const double correction_boundary=2*particle->GetPDGMass()/CLHEP::proton_mass_c2/a;
        for(double relative:{1.-1e-5,1.,1.+1e-5}) {
            const double e=correction_boundary*relative;
            if(e>grid.front() && e<emax)grid.push_back(e);
        }
        std::sort(grid.begin(),grid.end());grid.erase(std::unique(grid.begin(),grid.end()),grid.end());
        G4EmCalculator calc;
        double exported_ratio=0;
        for(double eu:grid) {
            const double kinetic=eu*a*MeV;
            const double full=calc.ComputeElectronicDEDX(kinetic,particle,material);
            const double restricted=calc.ComputeElectronicDEDX(kinetic,particle,material,cut);
            // Reinitialise actual transport state after calculator calls.
            track->SetKineticEnergy(kinetic);
            proc->SetDynamicMassCharge(native_mass_ratio,native_charge_square_ratio);
            proc->StartTracking(track);
            G4ForceCondition condition=NotForced;
            proc->PostStepGetPhysicalInteractionLength(*track,0.,&condition);
            auto* model=proc->currentModel;
            const double sp=proc->GetDEDX(kinetic,couple),range=proc->GetRange(kinetic,couple),lambda=proc->GetLambda(kinetic,couple);
            const double factor=proc->fFactor,q2=model->GetChargeSquareRatio(particle,material,kinetic);
            exported_ratio=proc->massRatio;
            auto* fluct=model->GetModelOfFluctuations();
            if(!fluct)throw std::runtime_error("Missing fluctuation model");
            const std::string key=model->GetName()+"/"+fluct->GetName();
            if(!model_ids.count(key)){int id=model_ids.size();model_ids[key]=id;models<<id<<','<<model->GetName()<<','<<fluct->GetName()<<'\n';}
            const int fid=fluct->GetName()=="IonFluc"?1:((fluct->GetName()=="UniversalFluc" || fluct->GetName()=="UrbanFluc")?0:-1);
            auto* dp=track->GetDynamicParticle();
            const double tmax=model->MaxSecondaryKinEnergy(dp),tcut=std::min(cut,tmax);
            double h=1e-8*mm,loss=sp*h;
            if(proc->isIon)model->CorrectionsAlongStep(couple,dp,h,loss);
            // Charge state for variance follows the selected transport model.
            if(proc->isIon)fluct->SetParticleAndCharge(particle,q2);
            const double variance=fluct->Dispersion(material,dp,tcut,tmax,mm);
            const double charge=particle->GetPDGCharge()/eplus;
            const double uq2=proc->isIon?q2:charge*charge;
            const double uni=(tmax/std::pow(dp->GetBeta(),2)-.5*tcut)*CLHEP::twopi_mc2_rcl2*mm*uq2*material->GetElectronDensity();
            out<<eu<<','<<full/(MeV/mm)<<','<<restricted/(MeV/mm)<<','<<sp/(MeV/mm)<<','<<range/mm<<','<<lambda*mm<<','<<factor<<','<<(loss/h-sp)/(MeV/mm)<<','<<variance/(MeV*MeV)<<','<<uni/(MeV*MeV)<<','<<q2<<','<<model_ids[key]<<','<<fid<<'\n';
        }
        out.close();if(!out)throw std::runtime_error("Node export failed");
        double form_scale=.8426*GeV;
        if(particle->GetPDGSpin()==0 && particle->GetPDGMass()<GeV)form_scale=.736*GeV;
        else if(particle->GetPDGMass()>GeV && z>1)form_scale/=G4NistManager::Instance()->GetA27(z);
        const double ff=2*CLHEP::electron_mass_c2/(form_scale*form_scale);
        const double mag=particle->GetPDGMagneticMoment()*particle->GetPDGMass()/(.5*eplus*CLHEP::hbar_Planck*CLHEP::c_squared);
        catalog<<name<<','<<z<<','<<a<<','<<particle->GetPDGMass()/MeV<<','<<material->GetDensity()/(g/cm3)<<','<<cut/MeV<<','<<material->GetIonisation()->GetMeanExcitationEnergy()/MeV<<','<<material->GetIonisation()->GetEnergy0fluct()/MeV<<','<<particle->GetPDGSpin()<<','<<proc->dRoverRange<<','<<proc->finalRange/mm<<','<<proc->linLossLimit<<','<<proc->isIon<<','<<exported_ratio<<','<<proc->GetProcessName()<<','<<ff*MeV<<','<<mag*mag-1<<','<<proc->lowestKinEnergy/MeV<<'\n';
        std::ofstream raw(name+"/vectors.csv");raw<<std::setprecision(17);
        raw<<"kind,index,x0,x1,y0,y_third,y_twothirds,y1,spline\n";
        int kind=0;
        for(auto* table:{proc->DEDXTable(),proc->RangeTableForLoss(),proc->InverseRangeTable(),proc->LambdaTable()}) {
            if(!table)throw std::runtime_error("Missing raw process table");
            auto* v=(*table)[proc->basedCoupleIndex];
            if(!v)throw std::runtime_error("Missing raw material vector");
            for(std::size_t i=0;i+1<v->GetVectorLength();++i){double x=v->Energy(i),h=v->Energy(i+1)-x;raw<<kind<<','<<i<<','<<x<<','<<x+h<<','<<v->Value(x)<<','<<v->Value(x+h/3)<<','<<v->Value(x+2*h/3)<<','<<v->Value(x+h)<<','<<v->GetSpline()<<'\n';}
            ++kind;
        }
        raw.close();if(!raw)throw std::runtime_error("Vector export failed");
        G4cout<<"[unified-em-export] "<<name<<" Z="<<z<<" A="<<a<<" models="<<model_ids.size()<<G4endl;
    }
    catalog.close();if(!catalog)throw std::runtime_error("Catalog export failed");
    track->SetTrackStatus(fStopAndKill);filled_=true;return true;
}
