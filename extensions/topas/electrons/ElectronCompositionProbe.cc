#include "ElectronCompositionProbe.hh"
#include "TsParameterManager.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4DynamicParticle.hh"
#include "G4Electron.hh"
#include "G4Gamma.hh"
#include "G4ProcessManager.hh"
#include "G4VEnergyLossProcess.hh"
#include "G4VEmProcess.hh"
#include "G4VEmModel.hh"
#include "G4ParticleChangeForLoss.hh"
#include "G4ParticleChangeForGamma.hh"
#include "G4ProductionCutsTable.hh"
#include "G4Material.hh"
#include "G4Element.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicsTable.hh"
#include "G4PhysicsVector.hh"
#include "G4PenelopeOscillatorManager.hh"
#include "G4PenelopeOscillator.hh"
#include "G4PenelopeIonisationXSHandler.hh"
#include "G4PenelopeCrossSection.hh"
#include "G4AtomicTransitionManager.hh"
#include "G4AtomicShell.hh"
#include "G4FluoTransition.hh"
#include "G4AugerTransition.hh"
#include "G4LossTableManager.hh"
#include "G4VAtomDeexcitation.hh"
#include "G4UAtomicDeexcitation.hh"
#include <set>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <vector>
namespace {
struct Row {
    std::int32_t run{},event{},step{1},kind{},pdg{},process{};
    double energy{},remaining{},deposit{},children{},pre[3]{0,0,1},post[3]{},length{},density{};
};
static_assert(sizeof(Row)==120);
}
ElectronCompositionProbe::ElectronCompositionProbe(TsParameterManager* pm,TsMaterialManager* mm,
    TsGeometryManager* gm,TsScoringManager* sm,TsExtensionManager* em,G4String name,
    G4String quantity,G4String output,G4bool sub):TsVScorer(pm,mm,gm,sm,em,name,quantity,output,sub) {}
void ElectronCompositionProbe::Require(bool ok,const char* why) {
    if(!ok){G4cerr<<"ElectronCompositionProbe audit: "<<why<<G4endl;fPm->AbortSession(1);}
}
G4bool ElectronCompositionProbe::ProcessHits(G4Step* source,G4TouchableHistory*) {
    if(done_||source->GetTrack()->GetParentID()!=0)return false;done_=true;
    Require(source->GetTrack()->GetDefinition()==G4Gamma::Gamma(),"photon trigger required");
    const auto* material=source->GetPreStepPoint()->GetMaterial();
    const auto* couple=source->GetPreStepPoint()->GetMaterialCutsCouple();
    G4VEnergyLossProcess* charged[2]{};G4VEmProcess* photon[3]{};
    const char* cn[]={"eIoni","eBrem"};const char* pn[]={"phot","compt","Rayl"};
    const auto* cl=G4Electron::Electron()->GetProcessManager()->GetProcessList();
    for(std::size_t j=0;j<cl->size();++j){auto* p=(*cl)[j];
        for(int i=0;i<2;++i)if(p->GetProcessName()==cn[i])charged[i]=dynamic_cast<G4VEnergyLossProcess*>(p);}
    const auto* pl=G4Gamma::Gamma()->GetProcessManager()->GetProcessList();
    for(std::size_t j=0;j<pl->size();++j)if(auto* em=dynamic_cast<G4VEmProcess*>((*pl)[j]))
        for(int i=0;i<3;++i)if(auto* q=em->GetEmProcess(pn[i]))photon[i]=q;
    for(auto* p:charged)Require(p,"missing charged process");for(auto* p:photon)Require(p,"missing photon process");
    auto* cuts=G4ProductionCutsTable::GetProductionCutsTable();
    auto& electronCuts=*const_cast<std::vector<double>*>(cuts->GetEnergyCutsVector(1));
    auto& photonCuts=*const_cast<std::vector<double>*>(cuts->GetEnergyCutsVector(0));
    const auto idx=couple->GetIndex();const double oldE=electronCuts[idx],oldP=photonCuts[idx];
    std::ifstream requests("requests.txt");Require(bool(requests),"missing requests");
    std::ofstream raw("conditional.phsp",std::ios::binary),audit("composition_audit.csv");
    audit<<std::setprecision(17)<<"cell,kind,process,energy_MeV,electron_cut_MeV,photon_cut_MeV,density_g_cm3,model_rate_per_mm,table_rate_per_mm,requested,accepted,max_energy_residual_MeV,model\n";
    int cell,kind,proc,count,event=0,queries=0;double energy,ecut,pcut;
    double builtE=oldE,builtP=oldP;
    while(requests>>cell>>kind>>proc>>energy>>ecut>>pcut>>count){
        Require(cell>=0&&(kind==0||kind==1)&&proc>=0&&proc<(kind?3:2)&&energy>0&&energy<=.65&&
            std::isfinite(energy)&&ecut>0&&pcut>0&&std::isfinite(ecut)&&std::isfinite(pcut)&&count>=0,"invalid request");
        electronCuts[idx]=ecut*MeV;photonCuts[idx]=pcut*MeV;
        // Rebuild the actual native process table after a cut change. Reading
        // its original lambda table with a different requested cut is invalid.
        // This offline probe owns one thread and restores original cuts/tables.
        if(electronCuts[idx]!=builtE||photonCuts[idx]!=builtP){
            charged[0]->BuildLambdaTable();charged[1]->BuildLambdaTable();
            builtE=electronCuts[idx];builtP=photonCuts[idx];
        }
        const G4ParticleDefinition* particle=kind?static_cast<G4ParticleDefinition*>(G4Gamma::Gamma()):static_cast<G4ParticleDefinition*>(G4Electron::Electron());
        G4Track track(new G4DynamicParticle(particle,{0,0,1},energy*MeV),0,{0,0,0});
        track.SetTouchableHandle(source->GetTrack()->GetTouchableHandle());track.SetWeight(1);
        G4Step step;step.SetTrack(&track);track.SetStep(&step);
        for(auto* point:{step.GetPreStepPoint(),step.GetPostStepPoint()}){
            point->SetMaterial(const_cast<G4Material*>(material));point->SetMaterialCutsCouple(couple);
            point->SetKineticEnergy(energy*MeV);point->SetWeight(1);
        }
        std::size_t modelIndex=idx;G4VEmModel* model=kind?photon[proc]->SelectModelForMaterial(energy*MeV,modelIndex):
            charged[proc]->SelectModelForMaterial(energy*MeV,modelIndex);
        Require(model,"missing model");model->SetupForMaterial(particle,material,energy*MeV);
        const double threshold=kind?ecut:proc?pcut:ecut;
        const double rate=model->CrossSectionPerVolume(material,particle,energy*MeV,threshold*MeV)*mm;
        const double tableRate=kind?photon[proc]->GetLambda(energy*MeV,couple,std::log(energy*MeV))*mm:
            charged[proc]->GetLambda(energy*MeV,couple,std::log(energy*MeV))*mm;
        Require(std::isfinite(rate)&&rate>=0,"invalid native model rate");
        Require(count==0||tableRate>0,"sample request outside native process support");
        double maxResidual=0;
        G4VParticleChange* original=nullptr;
        if(count){
            if(kind){
                photon[proc]->StartTracking(&track);G4ForceCondition condition=NotForced;
                photon[proc]->PostStepGetPhysicalInteractionLength(track,-1,&condition);
                original=photon[proc]->PostStepDoIt(track,step);
                for(int j=0;j<original->GetNumberOfSecondaries();++j)delete original->GetSecondary(j);
                original->Clear();
            }
            else{
                // Obtain the owning process's original change without exposing
                // protected model pointers. This isolated call is not a sample.
                charged[proc]->StartTracking(&track);G4ForceCondition condition=NotForced;
                charged[proc]->PostStepGetPhysicalInteractionLength(track,-1,&condition);
                original=charged[proc]->PostStepDoIt(track,step);
                for(int j=0;j<original->GetNumberOfSecondaries();++j)delete original->GetSecondary(j);
                original->Clear();
            }
        }
        auto* loss=dynamic_cast<G4ParticleChangeForLoss*>(original);
        auto* gamma=dynamic_cast<G4ParticleChangeForGamma*>(original);
        Require(!count||(kind?gamma!=nullptr:loss!=nullptr),"native change type");
        for(int sample=0;sample<count;++sample){
            if(kind)gamma->InitializeForPostStep(track);else loss->InitializeForPostStep(track);
            std::vector<G4DynamicParticle*> children;
            model->SampleSecondaries(&children,couple,track.GetDynamicParticle(),threshold*MeV,energy*MeV);
            const double remaining=(kind?gamma->GetProposedKineticEnergy():loss->GetProposedKineticEnergy())/MeV;
            const double deposit=(kind?gamma->GetLocalEnergyDeposit():loss->GetLocalEnergyDeposit())/MeV;
            const auto direction=kind?gamma->GetProposedMomentumDirection():loss->GetProposedMomentumDirection();
            double total=0;for(auto* child:children){Require(child->GetPDGcode()==11||child->GetPDGcode()==22,"unsupported branch");total+=child->GetKineticEnergy()/MeV;}
            const double residual=std::abs(energy-remaining-deposit-total);
            if(!(remaining>=0&&deposit>=0&&std::isfinite(residual)&&residual<1e-9))
                G4cerr<<std::setprecision(17)<<"MODEL_CLOSURE cell="<<cell<<" kind="<<kind<<" process="<<proc<<" E="<<energy<<" remaining="<<remaining<<" deposit="<<deposit<<" children="<<total<<G4endl;
            Require(remaining>=0&&deposit>=0&&std::isfinite(residual)&&residual<1e-9,"model energy closure");maxResidual=std::max(maxResidual,residual);
            Row r;r.run=cell;r.event=event++;r.process=proc;r.pdg=kind?22:11;r.energy=energy;r.remaining=remaining;
            r.deposit=deposit;r.children=total;r.density=material->GetDensity()/(g/cm3);
            for(int k=0;k<3;++k)r.post[k]=direction[k];raw.write(reinterpret_cast<char*>(&r),sizeof(r));
            for(auto* child:children){r.kind=1;r.pdg=child->GetPDGcode();r.remaining=child->GetKineticEnergy()/MeV;r.deposit=r.children=0;
                for(int k=0;k<3;++k)r.post[k]=child->GetMomentumDirection()[k];raw.write(reinterpret_cast<char*>(&r),sizeof(r));delete child;}
        }
        audit<<cell<<','<<kind<<','<<proc<<','<<energy<<','<<ecut<<','<<pcut<<','<<material->GetDensity()/(g/cm3)<<','<<rate<<','<<tableRate<<','<<count<<','<<count<<','<<maxResidual<<','<<model->GetName()<<'\n';
        ++queries;
    }
    // Optional composition-function extraction. Samples refer to explicit
    // physical cuts, never to density-node identifiers. Each lambda interval
    // is cubic in ENERGY, on the actual native cut-dependent logarithmic grid.
    std::ifstream lambdaCuts("lambda_cuts.txt");
    if(lambdaCuts){
        std::ofstream segments("lambda_segments.csv");
        segments<<std::setprecision(17)<<"request,process,cut_MeV,emin_MeV,emax_MeV,bins,interval,low_MeV,high_MeV,y0,y1,y2,y3\n";
        int request,process;double cut;
        while(lambdaCuts>>request>>process>>cut){
            Require(request>=0&&(process==0||process==1)&&cut>0&&std::isfinite(cut),"invalid lambda-cut request");
            (process?photonCuts[idx]:electronCuts[idx])=cut*MeV;
            charged[process]->BuildLambdaTable();
            auto* vector=(*charged[process]->LambdaTable())[idx];
            Require(vector&&vector->GetVectorLength()>1,"composition native lambda vector missing");
            const auto bins=vector->GetVectorLength()-1;
            for(std::size_t i=0;i<bins;++i){
                const double lo=vector->Energy(i),hi=vector->Energy(i+1);
                segments<<request<<','<<process<<','<<cut<<','<<vector->Energy(0)/MeV<<','<<vector->Energy(bins)/MeV<<','<<bins<<','<<i<<','<<lo/MeV<<','<<hi/MeV;
                for(double fraction:{.125,.375,.625,.875}){
                    const double e=lo+fraction*(hi-lo);
                    const double value=charged[process]->GetLambda(e,couple,std::log(e))*mm;
                    // Standalone custom materials must own their table vector.
                    Require(std::abs(value-vector->Value(e)*mm)<1e-10*std::max(1.,std::abs(value)),"unexpected base-couple lambda scaling");
                    segments<<','<<value;
                }
                segments<<'\n';
            }
        }
        Require(lambdaCuts.eof()&&bool(segments),"lambda-function export failure");
    }
    std::ifstream shellQueries("penelope_queries.txt");
    if(shellQueries){
        auto* table=G4PenelopeOscillatorManager::GetOscillatorManager()->GetOscillatorTableIonisation(material);
        const auto* transition=G4AtomicTransitionManager::Instance();
        std::ofstream oscillators("penelope_oscillators.csv"),weights("penelope_weights.csv");
        oscillators<<std::setprecision(17)<<"oscillator,strength,ionization_MeV,resonance_MeV,recoil_cut_MeV,z,shell_flag,binding_MeV\n";
        for(std::size_t i=0;i<table->size();++i){auto* oscillator=(*table)[i];
            const auto z=int(oscillator->GetParentZ()),flag=oscillator->GetShellFlag();
            const double binding=z>0&&flag<30?transition->Shell(z,flag-1)->BindingEnergy():0.;
            oscillators<<i<<','<<oscillator->GetOscillatorStrength()<<','<<oscillator->GetIonisationEnergy()/MeV<<','<<oscillator->GetResonanceEnergy()/MeV<<','<<oscillator->GetCutoffRecoilResonantEnergy()/MeV<<','<<z<<','<<flag<<','<<binding/MeV<<'\n';
        }
        // opt4 PenIoni runs from100eV to100keV: max(20 bins/decade,100).
        G4PenelopeIonisationXSHandler handler(100);
        weights<<std::setprecision(17)<<"request,energy_MeV,cut_MeV,density_correction,oscillator,probability\n";
        int request;double e,cut;
        while(shellQueries>>request>>e>>cut){
            Require(request>=0&&e>0&&e<=.1&&cut>0,"invalid Penelope query");
            handler.BuildXSTable(material,cut*MeV,G4Electron::Electron());
            const auto* cross=handler.GetCrossSectionTableForCouple(G4Electron::Electron(),material,cut*MeV);
            Require(cross&&cross->GetNumberOfShells()==table->size(),"Penelope composition shell count");
            for(std::size_t i=0;i<table->size();++i)
                weights<<request<<','<<e<<','<<cut<<','<<handler.GetDensityCorrection(material,e*MeV)<<','<<i<<','<<cross->GetNormalizedShellCrossSection(i,e*MeV)<<'\n';
        }
        Require(shellQueries.eof()&&bool(weights)&&bool(oscillators),"Penelope-function export failure");
        // Density-independent EADL vacancy laws. Shell IDs are NOT shell
        // indices; keep both explicitly for the FP32 cascade sampler.
        std::ofstream atoms("atomic_shells.csv"),radiative("atomic_radiative.csv"),auger("atomic_auger.csv"),settings("atomic_settings.txt");
        atoms<<std::setprecision(17)<<"z,index,shell_id,binding_MeV\n";
        radiative<<std::setprecision(17)<<"z,vacancy,target,energy_MeV,probability\n";
        auger<<std::setprecision(17)<<"z,vacancy,target,ejected,energy_MeV,probability\n";
        std::set<int> elements;
        for(auto* o:*table)if(o->GetParentZ()>0)elements.insert(int(o->GetParentZ()));
        for(int z:elements){
            for(int j=0;j<transition->NumberOfShells(z);++j){const auto* s=transition->Shell(z,j);
                atoms<<z<<','<<j<<','<<s->ShellId()<<','<<s->BindingEnergy()/MeV<<'\n';}
            if(z<6)continue;
            for(int j=0;j<transition->NumberOfReachableShells(z);++j){const auto* s=transition->ReachableShell(z,j);
                for(unsigned k=0;k<s->TransitionProbabilities().size();++k)
                    radiative<<z<<','<<s->FinalShellId()<<','<<s->OriginatingShellId(k)<<','<<s->TransitionEnergy(k)/MeV<<','<<s->TransitionProbability(k)<<'\n';}
            for(int j=0;j<transition->NumberOfReachableAugerShells(z);++j){const auto* s=transition->ReachableAugerShell(z,j);
                for(int target:*s->TransitionOriginatingShellIds())
                    for(unsigned k=0;k<s->AugerTransitionProbabilities(target)->size();++k)
                        auger<<z<<','<<s->FinalShellId()<<','<<target<<','<<s->AugerOriginatingShellId(k,target)<<','<<s->AugerTransitionEnergy(k,target)/MeV<<','<<s->AugerTransitionProbability(k,target)<<'\n';}
        }
        auto* deexcitation=G4LossTableManager::Instance()->AtomDeexcitation();
        settings<<"active="<<(deexcitation&&deexcitation->CheckDeexcitationActiveRegion(idx))<<"\nauger="<<(deexcitation&&deexcitation->IsAugerActive())<<"\ncascade="<<(deexcitation&&deexcitation->IsAugerCascadeActive())<<'\n';
        Require(bool(atoms)&&bool(radiative)&&bool(auger)&&bool(settings),"atomic-law export failure");
    }
    std::ifstream bremsQueries("brems_queries.txt");
    if(bremsQueries){
        std::ofstream spectrum("brems_integrals.csv");
        spectrum<<std::setprecision(17)<<"request,energy_MeV,fraction,mass_rate_per_mm_per_g_cm3\n";
        std::ofstream elementSpectrum;
        if(std::ifstream("brems_elements.enabled")){
            elementSpectrum.open("brems_element_integrals.csv");
            elementSpectrum<<std::setprecision(17)<<"request,energy_MeV,fraction,z,mass_rate_per_mm_per_g_cm3\n";
        }
        int request;double e,x;
        while(bremsQueries>>request>>e>>x){
            Require(request>=0&&e>=.00099&&e<=.65&&x>0&&x<=1,"invalid brems query");
            std::size_t modelIndex=idx;
            auto* model=charged[1]->SelectModelForMaterial(e*MeV,modelIndex);Require(model,"missing brems model");
            model->SetupForMaterial(G4Electron::Electron(),material,e*MeV);
            const double value=model->CrossSectionPerVolume(material,G4Electron::Electron(),e*MeV,x*e*MeV,e*MeV)*mm/(material->GetDensity()/(g/cm3));
            Require(std::isfinite(value)&&value>=0,"invalid brems spectrum integral");
            spectrum<<request<<','<<e<<','<<x<<','<<value<<'\n';
            if(elementSpectrum.is_open()){
                const auto* elements=material->GetElementVector();const auto* densities=material->GetVecNbOfAtomsPerVolume();
                double total=0.;
                for(unsigned j=0;j<material->GetNumberOfElements();++j){
                    const int z=(*elements)[j]->GetZasInt();
                    const double rate=model->ComputeCrossSectionPerAtom(G4Electron::Electron(),e*MeV,z,(*elements)[j]->GetA(),x*e*MeV,e*MeV)*densities[j]*mm/(material->GetDensity()/(g/cm3));
                    Require(std::isfinite(rate)&&rate>=0,"invalid element brems integral");total+=rate;
                    elementSpectrum<<request<<','<<e<<','<<x<<','<<z<<','<<rate<<'\n';
                }
                Require(std::abs(total-value)<1e-9*std::max(value,1e-30),"element brems integral sum");
            }
        }
        Require(bremsQueries.eof()&&bool(spectrum),"brems-function export failure");
    }
    std::ifstream atomicQueries("atomic_queries.txt");
    if(atomicQueries){
        std::ofstream output("atomic_events.bin",std::ios::binary);
        unsigned z,index,n,auger,cascade;double e,p;
        struct AtomicEvent {unsigned z,index,event,photons,electrons,auger,cascade,pad;double photon,electron,local,residual;};
        static_assert(sizeof(AtomicEvent)==64);
        while(atomicQueries>>z>>index>>e>>p>>n>>auger>>cascade){
            Require(z>0&&z<=104&&index<unsigned(G4AtomicTransitionManager::Instance()->NumberOfShells(z))&&e>0&&p>0&&n>0&&auger<=1&&cascade<=1,"invalid atomic query");
            // Geant4 11.3.2 aliases Auger and cascade and locks the flag on
            // its first setter. A fresh instance is required for each case.
            Require(auger==cascade,"native Auger/cascade flags are aliases");
            G4UAtomicDeexcitation deexcitation;deexcitation.SetFluo(true);deexcitation.SetAuger(auger);deexcitation.InitialiseForNewRun();
            const auto* shell=G4AtomicTransitionManager::Instance()->Shell(z,index);
            for(unsigned j=0;j<n;++j){
                AtomicEvent row{z,index,j,0,0,auger,cascade,0,0,0,shell->BindingEnergy()/MeV,0};
                std::vector<G4DynamicParticle*> children;deexcitation.GenerateParticles(&children,shell,z,p*MeV,e*MeV);
                for(auto* child:children){const double energy=child->GetKineticEnergy()/MeV;
                    if(energy<row.local){row.local-=energy;
                        if(child->GetDefinition()==G4Gamma::Gamma()){row.photon+=energy;++row.photons;}
                        else{Require(child->GetDefinition()==G4Electron::Electron(),"atomic particle");row.electron+=energy;++row.electrons;}}
                    delete child;}
                row.residual=std::abs(row.local+row.photon+row.electron-shell->BindingEnergy()/MeV);
                Require(row.local>=0&&row.residual<1e-12,"atomic energy closure");output.write(reinterpret_cast<char*>(&row),sizeof(row));
            }
        }
        Require(atomicQueries.eof()&&bool(output),"atomic validation export failure");
    }
    electronCuts[idx]=oldE;photonCuts[idx]=oldP;
    charged[0]->BuildLambdaTable();charged[1]->BuildLambdaTable();
    Require(requests.eof()&&queries>0&&bool(raw)&&bool(audit),"probe output failure");
    G4cout<<"ELECTRON_COMPOSITION_PROBE=PASS queries="<<queries<<" events="<<event<<G4endl;return true;
}
