#include "ElectronExtractionRange.hh"
#include "ElectronBoundaryProbe.hh"
#include "TsParameterManager.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4Electron.hh"
#include "G4ProcessManager.hh"
#include "G4VMultipleScattering.hh"
#include "G4VEnergyLossProcess.hh"
#include "G4GoudsmitSaundersonMscModel.hh"
#include "G4GoudsmitSaundersonTable.hh"
#include "G4GSPWACorrections.hh"
#include "G4Material.hh"
#include "G4EmParameters.hh"
#include "G4EmCalculator.hh"
#include "G4ProductionCutsTable.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include <fstream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <algorithm>
#include "G4PhysicsTable.hh"
#include "G4PhysicsVector.hh"
ElectronBoundaryProbe::ElectronBoundaryProbe(TsParameterManager* pm,TsMaterialManager* mm,
    TsGeometryManager* gm,TsScoringManager* sm,TsExtensionManager* em,G4String name,
    G4String quantity,G4String output,G4bool sub):TsVScorer(pm,mm,gm,sm,em,name,quantity,output,sub) {}
void ElectronBoundaryProbe::Require(bool ok,const char* why) {
    if(!ok){G4cerr<<"ElectronBoundaryProbe audit: "<<why<<G4endl;fPm->AbortSession(1);}
}
G4bool ElectronBoundaryProbe::ProcessHits(G4Step* source,G4TouchableHistory*) {
    if(done_ || source->GetTrack()->GetParentID()!=0)return false;
    done_=true;
    const auto* couple=source->GetPreStepPoint()->GetMaterialCutsCouple();
    const auto* material=couple->GetMaterial();auto* electron=G4Electron::Electron();
    G4GoudsmitSaundersonMscModel* gs=nullptr;G4VEnergyLossProcess* ion=nullptr;G4VEnergyLossProcess* brem=nullptr;
    const auto* list=electron->GetProcessManager()->GetProcessList();
    for(std::size_t i=0;i<list->size();++i){auto* p=(*list)[i];
        if(auto* m=dynamic_cast<G4VMultipleScattering*>(p))gs=dynamic_cast<G4GoudsmitSaundersonMscModel*>(m->GetModelByIndex(0));
        if(p->GetProcessName()=="eIoni")ion=dynamic_cast<G4VEnergyLossProcess*>(p);
        if(p->GetProcessName()=="eBrem")brem=dynamic_cast<G4VEnergyLossProcess*>(p);
    }
    Require(gs&&ion&&brem,"native GS/ionization/bremsstrahlung missing");
    auto* table=gs->GetGSTable();Require(table,"native GS table missing");
    auto* options=G4EmParameters::Instance();
    G4EmCalculator calculator;auto* cuts=G4ProductionCutsTable::GetProductionCutsTable();
    const double electronCut=(*cuts->GetEnergyCutsVector(1))[couple->GetIndex()];
    const double photonCut=(*cuts->GetEnergyCutsVector(0))[couple->GetIndex()];
    std::ofstream meta("boundary_options.txt"),output("boundary_parameters.csv");
    meta<<std::setprecision(17)<<"model="<<gs->GetName()<<"\nmott="<<gs->GetOptionMottCorrection()<<"\npwa="<<gs->GetOptionPWACorrection()
        <<"\nstep_type="<<int(options->MscStepLimitType())<<"\nskin="<<options->MscSkin()<<"\nrange_factor="<<options->MscRangeFactor()
        <<"\ndensity_g_cm3="<<material->GetDensity()/(g/cm3)<<"\nelectron_cut_MeV="<<electronCut/MeV<<"\nphoton_cut_MeV="<<photonCut/MeV<<"\nlowest_electron_energy_MeV="<<options->LowestElectronEnergy()/MeV<<"\nlinear_loss_limit="<<options->LinearLossLimit()<<"\n";
    output<<std::setprecision(17)<<"energy_MeV,elastic_mfp_mm,transport_mfp_mm,native_transport_mfp_mm,screening,scattering_power_correction,restricted_dedx_MeV_mm,restricted_range_mm,ionization_rate_mm,brem_rate_mm,raw_interpolated_ionization_rate_mm,raw_interpolated_brem_rate_mm\n";
    // Reconstruct the published Geant4 11.3.2 GS elastic mean-free-path
    // expression from its public material/cut correction interfaces. Compare
    // lambda1 with the public native transport cross section times the real
    // cut's SCP. CrossSectionPerVolume alone omits SCP; GetTransportMeanFreePath
    // instead requires hidden GS stepping state, unavailable to this probe.
    for(unsigned i=0;i<=256;++i){
        const double e=1e-5*std::pow(ElectronExtractionMaximumMeV()/1e-5,double(i)/256),pt2=e*(e+2*electron_mass_c2),beta2=pt2/(pt2+electron_mass_c2*electron_mass_c2);
        const int index=material->GetIndex();double scr=1,q1=1,g2=1,scp=1;
        if(gs->GetOptionMottCorrection()){
            table->GetMottCorrectionFactors(std::log(e),beta2,index,scr,q1,g2);
            scp=table->ComputeScatteringPowerCorrection(couple,e);
        }else if(gs->GetOptionPWACorrection())gs->GetPWACorrection()->GetPWACorrectionFactors(std::log(e),beta2,index,scr,q1,g2);
        const double bc=table->GetMoliereBc(index),a=table->GetMoliereXc2(index)/(4*pt2*bc)*scr;
        const double lambda0=beta2*(1+a)*scr/bc/scp,g1=2*a*((1+a)*std::log(1/a+1)-1),lambda1=lambda0/g1;
        const double native=1/(gs->CrossSectionPerVolume(material,electron,e)*scp);
        Require(std::isfinite(lambda0)&&lambda0>0&&std::isfinite(native)&&native>0&&std::abs(native/lambda1-1)<1e-10,"native GS mean free path mismatch");
        const double dedx=ion->GetDEDX(e,couple),range=ion->GetRange(e,couple);
        // The process's raw spline getter may overshoot below zero at a cut
        // threshold. Evaluate the native model with the actual production
        // cuts; retain raw getters as diagnostics, never silently clamp them.
        const double delta=calculator.ComputeCrossSectionPerVolume(e,electron,"eIoni",material,electronCut);
        const double rad=calculator.ComputeCrossSectionPerVolume(e,electron,"eBrem",material,photonCut);
        const double rawDelta=ion->GetLambda(e,couple),rawRad=brem->GetLambda(e,couple);
        if(!(std::isfinite(dedx)&&dedx>=0&&std::isfinite(range)&&range>=0&&std::isfinite(delta)&&delta>=0&&std::isfinite(rad)&&rad>=0))
            G4cerr<<std::setprecision(17)<<"BOUNDARY_TABLE_DIAGNOSTIC E="<<e<<" dedx="<<dedx<<" range="<<range<<" delta="<<delta<<" brem="<<rad<<G4endl;
        Require(std::isfinite(dedx)&&dedx>=0&&std::isfinite(range)&&range>=0&&std::isfinite(delta)&&delta>=0&&std::isfinite(rad)&&rad>=0,"native energy-loss tables invalid");
        output<<e/MeV<<','<<lambda0/mm<<','<<lambda1/mm<<','<<native/mm<<','<<a<<','<<scp<<','<<dedx/(MeV/mm)<<','<<range/mm<<','<<delta*mm<<','<<rad*mm<<','<<rawDelta*mm<<','<<rawRad*mm<<'\n';
    }
    Require(bool(output)&&bool(meta),"probe output failure");
    // Export each exact native spline interval using four interior values.
    // GetLambda supplies the actual couple density scaling. Interior samples
    // avoid aliasing a discontinuous table endpoint to a neighbouring piece.
    std::ofstream cubic("hard_rate_cubics.csv");
    cubic<<std::setprecision(17)<<"process,low_MeV,high_MeV,v0,v1,v2,v3,max_absolute_error\n";
    G4VEnergyLossProcess* processes[]={ion,brem};
    for(int p=0;p<2;++p){
        auto* table=processes[p]->LambdaTable();Require(table,"missing native lambda table");
        std::vector<double> edges{1e-5,ElectronExtractionMaximumMeV()};
        // Include all material vectors' knots: redundant subdivisions are
        // harmless, and avoid assuming currentCouple==basedCoupleIndex.
        for(auto* v:*table)if(v)for(std::size_t j=0;j<v->GetVectorLength();++j){
            const double e=v->Energy(j)/MeV;if(e>1e-5&&e<ElectronExtractionMaximumMeV())edges.push_back(e);
        }
        std::sort(edges.begin(),edges.end());edges.erase(std::unique(edges.begin(),edges.end()),edges.end());
        const double ts[]={.125,.375,.625,.875};
        for(std::size_t i=0;i+1<edges.size();++i){
            double values[4];const double lo=edges[i],hi=edges[i+1];
            for(int j=0;j<4;++j)values[j]=processes[p]->GetLambda((lo+(hi-lo)*ts[j])*MeV,couple)*mm;
            double error=0,scale=0;
            for(int j=0;j<4;++j)scale=std::max(scale,std::abs(values[j]));
            for(int j=0;j<=32;++j){const double t=(j+.5)/33;double predicted=0;
                for(int k=0;k<4;++k){double term=values[k];for(int l=0;l<4;++l)if(l!=k)term*=(t-ts[l])/(ts[k]-ts[l]);predicted+=term;}
                const double actual=processes[p]->GetLambda((lo+(hi-lo)*t)*MeV,couple)*mm;
                error=std::max(error,std::abs(actual-predicted));
            }
            Require(error<=1e-11*std::max(scale,1.),"native rate is not reconstructed cubic");
            cubic<<p<<','<<lo<<','<<hi;for(double v:values)cubic<<','<<v;cubic<<','<<error<<'\n';
        }
    }
    Require(bool(cubic),"rate cubic output failure");
    G4cout<<"ELECTRON_BOUNDARY_PROBE=PASS"<<G4endl;return true;
}
