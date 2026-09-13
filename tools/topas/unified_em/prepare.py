"""Prepare all-ion, density-resolved material EM extraction; no dose MC."""
from pathlib import Path
import json,sys,shutil
import numpy as np
R=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(R/'tools'))
from audit_schneider_response_scope import schneider_identity
import argparse,math
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=R/'scratch/unified_joint_em_20260913');parser.add_argument('--remote',default='/home/v/unified_joint_em_20260913');args=parser.parse_args()
out=args.output;remote=args.remote;out.mkdir(parents=True,exist_ok=True)
meta=json.loads((R/'data/schneider/schneider_stopping_v1.metadata.json').read_text())['section_manifest']
species=json.loads((R/'data/schneider/schneider_ion_section_stopping_v1.metadata.json').read_text())['species_za']
identities=[schneider_identity(R/'data/HUtoMaterialSchneider.txt',hu) for hu in range(-1000,2996)]
elements=['Hydrogen','Carbon','Nitrogen','Oxygen','Magnesium','Phosphorus','Sulfur','Chlorine','Argon','Calcium','Sodium','Potassium','Titanium']
materials=[dict(name='JEM_W',section=-1,density_g_cm3=1.,material='Water_75eV')];text=''
for sid in range(25):
 values=[v['density_g_cm3'] for v in identities if v['material_section']==sid]
 # Explicit density nodes, including the existing benchmark material density.
 extra=[v['density_g_cm3'] for v in identities if v['material_section']==sid and v['hu'] in [-1000,-535,100,1250]]
 count=max(2,math.ceil(math.log(max(values)/min(values))/math.log(1.05))+1)
 nodes=sorted(set([*np.geomspace(min(values),max(values),count),min(values),max(values),float(np.median(values)),*extra]))
 for k,rho in enumerate(nodes):
  name=f'JEM_S{sid:02d}D{k}';pairs=[(e,w) for e,w in zip(elements,meta[sid]['element_weights']) if w>0]
  text+=f'sv:Ma/{name}/Components = {len(pairs)} '+' '.join('"'+e+'"' for e,w in pairs)+'\n'
  text+=f'uv:Ma/{name}/Fractions = {len(pairs)} '+' '.join(str(w) for e,w in pairs)+'\n'
  text+=f'd:Ma/{name}/Density = {rho:.17g} g/cm3\nd:Ma/{name}/MeanExcitationEnergy = {meta[sid]["mean_excitation_energy_eV"]} eV\n'
  materials.append(dict(name=name,section=sid,density_g_cm3=rho))
# Explicit water composition matches TOPAS Water_75eV reference.
text+='sv:Ma/JEM_W/Components = 2 "Hydrogen" "Oxygen"\nuv:Ma/JEM_W/Fractions = 2 0.111894 0.888106\nd:Ma/JEM_W/Density = 1 g/cm3\nd:Ma/JEM_W/MeanExcitationEnergy = 75 eV\n'
(out/'materials.txt').write_text(text)
geometry=''
for i,m in enumerate(materials):
 geometry+=f's:Ge/V{i}/Parent = "World"\ns:Ge/V{i}/Type = "TsBox"\ns:Ge/V{i}/Material = "{m.get("material",m["name"])}"\nd:Ge/V{i}/HLX = 1 mm\nd:Ge/V{i}/HLY = 1 mm\nd:Ge/V{i}/HLZ = 1 mm\nd:Ge/V{i}/TransX = {i*4} mm\nd:Ge/V{i}/TransZ = 1 mm\n'
for z,a in species:
 d=out/f'ion_{z}_{a}';d.mkdir(exist_ok=True)
 s=f'includeFile = {remote}/materials.txt\ni:Ts/NumberOfThreads = 1\ni:Ts/Seed = 20260913\nb:Gr/Enable = "False"\ns:Ge/World/Material = "Vacuum"\nd:Ge/World/HLX = 1000 mm\nd:Ge/World/HLY = 500 mm\nd:Ge/World/HLZ = 500 mm\n'+geometry
 s+=f'''s:Ge/Source/Parent = "World"
s:Ge/Source/Type = "Group"
s:So/Beam/Type = "Beam"
s:So/Beam/Component = "Source"
s:So/Beam/BeamParticle = "GenericIon({z},{a})"
d:So/Beam/BeamEnergy = {100*a} MeV
u:So/Beam/BeamEnergySpread = 0
s:So/Beam/BeamPositionDistribution = "None"
s:So/Beam/BeamAngularDistribution = "None"
i:So/Beam/NumberOfHistoriesInRun = 1
s:Ph/Default/Type = "Geant4_Modular"
sv:Ph/Default/Modules = 2 "g4em-standard_opt4" "g4decay"
d:Ph/Default/EMRangeMax = 10 GeV
s:Sc/Export/Quantity = "CarbonStoppingPowerNtuple"
s:Sc/Export/Component = "V0"
s:Sc/Export/OutputType = "ASCII"
s:Sc/Export/OutputFile = "export"
s:Sc/Export/IfOutputFileAlreadyExists = "Exit"
'''
 (d/'topas.txt').write_text(s)
(out/'extraction_manifest.json').write_text(json.dumps({'schema':'unified_joint_em_extraction_v1','materials':materials,'species_za':species,'reference':'Geant4 11.3.2 default opt4, EMRangeMax 10 GeV','density_policy':'explicit density nodes for every Schneider section, adjacent density ratio <= 1.05; research interpolation requires validation','max_total_kinetic_MeV':6000,'grid':'8193 logarithmic energy nodes plus reference beam energies; original raw spline intervals also exported','scope':'charged ion electronic losses only; no hadronic package replacement'},indent=2))
print(len(materials),'material/density points x',len(species),'ions')
