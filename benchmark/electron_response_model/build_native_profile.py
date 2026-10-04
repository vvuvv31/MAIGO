#!/usr/bin/env python3
"""Link an isolated reference scorer using unchanged, prebuilt TOPAS libraries."""
import argparse
from pathlib import Path
import shlex
import shutil
import subprocess
from prototype import save, sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, default=Path('/mnt/sda/wuwei/real_ct_electron_diagnostic_20260928/build'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--scorer', choices=['ElectronHistoryProfile', 'ElectronFamilyProfile'], default='ElectronHistoryProfile')
    p.add_argument('--selected-history-scorer',action='store_true')
    p.add_argument('--gs-source',type=Path,help='Isolated official Geant4 11.3.2 GS source override (diagnostic only)')
    p.add_argument('--disable-gs-safety-optimization',action='store_true',
                   help='TRAINING/control only: retain displacement in homogeneous volumes; never modifies installed libraries')
    a=p.parse_args(); a.out=a.out.resolve(); a.out.mkdir(parents=True, exist_ok=False)
    native=Path(__file__).parent/'native'
    assert not a.disable_gs_safety_optimization or a.gs_source
    scorers=[a.scorer]+(['SelectedElectronTransportNtuple'] if a.selected_history_scorer else [])
    for scorer in scorers:
        for suffix in ['.cc','.hh']:shutil.copy2(native/(scorer+suffix),a.out/(scorer+suffix))
    manager=(a.base/'extensions/TsExtensionManager.cc').read_text()
    marker='// Insertion point for Scorers'
    assert manager.count(marker)==1
    for scorer in scorers:
        manager='#include "'+scorer+'.hh"\n'+manager.replace(marker,marker+f'''
    if (quantityNameLower=="{scorer.lower()}")
        return new {scorer}(pM,mM,gM,scM,this,currentScorerName,quantityName,outFileName,isSubScorer);
''')
    (a.out/'TsExtensionManager.cc').write_text(manager)
    flags={}
    for line in (a.base/'extensions/CMakeFiles/extensions.dir/flags.make').read_text().splitlines():
        if line.startswith('CXX_'):
            k,v=line.split(' = ',1); flags[k]=shlex.split(v)
    commands=[]
    extra=[]
    if a.gs_source:
        gs=a.gs_source.read_text()
        marker='G4bool G4GoudsmitSaundersonMscModel::gIsOptimizationOn = true;'
        assert gs.count(marker)==1
        if a.disable_gs_safety_optimization:
            gs=gs.replace(marker,marker.replace('true','false'))
        (a.out/'G4GoudsmitSaundersonMscModel.cc').write_text(gs)
        extra=['G4GoudsmitSaundersonMscModel']
    for name in scorers+['TsExtensionManager']+extra:
        cmd=['/usr/bin/c++',*flags['CXX_DEFINES'], '-I'+str(a.out), '-I'+str(a.base/'extensions'),
             *flags['CXX_INCLUDES'],*flags['CXX_FLAGS'],'-c',str(a.out/(name+'.cc')),'-o',str(a.out/(name+'.o'))]
        commands.append(cmd); subprocess.run(cmd, check=True, cwd=a.base)
    link=shlex.split((a.base/'CMakeFiles/topas.dir/link.txt').read_text())
    target=link.index('-o')+1; link[target]=str(a.out/'topas')
    objects=[str(a.out/(name+'.o')) for name in ['TsExtensionManager']+scorers+extra]
    index=next(i for i,x in enumerate(link) if x.endswith('libmain.a'))
    link[index:index]=objects
    commands.append(link); subprocess.run(link,check=True,cwd=a.base)
    inputs={str(a.base/x):sha(a.base/x) for x in link if x.endswith(('.a','.o')) and not Path(x).is_absolute()}
    save(a.out/'build_manifest.json',dict(commands=commands,base_build=str(a.base),
         base_binary_sha256=sha(a.base/'topas'),binary_sha256=sha(a.out/'topas'),
         native_inputs=inputs, source_sha256={x.name:sha(x) for x in a.out.iterdir() if x.suffix in ['.cc','.hh']},
         gs_original_sha256=sha(a.gs_source) if a.gs_source else None,
         gs_safety_optimization_disabled=a.disable_gs_safety_optimization,
         scope=('DIAGNOSTIC PHYSICS OVERRIDE; not an unchanged TOPAS acceptance reference' if extra else
                'Native diagnostic scorer only; all transport/physics libraries unchanged')))


if __name__=='__main__':
    main()
