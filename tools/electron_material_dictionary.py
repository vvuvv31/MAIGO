"""Bind V4 run-local indices to actual material composition, never density aliases."""
import hashlib
import json
import math
import re
from pathlib import Path
import numpy as np
from audit_schneider_response_scope import schneider_identity

MARKER='MAIGO_ELECTRON_MATERIAL_V1 '
CANONICAL_Z=[1,6,7,8,12,15,16,17,18,20,11,19,22]


def read_material_dictionary(log):
    result={}
    for line in log.splitlines():
        if MARKER not in line:continue
        m=json.loads(line.split(MARKER,1)[1])
        index=m['index']
        if type(index) is not int or index<0 or not isinstance(m['name'],str):
            raise ValueError('Invalid material identity')
        for field in ('density_g_cm3','mean_excitation_eV'):
            if not math.isfinite(m[field]) or m[field]<=0:raise ValueError('Invalid material property')
        elements=m['elements']
        if not elements:raise ValueError('Missing composition')
        seen=set()
        for el in elements:
            z=el['Z']
            if not math.isfinite(z) or z!=int(z) or not 1<=z<=118 or z in seen:
                raise ValueError('Invalid/duplicate element')
            seen.add(z)
            if not math.isfinite(el['A_g_mol']) or el['A_g_mol']<=0:
                raise ValueError('Invalid elemental mass')
            if not math.isfinite(el['mass_fraction']) or not 0<el['mass_fraction']<=1:
                raise ValueError('Invalid elemental fraction')
        if abs(sum(e['mass_fraction'] for e in elements)-1)>1e-8:
            raise ValueError('Composition fractions do not sum to one')
        m['elements']=sorted(elements,key=lambda e:e['Z'])
        if index in result and result[index]!=m:raise ValueError('Material index changed across workers')
        result[index]=m
    if not result:raise ValueError('Missing actual material dictionary; no inferred fallback')
    return result


def bind_schneider_materials(materials, states, source):
    text=Path(source).read_text()
    names=re.findall(r'^sv:Ge/Patient/SchneiderElements\s*=\s*13\s+(.*)$',text,re.M)
    expected_names=['Hydrogen','Carbon','Nitrogen','Oxygen','Magnesium','Phosphorus','Sulfur',
                    'Chlorine','Argon','Calcium','Sodium','Potassium','Titanium']
    if len(names)!=1 or re.findall(r'"([^"]+)"',names[0])!=expected_names:
        raise ValueError('Unexpected Schneider elemental order')
    bound={}
    used=set(map(int,states[:,42]))|set(map(int,states[states[:,43]>=0,43]))
    for index in used:
        if index not in materials:raise ValueError('Missing used material index')
        m=materials[index]
        match=re.fullmatch(r'PatientTissueFromHU(Negative)?(\d+)',m['name'])
        if not match:raise ValueError('Not an explicitly identified Schneider material')
        hu=int(match[2])*(-1 if match[1] else 1)
        identity=schneider_identity(source,hu)
        expected=float(format(identity['density_g_cm3'],'.6g'))
        if not math.isclose(m['density_g_cm3'],expected,rel_tol=1e-6,abs_tol=1e-8):
            raise ValueError('Runtime density disagrees with Schneider formula')
        sid=identity['material_section']
        matches=re.findall(r'^uv:Ge/Patient/SchneiderMaterialsWeight'+str(sid+1)+r'\s*=\s*(.*)$',text,re.M)
        if len(matches)!=1:raise ValueError('Missing/duplicate composition row')
        words=matches[0].split();weights=list(map(float,words[1:]))
        if int(words[0])!=13 or len(weights)!=13:raise ValueError('Invalid composition width')
        expected_composition={z:f for z,f in zip(CANONICAL_Z,weights) if f>0}
        actual={int(e['Z']):e['mass_fraction'] for e in m['elements']}
        if actual.keys()!=expected_composition.keys() or any(abs(actual[z]-expected_composition[z])>1e-8 for z in actual):
            raise ValueError('Runtime composition disagrees with Schneider section')
        for id_col,rho_col in ((42,20),(43,41)):
            rho=states[states[:,id_col]==index,rho_col]
            if not np.allclose(rho,m['density_g_cm3'],rtol=1e-10,atol=1e-12):
                raise ValueError('Step density disagrees with material dictionary')
        signature=hashlib.sha256(json.dumps(m['elements'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        bound[index]=dict(identity,material=m,composition_sha256=signature)
    return bound
