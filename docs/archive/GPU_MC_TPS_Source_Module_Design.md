# GPU Monte Carlo TPS Source Module Design

## 1. Overview

This document describes the design of a Treatment Planning System (TPS) source module for a GPU-based Monte Carlo radiation transport code.

The TPS source converts clinical beam parameters into GPU particle initial states:

- Particle position
- Particle direction
- Energy
- Statistical weight
- Particle type

The generated particles are transported by the GPU Monte Carlo engine.

---

# 2. TPS Beam Model

A TPS beam contains:

- Gantry angle
- Couch angle
- Collimator angle
- Isocenter
- SAD/SDD
- Energy
- MU weight
- PBS spot information

Example:

```json
{
 "gantry":90,
 "couch":0,
 "collimator":0,
 "isocenter":[0,0,0],
 "energy":150
}
```

---

# 3. Coordinate Systems

## 3.1 Patient Coordinate System

DICOM patient coordinates:

- X: Left → Right
- Y: Posterior → Anterior
- Z: Inferior → Superior

---

## 3.2 Beam Coordinate System

Local beam axis:

- u: lateral direction
- v: longitudinal direction
- w: beam propagation direction

Initial beam direction:

```
w = (0,0,-1)
```

---

# 4. Gantry Rotation

The source position is calculated:

\[
S = Iso + SAD \times n
\]

where:

- Iso: isocenter
- SAD: source-axis distance
- n: source direction vector

## Gantry 0°

Source above patient:

```
      Source

        |
        v

       Iso
```

Direction:

```
(0,0,-1)
```

## Gantry 90°

```
Source -----> Iso
```

Direction:

```
(-1,0,0)
```

## Gantry 180°

```
       Iso

        ^
        |

      Source
```

Direction:

```
(0,0,1)
```

## Gantry 270°

```
Iso <----- Source
```

Direction:

```
(1,0,0)
```

---

# 5. Rotation Matrix

Beam transformation:

\[
R = R_{gantry}R_{couch}R_{collimator}
\]

Global direction:

\[
d_{global}=R d_{local}
\]

CUDA:

```cpp
float3 rotateBeam(float3 localDir, Matrix3x3 R)
{
    return R * localDir;
}
```

---

# 6. GPU Particle Structure

```cpp
struct Particle
{
    float3 position;
    float3 direction;
    float energy;
    float weight;
    int type;
};
```

---

# 7. PBS Proton Source

PBS input:

- Energy
- Spot position
- MU weight
- Beam sigma

Example:

```json
{
 "energy":150,
 "x":5.2,
 "y":-3.1,
 "MU":0.02
}
```

Spot position:

\[
r=xu+yv
\]

Global position:

\[
r_{global}=Iso+SADn+Rr
\]

---

# 8. Source Kernel

GPU workflow:

```
TPS parameters

      |

Coordinate transformation

      |

Particle initialization

      |

Monte Carlo transport
```

---

# 9. Validation

Validate:

## Geometry

- Gantry 0
- Gantry 90
- Gantry 180
- Gantry 270

## Dose

Compare:

- Depth dose
- Lateral profile
- Absolute dose
- Gamma analysis

---

# 10. Common Errors

## Left-right flip

Cause:

DICOM LPS/RAS mismatch

## Gantry 90/270 inversion

Cause:

Rotation matrix sign error

## Patient orientation error

Support:

- HFS
- HFP
- FFS
- FFP

---

# 11. Summary

TPS source converts:

```
Gantry
Couch
Collimator
Isocenter
Energy
Spot
MU
```

into:

```
Particle position
Particle direction
Energy
Weight
```

Coordinate transformation is the key component.

---

# 12. MAIGO implementation contract

The implemented module is explicitly opt-in with `tpsSource: true` (the
`tps_source` alias is also accepted). The default is false, and the existing
TOPAS/CT source construction remains in a separate legacy branch so historical
CT gamma comparisons retain their source coordinates and RNG streams.

The flat YAML beam definition contains gantry, couch, collimator, isocenter,
SAD, patient position, particle type, and an optional PBS spot CSV. Required CSV
columns are `energy_MeVu,x_mm,y_mm,mu_weight`; optional columns carry energy
spread and bivariate Gaussian emittance. MU is converted into an exact integer
history allocation with the Hamilton method.

The clinical source is placed at `isocenter - SAD*w`, with spot offsets in the
rotated local `u/v` plane. On the GPU, TPS primaries travel through vacuum to
the first ray intersection with the finite voxel AABB. This permits cardinal
gantry directions such as 90 and 270 degrees without applying the legacy CT
z=0 projection. The current physics package supports carbon primaries only;
other particle types are rejected rather than silently transported as carbon.

See `config/beam_tps_source_example.yaml`, `validation/tps/spots_example.csv`,
and `validation/tps/README.md` for the executable interface.

The post-implementation 10M legacy CT regression reproduced the previous
physical-dose match: global 3%/3 mm gamma was 95.4294% and local gamma was
93.6898% at the 10% dose threshold (previously 95.4332% and about 93.69%).
