#pragma once

namespace carbon {
// Electron-operator identities are separate from ion EM material indices.
// 0..24 are Schneider compositions; 25 is TOPAS Water_75eV, never a tissue alias.
inline constexpr unsigned kElectronWaterSection=25;
inline constexpr unsigned kElectronMaterialCount=26;
}
