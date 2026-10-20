#pragma once
#include <array>
namespace carbon {
struct ChargedIon { int z; int a; };
// ABI order of the current TOPAS-derived stopping/EM packages (including Be6).
inline constexpr std::array<ChargedIon, 18> kChargedIons = {{
    {1, 1},
    {1, 2},
    {1, 3},
    {2, 3},
    {2, 4},
    {2, 6},
    {3, 6},
    {3, 7},
    {4, 7},
    {4, 9},
    {4, 10},
    {5, 8},
    {5, 10},
    {5, 11},
    {6, 10},
    {6, 11},
    {6, 12},
    {4, 6},
}};
inline constexpr int get_charged_species_idx(int z, int a) noexcept {
    if (z < 1 || z > 6 || a < 1 || a > 12) return -1;
    switch (z * 256 + a) {
    case 257: return 0;
    case 258: return 1;
    case 259: return 2;
    case 515: return 3;
    case 516: return 4;
    case 518: return 5;
    case 774: return 6;
    case 775: return 7;
    case 1031: return 8;
    case 1033: return 9;
    case 1034: return 10;
    case 1288: return 11;
    case 1290: return 12;
    case 1291: return 13;
    case 1546: return 14;
    case 1547: return 15;
    case 1548: return 16;
    case 1030: return 17;
    default: return -1;
    }
}
} // namespace carbon
