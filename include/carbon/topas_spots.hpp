#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace carbon {

// One PBS spot from a TOPAS-format "spots_*.txt" plan file (L0–L14 channels).
// Used only as a convenient input dump matching TOPAS TimeFeature Step files;
// carbon_mc runs spots in order (no timeline / Tf simulation).
// Channel mapping matches run_test_c_angle01.txt + spots_test_c_angle01.txt:
//   L0  spot / layer id
//   L1  (reserved / energy duplicate)
//   L2  BeamEnergy [MeV] total kinetic for the ion
//   L3  BeamEnergySpread (TOPAS unitless percent, 1.0 => 1%)
//   L4  NumberOfHistoriesInRun
//   L5  BeamPosition TransX [mm]
//   L6  BeamPosition TransZ [mm]
//   L7  BeamPosition RotX [deg]
//   L8  BeamPosition RotY [deg]
//   L9  SigmaX [mm]
//   L10 SigmaXprime
//   L11 CorrelationX
//   L12 SigmaY [mm]
//   L13 SigmaYprime
//   L14 CorrelationY
struct TopasSpot {
    int spot_id{0};
    double energy_MeV{0.0};
    double energy_spread_percent{0.0};
    std::size_t number_of_histories{0};
    double trans_x_mm{0.0};
    double trans_z_mm{0.0};
    double rot_x_deg{0.0};
    double rot_y_deg{0.0};
    double sigma_x_mm{0.0};
    double sigma_x_prime{0.0};
    double correlation_x{0.0};
    double sigma_y_mm{0.0};
    double sigma_y_prime{0.0};
    double correlation_y{0.0};
    double time_end_ms{0.0};
};

// Source pose in world coordinates for one spot (mm / unit direction).
struct SpotSourcePose {
    double origin_x_mm{0.0};
    double origin_y_mm{0.0};
    double origin_z_mm{0.0};
    // Orthonormal beam frame: particles leave along +uz; emittance in ux, uy.
    double ux_x{1.0}, ux_y{0.0}, ux_z{0.0};
    double uy_x{0.0}, uy_y{1.0}, uy_z{0.0};
    double uz_x{0.0}, uz_y{0.0}, uz_z{1.0};
};

struct TopasSpotPlan {
    std::vector<TopasSpot> spots{};
    std::string scatterer_name{"Scatterer1"};
    // Sum of the external optimization weights when apply_weights_from_csv()
    // has been used. Zero means the L4 histories from the TOPAS file are used.
    double total_plan_weight{0.0};
    // |TransY| in the TOPAS run file (default 450 mm SAD).
    double sad_mm{450.0};

    [[nodiscard]] std::size_t total_histories() const noexcept;
    [[nodiscard]] static TopasSpotPlan from_file(const std::filesystem::path& path);
    [[nodiscard]] static TopasSpotPlan from_files(
        const std::vector<std::filesystem::path>& paths);

    // Replace L4 by an exact Monte-Carlo allocation proportional to one weight
    // per input spot. Positive-weight spots receive at least one history; zero
    // weight spots are removed. Returns the number of removed spots.
    std::size_t apply_weights_from_csv(const std::filesystem::path& path,
                                       std::size_t total_histories);

    // TOPAS BeamPosition2 placement: t=(Tx,-SAD,Tz), R = Ry(RotY)*Rx(RotX),
    // local beam +Z → world, origin at translated+rotated source point.
    [[nodiscard]] SpotSourcePose pose_for_spot(const TopasSpot& spot) const noexcept;

    // TOPAS/Geant4 component placement used by the TPS plan: translation is in
    // the parent frame and is not rotated. TOPAS component rotations are passive
    // for the local source frame, hence the negative Rx/Ry angles here. This
    // makes Rx=90 deg at TransY=-SAD point the central ray towards +world-Y.
    [[nodiscard]] SpotSourcePose tps_zero_beam_pose_for_spot(
        const TopasSpot& spot) const noexcept;
};

// Convert a TOPAS world pose into the beam-aligned CT frame used by the GPU for
// this TPS 90-degree field. The spots rotations establish the fixed TPS 0-degree
// source direction; Patient RotZ=90 supplies the treatment angle. The patient is
// transformed back to its local frame, then its local +X becomes GPU +Z.
// ct_axis_min_mm is the first CT X coordinate (for this patient: -104 mm).
[[nodiscard]] SpotSourcePose transform_tps_90_pose_to_ct(
    const SpotSourcePose& world_pose,
    double patient_trans_x_mm,
    double patient_trans_y_mm,
    double patient_trans_z_mm,
    double patient_rot_z_deg,
    double ct_axis_min_mm) noexcept;

// Apply TOPAS-style Rx then Ry (degrees) to a vector.
void rotate_rx_ry(double rx_deg, double ry_deg, double& x, double& y, double& z) noexcept;

}  // namespace carbon
