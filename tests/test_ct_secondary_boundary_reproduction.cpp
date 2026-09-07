#include "carbon/ct_grid.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#ifdef SYCL_LANGUAGE_VERSION
#include <sycl/sycl.hpp>
#endif

static bool endpoint_checks() {
    for (int axis=0;axis<3;++axis) for (float sign : {-1.0F,1.0F}) {
        std::array<float,3> start{.25F,.25F,.25F},dir{};
        start[axis]=sign>0 ? .49F : .51F;dir[axis]=sign;
        const auto c=carbon::clamp_step_to_ct_faces_exact(.05F,
            start[0],start[1],start[2],dir[0],dir[1],dir[2],
            0,0,0,.5F,.5F,.5F,3,3,3);
        const auto end=carbon::ct_finish_exact_face_step(c,c.step_mm,start,dir,{0,0,0},{.5F,.5F,.5F});
        if (!c.hit_face || (sign>0 ? end[axis]<=.5F : end[axis]>=.5F))return false;
        const auto truncated=carbon::ct_finish_exact_face_step(c,.5F*c.step_mm,start,dir,{0,0,0},{.5F,.5F,.5F});
        if (truncated[axis]!=start[axis]+.5F*c.step_mm*sign)return false;
    }
    // A corner crosses three faces; snap all, not just the first axis.
    const auto c=carbon::clamp_step_to_ct_faces_exact(.05F,.49F,.49F,.49F,
        1,1,1,0,0,0,.5F,.5F,.5F,3,3,3);
    const auto p=carbon::ct_finish_exact_face_step(c,c.step_mm,{.49F,.49F,.49F},
        {1,1,1},{0,0,0},{.5F,.5F,.5F});
    return c.axis_mask==7 && p[0]>.5F && p[1]>.5F && p[2]>.5F;
}

int main() {
    if (!endpoint_checks())throw std::runtime_error("Host exact face regression");
#ifdef SYCL_LANGUAGE_VERSION
    sycl::queue q(sycl::gpu_selector_v);
    auto* ok=sycl::malloc_shared<bool>(1,q);
    q.single_task([=](){*ok=endpoint_checks();}).wait_and_throw();
    const bool passed=*ok;sycl::free(ok,q);
    if (!passed)throw std::runtime_error("Device exact face regression");
    std::cout << "Host/device exact face checks PASS\n";
#endif
    // This is a diagnostic of the CURRENT secondary fast path, not an
    // acceptance test blessing its overshoot. Two separated same-material
    // endpoints do not prove the intervening segment homogeneous.
    const float rho[3]={0.02F,1.0F,0.02F};
    const std::uint8_t mat[3]={0,8,0};
    for (float proposed : {0.05F,0.9F}) {
        const auto old=carbon::clamp_step_to_ct_faces_near_z_if_needed(
            proposed,0.49F,0.25F,0.25F,1,0,0,
            0,0,0,0.5F,0.5F,0.5F,3,1,1,rho,mat,rho[0],mat[0],true,nullptr);
        const auto exact=carbon::clamp_step_to_ct_faces_exact(
            proposed,0.49F,0.25F,0.25F,1,0,0,
            0,0,0,0.5F,0.5F,0.5F,3,1,1);
        if (!(old>exact.step_mm+0.01F) || std::fabs(exact.step_mm-0.01F)>1e-6F)
            throw std::runtime_error("Reproduction changed; inspect before repair");
        std::cout << "proposed=" << proposed << " legacy=" << old
                  << " exact=" << exact.step_mm << '\n';
    }
}
