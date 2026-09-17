#pragma once
// Relativistic elastic two-body kinematics. The caller supplies nuclear rest
// masses (MeV) and a sampled CM cosine from a projectile/target-specific bank.
// This function does not assume or validate an angular probability law.
#include <cmath>
#include "carbon/schneider_rate_table.hpp"
namespace carbon {
inline int elastic_target_index_from_z(int z) noexcept {
    for (std::size_t i=0;i<kSchneiderCanonicalZ.size();++i)
        if(kSchneiderCanonicalZ[i]==z) return static_cast<int>(i);
    return -1;
}
inline constexpr float kElasticTargetMassU[13]={1,12,14,16,24,31,32,35,40,40,23,39,48};
struct NuclearElasticOutcome {
    float projectile_ke_MeV{},recoil_ke_MeV{};
    float proj_dir_x{},proj_dir_y{},proj_dir_z{1};
    float recoil_dir_x{},recoil_dir_y{},recoil_dir_z{1};
};
inline NuclearElasticOutcome elastic_two_body(float kinetic, float dx,float dy,float dz,
    double cos_cm,float u_phi,double m1,double m2) noexcept {
    NuclearElasticOutcome out{};
    if(!std::isfinite(kinetic)||kinetic<0||!std::isfinite(m1)||!std::isfinite(m2)||
       m1<=0||m2<=0||!std::isfinite(cos_cm)||cos_cm < -1||cos_cm>1||
       !std::isfinite(u_phi)||u_phi<0||u_phi>1) return out;
    double norm=std::sqrt(double(dx)*dx+double(dy)*dy+double(dz)*dz);
    if(!(norm>0)||!std::isfinite(norm)) return out;
    double nx=dx/norm,ny=dy/norm,nz=dz/norm;
    const double T=kinetic, p=std::sqrt(T*(T+2*m1));
    const double s=(m1+m2)*(m1+m2)+2*m2*T;
    const double root_s=std::sqrt(s), pcm=p*m2/root_s;
    const double gamma=(m1+m2+T)/root_s;
    // t = 2 p_cm^2 (1-cos_cm), T_recoil = t/(2 m_target).
    const double recoil=pcm*pcm*(1-double(cos_cm))/m2;
    const double remaining=T-recoil;
    const double pz1=p-gamma*pcm*(1-double(cos_cm));
    const double pz2=gamma*pcm*(1-double(cos_cm));
    const double pt=pcm*std::sqrt((1-double(cos_cm))*(1+double(cos_cm)));
    double ax=0,ay=1;
    if(std::fabs(ny)>.9){ax=1;ay=0;}
    double tx=ay*nz,ty=-ax*nz,tz=ax*ny-ay*nx;
    norm=std::sqrt(tx*tx+ty*ty+tz*tz);tx/=norm;ty/=norm;tz/=norm;
    const double bx=ny*tz-nz*ty,by=nz*tx-nx*tz,bz=nx*ty-ny*tx;
    const double phi=6.2831853071795864769*u_phi;
    const double rx=tx*std::cos(phi)+bx*std::sin(phi);
    const double ry=ty*std::cos(phi)+by*std::sin(phi);
    const double rz=tz*std::cos(phi)+bz*std::sin(phi);
    const double n1=std::sqrt(pz1*pz1+pt*pt),n2=std::sqrt(pz2*pz2+pt*pt);
    out.projectile_ke_MeV=static_cast<float>(remaining>0?remaining:0);
    out.recoil_ke_MeV=static_cast<float>(recoil);
    if(n1>0){out.proj_dir_x=(nx*pz1+rx*pt)/n1;out.proj_dir_y=(ny*pz1+ry*pt)/n1;out.proj_dir_z=(nz*pz1+rz*pt)/n1;}
    if(n2>0){out.recoil_dir_x=(nx*pz2-rx*pt)/n2;out.recoil_dir_y=(ny*pz2-ry*pt)/n2;out.recoil_dir_z=(nz*pz2-rz*pt)/n2;}
    return out;
}
// Legacy diagnostic only. Isotropic CM is not a TOPAS angular model.
inline NuclearElasticOutcome sample_c12_target_elastic(float T,float dx,float dy,float dz,
    float u,float phi,float target_a) noexcept {
    constexpr double carbon_mass=11174.86323534;
    const double target_mass=target_a==1?938.27208816:target_a*931.49410242;
    return elastic_two_body(T,dx,dy,dz,2*u-1,phi,carbon_mass,target_a>0?target_mass:0);
}
} // namespace carbon
