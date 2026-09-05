#include "carbon/ct_grid.hpp"
#include <filesystem>
#include <iostream>
#include <stdexcept>

// Benchmark-only lossless axis packing. No density or material approximation.
int main(int argc, char** argv) {
    try {
        if (argc != 4) throw std::runtime_error("usage: prepare_ct DICOM SCHNEIDER OUTPUT");
        if (std::filesystem::exists(argv[3])) throw std::runtime_error("refuse overwrite");
        auto src = carbon::CtGrid::from_dicom_directory(argv[1], argv[2], "centered");
        auto dst = src;
        dst.nx = src.ny; dst.ny = src.nz; dst.nz = src.nx;
        dst.spacing_x_mm=src.spacing_y_mm;
        dst.spacing_y_mm=src.spacing_z_mm;
        dst.spacing_z_mm=src.spacing_x_mm;
        dst.origin_x_mm=-0.5F*dst.nx*dst.spacing_x_mm;
        dst.origin_y_mm=-0.5F*dst.ny*dst.spacing_y_mm;
        dst.origin_z_mm=0;
        for (unsigned z=0; z<src.nz; ++z)
            for (unsigned y=0; y<src.ny; ++y)
                for (unsigned x=0; x<src.nx; ++x) {
                    auto a=(static_cast<size_t>(z)*src.ny+y)*src.nx+x;
                    auto b=(static_cast<size_t>(src.nx-1-x)*dst.ny+z)*dst.nx+y;
                    dst.density_g_per_cm3[b]=src.density_g_per_cm3[a];
                    dst.material_id[b]=src.material_id[a];
                }
        dst.write_binary(argv[3]);
        std::cout<<dst.nx<<" "<<dst.ny<<" "<<dst.nz<<"\n";
    } catch (const std::exception& e) {std::cerr<<e.what()<<"\n";return 1;}
}
