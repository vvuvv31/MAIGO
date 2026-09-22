// Device-libm probe: nextafter/expm1/log1p/exp/log accuracy on the GPU.
// Prints raw hex so host/device can be compared bitwise.
#include <sycl/sycl.hpp>
#include <cstdio>

int main() {
    sycl::queue q{sycl::gpu_selector_v};
    std::printf("device: %s\n",
                q.get_device().get_info<sycl::info::device::name>().c_str());
    struct Out {
        float na, na_d;
        float e1, e2, l1, l2, e3, lg;
    } out{};
    {
        sycl::buffer<Out, 1> b(&out, 1);
        q.submit([&](sycl::handler& h) {
            auto a = b.get_access<sycl::access::mode::write>(h);
            h.single_task([=]() {
                a[0].na = sycl::nextafter(0.05F, 1.0e30F) - 0.05F;
                a[0].na_d =
                    sycl::nextafter(0.05F, 1.0e30F) - 0.05F;  // same, sanity
                a[0].e1 = sycl::expm1(-3.4e-7F);
                a[0].e2 = sycl::expm1(-0.0693150F);
                a[0].l1 = sycl::log1p(-3.4e-7F);
                a[0].l2 = sycl::log1p(-0.0625F);
                a[0].e3 = sycl::exp(-0.0693150F);
                a[0].lg = sycl::log(0.9375F);
            });
        }).wait_and_throw();
    }
    std::printf("nextafter(0.05)-0.05 = %a (expect 0x1p-28=%a)\n",
                (double)out.na, (double)0x1p-28f);
    std::printf("expm1(-3.4e-7) = %a\n", (double)out.e1);
    std::printf("expm1(-0.069315) = %a\n", (double)out.e2);
    std::printf("log1p(-3.4e-7) = %a\n", (double)out.l1);
    std::printf("log1p(-0.0625) = %a\n", (double)out.l2);
    std::printf("exp(-0.069315) = %a\n", (double)out.e3);
    std::printf("log(0.9375) = %a\n", (double)out.lg);
    // host reference
    std::printf("host nextafter diff = %a\n",
                (double)(sycl::nextafter(0.05F, 1.0e30F) - 0.05F));
    return 0;
}
