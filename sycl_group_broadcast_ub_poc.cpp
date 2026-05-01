// Minimal reproducer: sycl::group_broadcast accepted with non-uniform source_id,
// produces wrong results at runtime on Intel PVC with icpx 2025.3.
//
// Build:
//   icpx -fsycl -O2 -o sycl_group_broadcast_ub_poc sycl_group_broadcast_ub_poc.cpp
// Run on a PVC tile:
//   ZE_AFFINITY_MASK=0.0 ./sycl_group_broadcast_ub_poc
//
// Per SYCL 2020 section 4.17.4.3, the source_id argument of group_broadcast
// must be uniform across the group. Passing a per-lane (non-uniform) value is
// undefined behavior. icpx accepts the call with no diagnostic. The correct
// non-uniform shuffle is sycl::select_from_group (SYCL 2020 sub_group ext).

#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>

constexpr int SG = 16;

int main() {
    sycl::queue q{sycl::gpu_selector_v};
    auto dev = q.get_device();

    std::cout << "device: "  << dev.get_info<sycl::info::device::name>() << "\n";
    std::cout << "driver: "  << dev.get_info<sycl::info::device::driver_version>() << "\n";
#ifdef SYCL_LANGUAGE_VERSION
    std::cout << "SYCL_LANGUAGE_VERSION = " << SYCL_LANGUAGE_VERSION << "\n";
#endif
#ifdef __INTEL_LLVM_COMPILER
    std::cout << "__INTEL_LLVM_COMPILER = " << __INTEL_LLVM_COMPILER << "\n";
#endif
#ifdef __VERSION__
    std::cout << "__VERSION__ = " << __VERSION__ << "\n";
#endif

    constexpr int N = SG;
    std::vector<int> bcast(N, -1), shuf(N, -1);
    int sg_width_used = 0;

    {
        sycl::buffer<int, 1> b_bcast(bcast.data(), N);
        sycl::buffer<int, 1> b_shuf (shuf .data(), N);
        sycl::buffer<int, 1> b_width(&sg_width_used, 1);

        q.submit([&](sycl::handler& h) {
            auto a_bcast = b_bcast.get_access<sycl::access_mode::write>(h);
            auto a_shuf  = b_shuf .get_access<sycl::access_mode::write>(h);
            auto a_width = b_width.get_access<sycl::access_mode::write>(h);
            h.parallel_for(sycl::nd_range<1>{sycl::range<1>(N), sycl::range<1>(N)},
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                    auto sg   = it.get_sub_group();
                    int  lane = static_cast<int>(sg.get_local_id()[0]);
                    int  myval = 100 + lane;     // each lane has a unique value
                    int  src   = lane / 2;       // NON-UNIFORM source id (UB for group_broadcast)

                    // SYCL 2020 group function. Spec requires source_id uniform.
                    int via_broadcast = sycl::group_broadcast(sg, myval, src);
                    // SYCL 2020 sub_group shuffle. Spec permits per-lane source_id.
                    int via_select    = sycl::select_from_group(sg, myval, src);

                    a_bcast[lane] = via_broadcast;
                    a_shuf [lane] = via_select;
                    if (lane == 0) a_width[0] = static_cast<int>(sg.get_local_range()[0]);
                });
        }).wait_and_throw();
    }

    std::cout << "subgroup width used = " << sg_width_used << "\n\n";

    int select_mismatches = 0;
    int broadcast_mismatches = 0;
    std::cout << "lane | src=lane/2 | expected | group_broadcast | select_from_group\n";
    std::cout << "-----+------------+----------+-----------------+------------------\n";
    for (int lane = 0; lane < N; ++lane) {
        int src      = lane / 2;
        int expected = 100 + src;
        std::cout << "  " << (lane < 10 ? " " : "") << lane
                  << "  |     " << src << (src < 10 ? " " : "")
                  << "     |    " << expected
                  << "   |       " << bcast[lane]
                  << "       |        " << shuf[lane] << "\n";
        if (shuf[lane]  != expected) select_mismatches++;
        if (bcast[lane] != expected) broadcast_mismatches++;
    }
    std::cout << "\n";

    std::cout << "select_from_group mismatches vs expected per-lane shuffle: "
              << select_mismatches << " (should be 0)\n";
    std::cout << "group_broadcast   mismatches vs per-lane shuffle:          "
              << broadcast_mismatches
              << " (UB per spec; nonzero demonstrates wrong-result UB).\n";

    // Also print whether group_broadcast did at least produce some single uniform
    // value across all lanes (a "lucky" UB outcome) or scattered garbage.
    bool bcast_uniform = true;
    for (int lane = 1; lane < N; ++lane)
        if (bcast[lane] != bcast[0]) { bcast_uniform = false; break; }
    std::cout << "group_broadcast result uniform across lanes? "
              << (bcast_uniform ? "yes" : "no") << "\n";

    if (select_mismatches != 0) {
        std::cerr << "FAIL: select_from_group did not produce the per-lane shuffle.\n";
        return 2;
    }
    // Return non-zero if the UB call produced wrong results (the bug we want to show).
    return broadcast_mismatches > 0 ? 1 : 0;
}
