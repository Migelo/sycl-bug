# `sycl::group_broadcast` silently accepts non-uniform `source_id` and returns wrong results on PVC (icpx 2025.3.2)

## Versions

- Compiler: `Intel(R) oneAPI DPC++/C++ Compiler 2025.3.2 (2025.3.2.20260112)` (`__INTEL_LLVM_COMPILER = 20250302`)
- `SYCL_LANGUAGE_VERSION = 202012`
- Device: `Intel(R) Data Center GPU Max 1550` (Ponte Vecchio)
- Level Zero driver: `1.6.33578+42`
- Subgroup width: 16 (`[[sycl::reqd_sub_group_size(16)]]`)
- Build: `icpx -fsycl -O2 -o poc poc.cpp` (no warnings)

## Summary

SYCL 2020 §4.17.4.3 requires the `source_id` argument of `sycl::group_broadcast` to be uniform across the group (it lowers to SPIR-V `OpGroupBroadcast`, which has the same uniformity requirement).

icpx 2025.3.2 accepts a per-lane (non-uniform) `source_id` with **no diagnostic**, and at runtime on PVC silently returns the value of lane 0 to every lane — i.e. it uses one lane's `source_id` for the whole subgroup, no warning, no abort. The correct primitive for per-lane shuffles is `sycl::select_from_group`; both live in `<sycl/sycl.hpp>` and are easy to confuse.

## Minimal reproducer

```cpp
#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>

constexpr int SG = 16;

int main() {
    sycl::queue q{sycl::gpu_selector_v};
    auto dev = q.get_device();
    std::cout << "device: " << dev.get_info<sycl::info::device::name>() << "\n";
    std::cout << "driver: " << dev.get_info<sycl::info::device::driver_version>() << "\n";
    std::cout << "SYCL_LANGUAGE_VERSION = " << SYCL_LANGUAGE_VERSION << "\n";
#ifdef __INTEL_LLVM_COMPILER
    std::cout << "__INTEL_LLVM_COMPILER = " << __INTEL_LLVM_COMPILER << "\n";
#endif

    constexpr int N = SG;
    std::vector<int> bcast(N, -1), shuf(N, -1);
    {
        sycl::buffer<int, 1> b_bcast(bcast.data(), N);
        sycl::buffer<int, 1> b_shuf (shuf .data(), N);
        q.submit([&](sycl::handler& h) {
            auto a_bcast = b_bcast.get_access<sycl::access_mode::write>(h);
            auto a_shuf  = b_shuf .get_access<sycl::access_mode::write>(h);
            h.parallel_for(sycl::nd_range<1>{sycl::range<1>(N), sycl::range<1>(N)},
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                    auto sg   = it.get_sub_group();
                    int  lane = static_cast<int>(sg.get_local_id()[0]);
                    int  myval = 100 + lane;
                    int  src   = lane / 2;          // NON-UNIFORM source id
                    a_bcast[lane] = sycl::group_broadcast(sg, myval, src);
                    a_shuf [lane] = sycl::select_from_group(sg, myval, src);
                });
        }).wait_and_throw();
    }
    for (int lane = 0; lane < N; ++lane) {
        int expected = 100 + (lane / 2);
        std::cout << "lane " << lane << " expected " << expected
                  << " group_broadcast=" << bcast[lane]
                  << " select_from_group=" << shuf[lane] << "\n";
    }
}
```

Build: `icpx -fsycl -O2 -o poc poc.cpp` (no diagnostic). Run: `ZE_AFFINITY_MASK=0.0 ./poc`.

## Expected behavior

Either (a) a compile-time diagnostic, or (b) the per-spec semantics of *some* uniform `source_id` (every lane observes the value of one chosen lane). Silent acceptance with no diagnostic is the problem.

## Observed behavior

```
lane | src=lane/2 | expected | group_broadcast | select_from_group
   0 |     0      |    100   |       100       |        100
   1 |     0      |    100   |       100       |        100
   2 |     1      |    101   |       100       |        101
   3 |     1      |    101   |       100       |        101
   ...
  14 |     7      |    107   |       100       |        107
  15 |     7      |    107   |       100       |        107

select_from_group mismatches: 0
group_broadcast   mismatches: 14
group_broadcast result uniform across lanes? yes
```

`group_broadcast` returned lane 0's value on every lane (one of the spec-allowed uniform behaviors), but the call was accepted silently. The mental model said "shuffle"; the runtime did "broadcast"; no diagnostic.

## Workaround

Use `sycl::select_from_group(sg, value, source_id)` for any per-lane source. The same PoC shows it produces the intended per-lane shuffle (zero mismatches).

## Why it matters

In an internal codebase this pattern appeared inside a particle-sort kernel that derives a destination slot from a subgroup-local exclusive scan. Each lane needed a different lane's value, so we wrote `group_broadcast(sg, base, src_lane)` thinking it shuffled. icpx compiled it without warning. On PVC all lanes received the same dest-slot, indices collided, downstream reads loaded stale data, and several iterations later the kernel took a GPU page fault. Chain: silent accept → wrong-but-plausible result → memory corruption → page fault.

## Suggested fixes

1. Compile-time diagnostic when `group_broadcast`'s `source_id` is provably non-uniform (e.g. derived from `get_local_id`).
2. Header doc-comment pointing users to `select_from_group` for per-lane shuffles.
3. Optional runtime check under a debug flag.
