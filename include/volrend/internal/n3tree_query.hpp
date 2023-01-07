#pragma once
#include "volrend/common.hpp"
#include "volrend/camera.hpp"

#include "half.hpp"

#include "volrend/internal/data_spec.hpp"

namespace volrend {
namespace internal {
namespace {

VOLREND_COMMON_FUNCTION static void query_single_from_root(
    const TreeSpec& tree, float* VOLREND_RESTRICT xyz,
    const half** VOLREND_RESTRICT out, float* VOLREND_RESTRICT cube_sz) {
    const float fN = tree.N; // 2
    // Getting the 0-1 index of the points, clamp it to make computation stable
    xyz[0] = VOLREND_MAX(VOLREND_MIN(xyz[0], 1.f - 1e-6f), 0.f);
    xyz[1] = VOLREND_MAX(VOLREND_MIN(xyz[1], 1.f - 1e-6f), 0.f);
    xyz[2] = VOLREND_MAX(VOLREND_MIN(xyz[2], 1.f - 1e-6f), 0.f);
    int64_t ptr = 0;
    *cube_sz = fN; // 2
    while (true) {
        // Find index of query point, in {0, ... N^3}
        // float index = 4.f * (xyz[2] > 0.5f) + 2.f * (xyz[1] > 0.5f) + (xyz[0]
        // > 0.5f);
        // This is just pin-pointing the location of the actual xyz
        float index = 0.f;
/** 
 * #pragma unroll is a compiler optimization that can, for example, replace a piece of code like
for ( int i = 0; i < 5; i++ )
    b[i] = i;

 * with

b[0] = 0;
b[1] = 1;
b[2] = 2;
b[3] = 3;
b[4] = 4;
 */
#pragma unroll 3
        for (int i = 0; i < 3; ++i) {
            xyz[i] *= fN;
            const float idx_dimi = floorf(xyz[i]);
            index = index * fN + idx_dimi;
            xyz[i] -= idx_dimi;
        }

        // now the index is loaded, we can use it to find the child

        // Find child offset
        const int64_t sub_ptr = ptr + (int32_t)index;
        const int64_t skip = tree.child[sub_ptr]; // what is this child? Number of blocks to skip?

        // Add to output
        if (skip == 0 /* || *cube_sz >= max_cube_sz*/) {
            *out = tree.data + sub_ptr * tree.data_dim;
            break;
        }
        *cube_sz *= fN;

        ptr += skip * tree.N3;
    }
}

}  // namespace
}  // namespace internal
}  // namespace volrend
