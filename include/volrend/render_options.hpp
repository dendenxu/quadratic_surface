#pragma once

#include "volrend/common.hpp"

// Max global basis
#define VOLREND_GLOBAL_BASIS_MAX 25

namespace volrend {

// Rendering options
struct RenderOptions {
    // * BASIC RENDERING
    // Epsilon added to steps to prevent hitting current box again
    float step_size = 1e-4f;  // Ah so this is the 1/eps stuff opt.step_size

    // If a point has sigma < this amount, considers sigma = 0
    float sigma_thresh = 1e-2f;

    // If remaining light intensity/alpha < this amount stop marching
    float stop_thresh = 1e-2f;

    // Background brightness
    float background_brightness = 0.f;  // ? defaults to black, why doesn't this work

    // * VISUALIZATION
    // Rendering bounding box (relative to outer tree bounding box [0, 1])
    // [minx, miny, minz, maxx, maxy, maxz]
    float render_bbox[6] = {0.f, 0.f, 0.f, 1.f, 1.f, 1.f};

    // Range of basis functions to use
    // no effect if RGBA data format
    int basis_minmax[2] = {0, VOLREND_GLOBAL_BASIS_MAX - 1};

    // Rotation applied to viewdirs for all rays
    float rot_dirs[3] = {0.f, 0.f, 0.f};

    // * ADVANCED VISUALIZATION

    // Draw a (rather low-quality) grid to help visualize the octree
    bool show_grid = false;
    // Grid max depth
    int grid_max_depth = 4;
    // Whether to render the guide mesh explicitly
    bool render_guide_mesh = false;
    bool render_dynamic_nerf = true;

    bool show_template = true;

#ifdef VOLREND_CUDA
    // Render depth instead of color, currently CUDA only
    bool render_depth = false;
#endif

    // * Probe for inspecting lumispheres
    bool enable_probe = false;
    float probe[3] = {0.f, 0.f, 1.f};
    int probe_disp_size = 100;

    // * Guided mesh params
    float t_off_out = 0.01f;  // how much to render outside the guide mesh, in meter
    float t_off_in = 0.1f;    // how much to render inside the mesh, in meter

    // * Use offset to render
    bool use_offset = false;

    bool visualize_offset = false;

    bool visualize_unseen = false;
    bool visualize_intensity = false;

    float vis_offset_multiplier = 10.0;

    float vis_color_multiplier = 1.0;

    float vis_color_offset = 0.0;

    bool show_bone_association = false;

    bool apply_sigmoid = true;

    bool bigpose_geometry = true;

    bool use_face_normal = true;
};

}  // namespace volrend
