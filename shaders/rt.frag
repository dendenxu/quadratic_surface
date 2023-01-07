#version 330
#pragma vscode_glsllint_stage : frag
precision highp float;
precision highp int;

#define VOLREND_GLOBAL_BASIS_MAX 16

#define FORMAT_RGBA 0
#define FORMAT_SH 1
#define FORMAT_SG 2
#define FORMAT_ASG 3

in highp vec4 tpose_cam_center;  // in tpose space
in highp vec4 tpose_frag_pos;    // in tpose space
in highp vec4 deform_frag_pos;   // in tpose space
in highp vec4 world_frag_pos;    // interpolated fragment 3D location, in world space
in mat4 p2t;                     // the transform that map this FragPos to tpose location (which is also in world space)

// The output color
out vec4 frag_color;

// Computer vision style camera
struct Camera {
    mat4x3 transform;
    mat4x4 w2c;
    mat4x4 K;
    vec2 reso;
    vec2 focal;
};

// Store tree data
struct N3TreeSpec {
    int N;
    int data_dim;    // usable datadim, including extra_slot
    int extra_slot;  // if not zero, we'll do an offset first
    int format;
    int basis_dim;
    float ndc_width;
    float ndc_height;
    float ndc_focal;
    vec3 center;
    vec3 scale;
};

// Store render options
struct RenderOptions {
    // Epsilon added to each step
    float step_size;
    // If remaining light intensity/alpha < this amount stop marching
    float stop_thresh;
    // If sigma < this, skips
    float sigma_thresh;
    // Background brightness
    float background_brightness;

    // Rendering bounding box (relative to outer tree bounding box [0, 1])
    // [minx, miny, minz, maxx, maxy, maxz]
    float render_bbox[6];
    // Range of basis functions to use
    int basis_minmax[2];
    // Rotation applied to viewdirs for all rays
    vec3 rot_dirs;

    float t_off_out;  // how much to render outside the guide mesh
    float t_off_in;   // how much to render inside the mesh

    bool use_offset;  // FIXME: should check whether extra_slot is available
    bool visualize_offset;

    bool visualize_unseen;
    bool visualize_intensity;

    float vis_offset_multiplier;

    float vis_color_multiplier;
    float vis_color_offset;

    bool apply_sigmoid;

    vec3 probe;
    float probe_disp_size;

    bool show_template;
    bool bigpose_geometry;
};

uniform Camera cam;
uniform RenderOptions opt;
uniform N3TreeSpec tree;

uniform bool drawing_probe;  // THIS IS AN IMPORTANT SWITCH FOR THE SHADER

uniform int tree_child_dim;
uniform highp isampler2D tree_child_tex;
uniform int tree_data_dim;  // ? you've got the tree.data_dim, why still need this? Or is this just the texture data_dim?
uniform mediump sampler2D tree_data_tex;

// Mesh rendering compositing
uniform mediump sampler2D mesh_depth_tex;
uniform mediump sampler2D mesh_color_tex;

// Hacky ways to store octree in 2 textures
float get_tree_data(int y, int x) {
    return texelFetch(tree_data_tex, ivec2(x, y), 0).r;
}
int index_tree_child(int i) {
    int y = i / tree_child_dim;
    int x = i % tree_child_dim;
    return texelFetch(tree_child_tex, ivec2(x, y), 0).r;
}

// **** N^3 TREE IMPLEMENTATION ****

// Tree query, returns
// (start index of leaf node in tree_data, leaf node scale 2^depth)
int query_single_from_root(inout vec3 xyz, out float cube_sz) {
    // float fN = float(tree.N);
    // int N3 = tree.N * tree.N * tree.N;
    xyz = clamp(xyz, 0.f, 1.f - 1e-6f);
    int ptr = 0;  // starting with location 0
    int sub_ptr = 0;
    vec3 idx;
    for (cube_sz = 1.f; /*cube_sz < 11*/; ++cube_sz) {  // increment cube_sz during query, why?
        idx = floor(xyz * 2.0);                         // [0,1]^3 -> [0,2]^3, then goto the 1 of the 8, easy indexing into the octree

        // Find child offset
        sub_ptr = ptr + int(idx.x * 4.0 + idx.y * 2.0 + idx.z);  // current node + index of child = child node (refer to as indexed node later)
        int skip = index_tree_child(sub_ptr);                    // get number of child for the indexed node
        xyz = xyz * 2.0 - idx;                                   // [0,1]^3 -> [0,2]^3 and minus the lower bound to be [0,1]^3 again, this time in a smaller cube

        // Add to output
        if (skip == 0) {
            break;
        }
        ptr += skip * 8;  //N3;
    }
    cube_sz = 1 / pow(2.0, cube_sz);  // ! why? for precision?
    return sub_ptr;                   // sub_ptr is just the smallest child of the queried xyz
}

void rodrigues(vec3 aa, inout vec3 dir) {
    float angle = length(aa);
    if (angle < 1e-6) return;
    vec3 k = aa / angle;
    float cos_angle = cos(angle);
    float sin_angle = sin(angle);
    vec3 cp = cross(k, dir);
    float dot = dot(k, dir);
    dir = dir * cos_angle + cp * sin_angle + k * dot * (1.0 - cos_angle);
}

// **** CORE RAY TRACER IMPLEMENTATION ****
void maybe_precalc_basis(const vec3 dir, inout float outb[VOLREND_GLOBAL_BASIS_MAX]) {
    {
        outb[0] = 0.28209479177387814;
        float x = dir[0], y = dir[1], z = dir[2];
        float xx = x * x, yy = y * y, zz = z * z;
        float xy = x * y, yz = y * z, xz = x * z;
        switch (tree.basis_dim) {
        case 16:
            outb[9] = -0.5900435899266435 * y * (3.f * xx - yy);
            outb[10] = 2.890611442640554 * xy * z;
            outb[11] = -0.4570457994644658 * y * (4.f * zz - xx - yy);
            outb[12] = 0.3731763325901154 * z * (2.f * zz - 3.f * xx - 3.f * yy);
            outb[13] = -0.4570457994644658 * x * (4.f * zz - xx - yy);
            outb[14] = 1.445305721320277 * z * (xx - yy);
            outb[15] = -0.5900435899266435 * x * (xx - 3.f * yy);
        case 9:
            outb[4] = 1.0925484305920792 * xy;
            outb[5] = -1.0925484305920792 * yz;
            outb[6] = 0.31539156525252005 * (2.f * zz - xx - yy);
            outb[7] = -1.0925484305920792 * xz;
            outb[8] = 0.5462742152960396 * (xx - yy);
        case 4:
            outb[1] = -0.4886025119029199 * y;
            outb[2] = 0.4886025119029199 * z;
            outb[3] = -0.4886025119029199 * x;
        }
    }
}

void dda_world(vec3 cen, vec3 _invdir, out float tmin, out float tmax) {
    // AABB for user input
    float t1, t2;
    tmin = 0.0f;
    tmax = 1e9f;
    for (int i = 0; i < 3; ++i) {
        // computing AABB for the bounding box
        t1 = (opt.render_bbox[i] - cen[i]) * _invdir[i];
        t2 = (opt.render_bbox[i + 3] - cen[i]) * _invdir[i];
        tmin = max(tmin, min(t1, t2));
        tmax = min(tmax, max(t1, t2));
    }
}

void dda_unit(vec3 cen, vec3 _invdir, out float tmax) {
    // AABB for [0,1]^3
    float t1, t2;
    tmax = 1e9f;
    for (int i = 0; i < 3; ++i) {
        t1 = -cen[i] * _invdir[i];  // (0 - cen[i]) / dir[i], calculating t1 for intersection
        t2 = t1 + _invdir[i];       // (1 - cen[i]) / dir[i], calculating t2 for intersection
        tmax = min(tmax, max(t1, t2));
    }
}

void _get_scaled(vec3 scaling, inout vec3 offset) {
    offset *= scaling;
}

float _get_delta_scale(vec3 scaling, inout vec3 dir) {
    // FIXME: this function not only computes the delta_scale (what 1 will become in this world), but also updates the dir`
    // Should be extracted as two functional blocks
    dir *= scaling;                         // transform world direction to tree direction in [0,1]^3
    float delta_scale = 1.0 / length(dir);  // compute what the original length one becomes in this direction
    dir *= delta_scale;                     // normalize again
    return delta_scale;                     // this should be used later in composition of the volume density
}

void retrieve_cursor_lumisphere_kernel(out int tree_y, out int tree_x) {
    // everyone has to do this query, might not be so efficient to write it in a fragment shader?
    float cube_sz;
    vec3 pos = tree.center + opt.probe * tree.scale;                     // elementwise scaling
    int doffset = query_single_from_root(pos, cube_sz) * tree.data_dim;  // tree.data_dim is just the int size of data stored at each leaf
    // Note that this call will also tell us the size of the cube we're in
    tree_y = doffset / tree_data_dim;  // using texture to store data, get row number
    tree_x = doffset % tree_data_dim;  // get colume number
}

vec3 trace_ray(vec3 dir, vec3 vdir, vec3 cen, float tmin_out, float tmax_bg, vec3 bg_color) {
    float delta_scale = _get_delta_scale(tree.scale, dir);  // what 1 in world becomes in this direction, this also updates the dir input to the distorted [0,1]^3
    vec3 output_color;                                      // output
    vec3 invdir = 1.f / (dir + 1e-9);                       // for ease of computation
    float tmin, tmax;                                       // ray casting min/max
    dda_world(cen, invdir, tmin, tmax);                     // consider user specified bb_min and bb_max (do an AABB)
    tmin = max(tmin, tmin_out / delta_scale);               // depth map of already rendered meshes
    tmax = min(tmax, tmax_bg / delta_scale);                // depth map of already rendered meshes

    if (tmax < 0.f || tmin > tmax || tree_data_dim == 0) {
        // Ray doesn't hit box or tree not loaded
        if (opt.visualize_unseen) {
            output_color = vec3(0.8118, 0.2431, 0.2431);
        } else {
            output_color = bg_color;
        }
    } else {
        output_color = vec3(.0f);
        float basis_fn[VOLREND_GLOBAL_BASIS_MAX];
        maybe_precalc_basis(vdir, basis_fn);
        for (int i = 0; i < opt.basis_minmax[0]; ++i) {
            basis_fn[i] = 0.f;
        }
        for (int i = opt.basis_minmax[1] + 1; i < VOLREND_GLOBAL_BASIS_MAX; ++i) {
            basis_fn[i] = 0.f;
        }

        float light_intensity = 1.f;
        float t = tmin;
        // int n_steps = 0;
        while (t < tmax) {
            // ++n_steps;
            vec3 location = cen + t * dir;  // from center, get the actual position to sample
            vec3 pos = location;
            float cube_sz;

            // PlenOctree will try to locate a box for the current ray position
            // Note that this modifies both pos and cube_sz
            int doffset = query_single_from_root(pos, cube_sz) * tree.data_dim;  // tree.data_dim is just the int size of data stored at each leaf
            // Note that this call will also tell us the size of the cube we're in
            int tree_y = doffset / tree_data_dim;  // using texture to store data, get row number
            int tree_x = doffset % tree_data_dim;  // get colume number

            // Computing the stopping tmax in terms of the current 3D positoin in [0,1]^3
            float subcube_tmax;
            dda_unit(pos, invdir, subcube_tmax);

            // Getting the current cube tmax, yeah, just as the name suggests, subcube_tmax
            float t_subcube = subcube_tmax * cube_sz;  // getting the tmax to get beyond the current cube, this is interesting...

            // APPLYING OFFSET JUST BEFORE THE QUERY
            // TODO: reconsider this, should we jump through the current box?
            // FIXME: Current assuming the offset is always 3D
            if (tree.extra_slot != 0 && opt.use_offset) {
#define GET_OFFSET_COMPONENT(off) get_tree_data(tree_y, tree_x + tree.data_dim - tree.extra_slot + off)
                vec3 offset = vec3(GET_OFFSET_COMPONENT(0), GET_OFFSET_COMPONENT(1), GET_OFFSET_COMPONENT(2));
                if (opt.visualize_offset) {
                    return abs(offset) * opt.vis_offset_multiplier;
                }
                _get_scaled(tree.scale, offset);
                pos = location + offset;  // get the query again
                doffset = query_single_from_root(pos, cube_sz) * tree.data_dim;
                tree_y = doffset / tree_data_dim;  // using texture to store data, get row number
                tree_x = doffset % tree_data_dim;  // get colume number

                // dda_unit(pos, invdir, subcube_tmax);
                // t_subcube = min(subcube_tmax * cube_sz, t_subcube);
            }

            // if we just use opt.step_size, the rendered image would look nearly the same bu 2 fps
            float delta_t = t_subcube + opt.step_size;  // get beyond current cube and do another sampling
            // Thus, everytime, this value should be just the same...
            // And it should be AABB_world * cube_size
            // but, isn't this approach

            float sigma = get_tree_data(tree_y, tree_x + tree.data_dim - tree.extra_slot - 1);  // query texture
            if (sigma > opt.sigma_thresh) {
                // this is computing exp(-\sigma_j \delta_j)
                // ! trying to hard-code this to 0.005 to match the network
                // float att = min(exp(-0.005 * delta_scale * sigma), 1.f);    // transparency of this block
                float att = min(exp(-delta_t * delta_scale * sigma), 1.f);  // transparency of this block
                // this is computing T_i * exp(-\sigma_j \delta_j) for current ray segment
                float weight = light_intensity * (1.f - att);  // get current opacity * accumulated transparency

                if (tree.format != FORMAT_RGBA) {
                    int off = tree_x;
#define MUL_BASIS_I(t) basis_fn[t] * get_tree_data(tree_y, off + t)
                    for (int t = 0; t < 3; ++t) {
                        float tmp = basis_fn[0] * get_tree_data(tree_y, off);
                        switch (tree.basis_dim) {
                        case 16:
                            tmp += MUL_BASIS_I(9) +
                                   MUL_BASIS_I(10) +
                                   MUL_BASIS_I(11) +
                                   MUL_BASIS_I(12) +
                                   MUL_BASIS_I(13) +
                                   MUL_BASIS_I(14) +
                                   MUL_BASIS_I(15);

                        case 9:
                            tmp += MUL_BASIS_I(4) +
                                   MUL_BASIS_I(5) +
                                   MUL_BASIS_I(6) +
                                   MUL_BASIS_I(7) +
                                   MUL_BASIS_I(8);

                        case 4:
                            tmp += MUL_BASIS_I(1) +
                                   MUL_BASIS_I(2) +
                                   MUL_BASIS_I(3);
                        }

                        // Maybe apply activation?
                        if (opt.apply_sigmoid) {
                            tmp = 1 / (1.0 + exp(-tmp));
                        }

                        // Apply the offset and multiplier
                        tmp = opt.vis_color_multiplier * tmp + opt.vis_color_offset;

                        // Actually accumulate the color
                        output_color[t] += weight * tmp;

                        // go to next
                        off += tree.basis_dim;
                    }
                } else {
                    // for RGBA, we're still accumulating as if provided volume density
                    for (int t = 0; t < 3; ++t) {
                        // ! some modification here
                        float tmp = get_tree_data(tree_y, tree_x + t);
                        // ! Maybe apply activation?
                        // FIXME: previous implementation does not apply sigmoid activation after the color is out
                        if (opt.apply_sigmoid) {
                            tmp = 1 / (1.0 + exp(-tmp));
                        }
                        tmp = opt.vis_color_multiplier * tmp + opt.vis_color_offset;
                        output_color[t] += weight * tmp;
                    }
                }

                // accumulating trasparency
                light_intensity *= att;                   // actually, this is accumulated transparency
                if (light_intensity < opt.stop_thresh) {  // if the accumulated transparency is small enough stop and of course we should stop rendering if it's current ly zero
                    // Almost full opacity, stop
                    output_color *= 1.f / (1.f - light_intensity);  // thinking about what the color would be like if we've got full opacity, probability stuff
                    light_intensity = 0.f;
                    break;
                }
            }
            t += delta_t;  // this is actually to move beyond this box other than taking some fixed step
        }
        if (opt.visualize_intensity) {
            return vec3(1 - light_intensity);
        } else {
            output_color += light_intensity * bg_color;  // mix with background color (rendered mesh)
        }
    }
    return output_color;
}

vec4 draw_probe(out bool drawn) {
    // DRAWING PROBE
    int tree_y, tree_x;
    retrieve_cursor_lumisphere_kernel(tree_y, tree_x);
    float basis_fn[VOLREND_GLOBAL_BASIS_MAX];

    float x = gl_FragCoord.x, y = gl_FragCoord.y;
    float width = cam.reso[0], height = cam.reso[1];
    float xx = x - (width - opt.probe_disp_size - 5);
    float yy = y - (height - opt.probe_disp_size - 5);

    vec3 cen;
    cen[0] = -(xx / (0.5f * opt.probe_disp_size) - 1.f);
    cen[1] = -(yy / (0.5f * opt.probe_disp_size) - 1.f);
    float c = cen[0] * cen[0] + cen[1] * cen[1];
    cen[2] = -sqrt(1 - c);            // compute the direction
    cen = mat3(cam.transform) * cen;  // apply camera transformation

    vec4 output_color;

    if (c <= 1.f) {
        if (tree.basis_dim >= 0) {
            maybe_precalc_basis(cen, basis_fn);
            for (int t = 0; t < 3; ++t) {
                int off = t * tree.basis_dim;
                float tmp = 0.f;
                for (int i = opt.basis_minmax[0]; i <= opt.basis_minmax[1]; ++i) {
                    tmp += basis_fn[i] * get_tree_data(tree_y, tree_x + off + i);
                }
                // Maybe apply activation?
                if (opt.apply_sigmoid) {
                    tmp = 1 / (1.0 + exp(-tmp));
                }

                // Apply the offset and multiplier
                tmp = opt.vis_color_multiplier * tmp + opt.vis_color_offset;
                output_color[t] = tmp;
            }
            output_color[3] = 1.f;
        } else {
            for (int i = 0; i < 3; ++i) {
                float tmp = get_tree_data(tree_y, tree_x + i);
                // Maybe apply activation?
                if (opt.apply_sigmoid) {
                    tmp = 1 / (1.0 + exp(-tmp));
                }

                // Apply the offset and multiplier
                tmp = opt.vis_color_multiplier * tmp + opt.vis_color_offset;
                output_color[i] = tmp;
            }
            output_color[3] = 1.f;
        }
        drawn = true;
    } else {
        drawn = false;
        output_color[0] = output_color[1] = output_color[2] = output_color[3] = 0.f;  // no output_color if not quering
    }
    return output_color;
}

void main() {
    if (drawing_probe) {
        bool drawn;
        vec4 color = draw_probe(drawn);
        if (!drawn) discard;
        // frag_color = vec4(1.0, 0, 0, 1.0);
        frag_color = color;
    } else {
        float tmax_bg = 1e9;
        float tmin_fg = 0;
        vec3 rgb;
        vec3 dir;
        vec3 vdir;
        vec3 cen;

        // Get depth of drawn meshes
        ivec2 screen_pt = ivec2(gl_FragCoord.x, gl_FragCoord.y);
        // mesh_depth was computed using length(CamFragPos.xyz), thus this is not actually a depth but a t for camera cen and unit direction (which is the length to the object) (length is preserved in rigid transformation)
        float tmax_mesh = texelFetch(mesh_depth_tex, screen_pt, 0).r;  // mesh's projection is just effectively the ray casting process
        vec4 mesh_color = texelFetch(mesh_color_tex, screen_pt, 0);
        vec3 bg_color = vec3(mesh_color);

        if (opt.show_template) {
            vec2 xy = (vec2(gl_FragCoord) - 0.5 * cam.reso + vec2(-0.5, 0.5)) / cam.focal;
            dir = normalize(vec3(xy, -1.0));
            dir = normalize(mat3(cam.transform) * dir);
            cen = cam.transform[3];

            vdir = dir;
            cen = tree.center + cen * tree.scale;
            rodrigues(opt.rot_dirs, vdir);
            tmin_fg = 0;
            tmax_bg = tmax_mesh;
        } else {
            // if (opt.show_template) {
            //     frag_color = vec4(0, 0, 1, 1);
            //     return;
            // }
            // vec3 shadeNormal = vec3(0, 0, 1);
            // vec3 xTangent = dFdx(vec3(world_frag_pos));
            // vec3 yTangent = dFdy(vec3(world_frag_pos));
            // shadeNormal = normalize(cross(xTangent, yTangent));
            // frag_color = vec4(shadeNormal * 0.5 + vec3(0.5), 0.5);  // transform [-1,1] to [0, 1]
            // return;

            // gl_FragCoord is in viewport coordinates
            // ! why -0.5, 0.5 here?
            // vec2 xy = (vec2(gl_FragCoord) - 0.5 * cam.reso + vec2(-0.5, 0.5)) / cam.focal;  // (x, y, -1) in the original camera space
            // Camera to NDC: cam.focal * xy / z
            // NDC to viewport: xy * wh + wh/2

            // thus the vector itself is the ray direction
            // you just have to orient it to the world coordinates

            vec3 pos = vec3(tpose_frag_pos);  // interpolated position in tpose coords
            cen = vec3(tpose_cam_center);     // OpenGL default to column major, this is getting the last column

            // and the transform is just the camera to world transformation, whose third column is the center of the camera in world coordinates
            // now we have the camera center
            // vec3 dir = normalize(mat3(cam.transform) * vec3(xy, -1.0));  // length of 1 // cam2world

            vec3 ray = pos - cen;
            float t = length(ray);               // this is the t to go from the camera center to the world position
            float tmin_out = t - opt.t_off_out;  // 10mm for 1cm
            float tmax_in = t + opt.t_off_in;    // 10mm for 1cm
            dir = ray / t;                       // length of 1

            vdir = dir;                            // view direction and ray direction (they should correspond)
            cen = tree.center + cen * tree.scale;  // offset the camera center in tree's coordinate system, which just involves a scaling and translation: [0,1]^3
            pos = tree.center + pos * tree.scale;  // similar to the camera center, transform it to tree coordinate system: [0,1]^3
            // TODO: make FragPos a standard vec3 to reduce memory footprint
            // TODO: actually we shouln't be needing the camera center no more
            // now the cen variable and dir/vdir variables are all in tree's coordinate system
            rodrigues(opt.rot_dirs, vdir);
            // ! this will follow the mesh strickly thus there's gonna be some missing part
            tmin_fg = tmin_out;
            tmax_bg = min(tmax_in, tmax_mesh);
        }

        rgb = trace_ray(dir, vdir, cen, tmin_fg, tmax_bg, bg_color);
        rgb = clamp(rgb, 0.0, 1.0);
        frag_color = vec4(rgb, 1.0);
    }
}
