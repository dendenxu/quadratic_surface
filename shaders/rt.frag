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
in mat4 p2t;                     // the c2w that map this FragPos to tpose location (which is also in world space)

// The output color
out vec4 frag_color;

// Computer vision style camera
struct Camera {
    mat4x3 c2w;  // c2w
    mat4x4 w2c;  // w2c
    mat4x4 K;
    mat4x4 inv_K;
    vec2 reso;
    vec2 focal;
};

// // Store tree data
// struct N3TreeSpec {
//     int N;
//     int data_dim;    // usable datadim, including extra_slot
//     int extra_slot;  // if not zero, we'll do an offset first
//     int format;
//     int basis_dim;
//     float ndc_width;
//     float ndc_height;
//     float ndc_focal;
//     vec3 center;
//     vec3 scale;
// };

// // Store render options
// struct RenderOptions {
//     // Epsilon added to each step
//     float step_size;
//     // If remaining light intensity/alpha < this amount stop marching
//     float stop_thresh;
//     // If sigma < this, skips
//     float sigma_thresh;
//     // Background brightness
//     float background_brightness;

//     // Rendering bounding box (relative to outer tree bounding box [0, 1])
//     // [minx, miny, minz, maxx, maxy, maxz]
//     float render_bbox[6];
//     // Range of basis functions to use
//     int basis_minmax[2];
//     // Rotation applied to viewdirs for all rays
//     vec3 rot_dirs;

//     float t_off_out;  // how much to render outside the guide mesh
//     float t_off_in;   // how much to render inside the mesh

//     bool use_offset;  // FIXME: should check whether extra_slot is available
//     bool visualize_offset;

//     bool visualize_unseen;
//     bool visualize_intensity;

//     float vis_offset_multiplier;

//     float vis_color_multiplier;
//     float vis_color_offset;

//     bool apply_sigmoid;

//     vec3 probe;
//     float probe_disp_size;

//     bool show_template;
//     bool bigpose_geometry;
// };

uniform Camera cam;
// uniform RenderOptions opt;
// uniform N3TreeSpec tree;

// uniform bool drawing_probe;  // THIS IS AN IMPORTANT SWITCH FOR THE SHADER

// uniform int tree_child_dim;
// uniform highp isampler2D tree_child_tex;
// uniform int tree_data_dim;  // ? you've got the tree.data_dim, why still need this? Or is this just the texture data_dim?
// uniform mediump sampler2D tree_data_tex;

// Mesh rendering compositing
// uniform mediump sampler2D mesh_depth_tex;
// uniform mediump sampler2D mesh_color_tex;

// Ray Tracing Quadric shapes (with box constraint)
// based on the paper: Ray Tracing Arbitrary Objects on the GPU, A. Wood et al
const mat4 cylinder = mat4(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, -0.25);

const mat4 sphere = mat4(
    4.0, 0.0, 0.0, 0.0,
    0.0, 4.0, 0.0, 0.0,
    0.0, 0.0, 4.0, 0.0,
    0.0, 0.0, 0.0, -1.0);

const mat4 ellipticParaboloid = mat4(
    4.0, 0.0, 0.0, 0.0,
    0.0, 4.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
    0.0, 0.0, 1.0, 0.0);

const mat4 hyperbolicParaboloid = mat4(
    4.0, 0.0, 0.0, 0.0,
    0.0, -4.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
    0.0, 0.0, 1.0, 0.0);

const mat4 circularCone = mat4(
    4.0, 0.0, 0.0, 0.0,
    0.0, -4.0, 0.0, 0.0,
    0.0, 0.0, 4.0, 0.0,
    0.0, 0.0, 0.0, 0.0);

const mat4 quadraticPlane = mat4(
    1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 1.0, 0.0, 0.0);

const mat4 hyperbolicPlane = mat4(
    1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 2.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 2.0, 0.0, 0.0);

const mat4 intersectingPlanes = mat4(
    0.0, 1.0, 0.0, 0.0,
    1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0);

const float EPSILON = 0.000001;

const int samples = 4;  // per x,y per fragment

bool getPointAtTime(in float t, in vec4 ro, in vec4 rd, out vec3 point) {
    if (t < 0.0) {
        return false;
    }

    point = ro.xyz + t * rd.xyz;

    // constrain to a box
    return all(greaterThanEqual(point, vec3(-0.5 - EPSILON))) && all(lessThanEqual(point, vec3(0.5 + EPSILON)));
}

// adapted from https://iquilezles.org/articles/intersectors
bool intersect_box(in vec4 ro, in vec4 rd, out vec4 outPos) {
    vec3 m = 1.0 / rd.xyz;
    vec3 n = m * ro.xyz;
    vec3 k = abs(m);
    vec3 t1 = -n - k;
    vec3 t2 = -n + k;
    float tN = max(max(t1.x, t1.y), t1.z);
    float tF = min(min(t2.x, t2.y), t2.z);
    if (tN > tF || tF < 0.0) return false;  // no intersection
    outPos = ro + rd * tN;
    return true;
}

bool intersect_quadric(in mat4 shape, in vec4 ro, in vec4 rd, out vec3 point) {
    vec4 rda = shape * rd;
    vec4 roa = shape * ro;

    // quadratic equation
    float a = dot(rd, rda);
    float b = dot(ro, rda) + dot(rd, roa);
    float c = dot(ro, roa);

    if (abs(a) < EPSILON) {
        if (abs(b) < EPSILON) {
            return getPointAtTime(c, ro, rd, point);
        }

        return getPointAtTime(-c / b, ro, rd, point);
    }

    float square = b * b - 4.0 * a * c;

    if (square < EPSILON) {
        return false;  // no hit
    }

    float temp = sqrt(square);
    float denom = 2.0 * a;

    float t1 = (-b - temp) / denom;
    float t2 = (-b + temp) / denom;

    // draw both sides but pick the closest point
    vec3 p1 = vec3(0.0);
    vec3 p2 = vec3(0.0);

    bool hasP1 = getPointAtTime(t1, ro, rd, p1);
    bool hasP2 = getPointAtTime(t2, ro, rd, p2);

    if (!hasP1) {
        point = p2;
        return hasP2;
    }

    if (!hasP2) {
        point = p1;
        return true;
    }

    if (t1 < t2) {
        point = p1;
    } else {
        point = p2;
    }

    return true;
}

vec3 draw_quadric(in mat4 shape, in vec4 ro, in vec4 rd) {
    vec3 coll_point = vec3(0.0);

    // intersect the bounding box first and use the intersected origin for solving the quadric
    // idea from mla: https://www.shadertoy.com/view/wdlBR2
    if (intersect_box(ro, rd, ro) && intersect_quadric(shape, ro, rd, coll_point)) {
        // some simple fake shading for now
        return normalize((shape * vec4(coll_point, 1.0)).xyz) / 2.0 + 0.5;
    } else {
        // otherwise return black for now
        return vec3(0.0);
    }
}

void main() {
    // https://www.shadertoy.com/view/fl3SDN
    // https://computergraphics.stackexchange.com/questions/5724/glsl-can-someone-explain-why-gl-fragcoord-xy-screensize-is-performed-and-for

    // screen coordinate ray origin
    float aspect_ratio = cam.reso.x / cam.reso.y;
    vec3 ray_position = vec3(0.0, 0.0, -10.0);
    // screen coordiante ray target
    vec3 ray_target = vec3((gl_FragCoord.xy / cam.reso.xy) * 2.0 - 1.0, 1.0);
    ray_target.y /= aspect_ratio;

    vec2 ray_step = (1.0 / cam.reso.xy) / float(samples);
    // mat4 rot_matrix = mat4(cam.w2c);
    mat4 rot_matrix = mat4(1.0);
    vec3 result = vec3(0.0);

    for (int y = 0; y < samples; y++) {
        for (int x = 0; x < samples; x++) {
            vec3 ray_dir = normalize(ray_target + vec3(ray_step * vec2(x, y), 0.0) - ray_position);
            vec4 new_dir = rot_matrix * vec4(ray_dir, 0.0);

            // quadrics
            vec3 pixel = vec3(0.0);

            // pixel += draw_quadric(cylinder, vec4(ray_position - vec3(-3.0, 1.0, 30.0), 1.0) * rot_matrix, new_dir);
            // pixel += draw_quadric(sphere, vec4(ray_position - vec3(-1.5, 1.0, 30.0), 1.0) * rot_matrix, new_dir);
            pixel += draw_quadric(ellipticParaboloid, rot_matrix * vec4(ray_position, 1.0), new_dir);
            // pixel += draw_quadric(hyperbolicParaboloid, vec4(ray_position - vec3(1.5, 1.0, 30.0), 1.0) * rot_matrix, new_dir);
            // pixel += draw_quadric(circularCone, vec4(ray_position - vec3(3.0, 1.0, 30.0), 1.0) * rot_matrix, new_dir);
            // pixel += draw_quadric(quadraticPlane, vec4(ray_position - vec3(-2.0, -1.0, 30.0), 1.0) * rot_matrix, new_dir);
            // pixel += draw_quadric(hyperbolicPlane, vec4(ray_position - vec3(0.0, -1.0, 30.0), 1.0) * rot_matrix, new_dir);
            // pixel += draw_quadric(intersectingPlanes, vec4(ray_position - vec3(2.0, -1.0, 30.0), 1.0) * rot_matrix, new_dir);

            result += clamp(pixel, 0.0, 1.0);
        }
    }

    frag_color = vec4(result / float(samples * samples), 1.0);
}
