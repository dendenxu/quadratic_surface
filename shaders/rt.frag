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
    vec3 center;
};

uniform Camera cam;

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

    // vec2 fcl_inv = 1.0 / vec2(cam.K[0].x, cam.K[1].y);     // inverse of K
    vec2 fcl_inv = 1.0 / cam.reso;                         // inverse of K
    vec2 pix_scr = gl_FragCoord.xy * fcl_inv * 2.0 - 1.0;  // screen space pixel
    vec2 pix_stp = (1.0 * fcl_inv) / float(samples);       // super sampling substeps

    // mat4 rot_mat = mat4(cam.c2w);
    mat4 rot_mat = mat4(1.0);   // camera to world transform
    vec3 ray_ori = cam.center;  // world space origin

    vec3 result = vec3(0.0);
    for (int y = 0; y < samples; y++) {
        for (int x = 0; x < samples; x++) {
            vec2 pix_sub = pix_scr + pix_stp * vec2(x, y);      // subpixel
            vec3 pix_cam = vec3(pix_sub, 1.0);                  // camera space pixel
            vec3 pix_wld = vec3(rot_mat * vec4(pix_cam, 1.0));  // world space pixel
            vec3 ray_dir = normalize(pix_wld - ray_ori);        // ray direction

            // quadrics
            vec3 pixel = vec3(0.0);
            pixel += draw_quadric(sphere, vec4(ray_ori, 1.0), vec4(ray_dir, 0.0));
            result += clamp(pixel, 0.0, 1.0);
        }
    }
    frag_color = vec4(result / float(samples * samples), 1.0);
}
