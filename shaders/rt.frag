#version 330
#pragma vscode_glsllint_stage : frag
precision highp float;
precision highp int;

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

struct Quadric {
    mat4x4 shape;
};

uniform Camera cam;

// Ray Tracing Quadric shapes (with box constraint)
// based on the paper: Ray Tracing Arbitrary Objects on the GPU, A. Wood et al
uniform Quadric qua;

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
        // if (intersect_quadric(shape, ro, rd, coll_point)) {
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
    mat4 rot_matrix = mat4(cam.c2w);
    vec3 result = vec3(0.0);
    for (int y = 0; y < samples; y++) {
        for (int x = 0; x < samples; x++) {
            vec3 ray_dir = normalize(ray_target + vec3(ray_step * vec2(x, y), 0.0) - ray_position);
            vec4 new_dir = rot_matrix * vec4(ray_dir, 0.0);
            vec4 new_pos = rot_matrix * vec4(ray_position, 1.0);

            // quadrics
            vec3 pixel = vec3(0.0);
            pixel += draw_quadric(qua.shape, new_pos, new_dir);
            result += clamp(pixel, 0.0, 1.0);
        }
    }

    frag_color = vec4(result / float(samples * samples), 1.0);
}
