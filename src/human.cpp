#include "volrend/human.hpp"

#include <cnpy.h>
#include <tiny_obj_loader.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>

#include "half.hpp"
#include "volrend/common.hpp"
#include "volrend/internal/shader.hpp"

namespace volrend {

template <class scalar_t>
void _cross3(const scalar_t* a, const scalar_t* b, scalar_t* out) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

template <class scalar_t>
scalar_t _dot3(const scalar_t* a, const scalar_t* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

template <typename scalar_t>
scalar_t _norm(scalar_t* dir) {
    return sqrtf(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
}

template <typename scalar_t>
void _normalize(scalar_t* dir) {
    scalar_t norm = _norm(dir);
    if (norm > 1e-6) {
        dir[0] /= norm;
        dir[1] /= norm;
        dir[2] /= norm;
    }
}

void estimate_normals(std::vector<float>& verts, std::vector<float>& normal, const std::vector<int>& faces) {
    // Init some reused memory
    glm::vec3 a, b, cross(0, 0, 1);
    unsigned int off[3];

    // For every vertex group, the normal (located at the end of the 9 values) should init to zero
    for (int i = 0; i < normal.size() / NORM_SZ; ++i) {
        for (int j = 0; j < 3; ++j) normal[i * NORM_SZ + j] = 0.f;
    }

    // Loop through each face, with three vertices
    for (int i = 0; i < faces.size() / 3; ++i) {
        off[0] = faces[3 * i] * VERT_SZ;
        off[1] = faces[3 * i + 1] * VERT_SZ;
        off[2] = faces[3 * i + 2] * VERT_SZ;

        // I'm guessing they're not pointing outwards
        // Get vector B-A
        a = glm::make_vec3(&verts[off[1]]) - glm::make_vec3(&verts[off[0]]);
        // Get vector C-A
        b = glm::make_vec3(&verts[off[2]]) - glm::make_vec3(&verts[off[0]]);

        // // Find their cross product
        // prev = cross;
        cross = glm::cross(a, b);

        // For every associated vectex, accumulate the normal for later normalization
        for (int j = 0; j < 3; ++j) {
            float* ptr = &normal[off[j]];
            for (int k = 0; k < 3; ++k) {
                ptr[k] = cross[k];
            }
        }
    }

    for (int i = 0; i < normal.size() / NORM_SZ; ++i) {
        _normalize(&normal[i * NORM_SZ]);
    }
}

#define NUM_POSE_BONE 24
#define NUM_BONE 4

const char* VERT_SHADER_SRC =
    R"glsl(
#version 330
#pragma vscode_glsllint_stage: vert
#define NUM_POSE_BONE 24
#define NUM_BONE 4

uniform mat4 K;
uniform mat4 MV;
uniform mat4 M; // model transform
uniform mat4 V; // this is currently useless due to MV
uniform bool bigpose_geometry;

uniform mat4x4 rigid[NUM_POSE_BONE]; // pose should be a set of 24 affine transfromation
uniform mat4x4 bigrigid[NUM_POSE_BONE]; // pose should be a set of 24 affine transfromation
// this transformation should map a point in tpose to world

layout(location=0) in vec3 a_pos; // location 0
layout(location=1) in vec3 a_deform;
layout(location=2) in vec3 a_normal;
layout(location=3) in vec4 a_blend_weights; // blending weight
layout(location=4) in vec4 a_bone_assoc; // bone index for blending weight


out lowp vec3 vert_color;
out highp vec4 world_frag_pos;
out highp vec4 view_frag_pos;
out highp vec3 world_normal;

void main()
{
    // linear blend skinning
    mat4 t2p = mat4(0);
    float bw_sum = 0.f;
    vec4 pos = vec4(a_pos + a_deform, 1.f); // in object space
    // vec4 pos = vec4(a_pos, 1.f); // in object space
    // vec4 pos = vec4(a_deform, 1.f); // in object space
    for (int i = 0; i < NUM_BONE; i++) {
        bw_sum += a_blend_weights[i]; // if this doesn't sums up to one, we normalize it
        t2p += a_blend_weights[i] * rigid[int(a_bone_assoc[i])];
    }
    t2p /= bw_sum; // normalize


    if (bigpose_geometry) {
        mat4 b2t = mat4(0);
        bw_sum = 0;
        for (int i = 0; i < NUM_BONE; i++) {
            bw_sum += a_blend_weights[i]; // if this doesn't sums up to one, we normalize it
            b2t += a_blend_weights[i] * inverse(bigrigid[int(a_bone_assoc[i])]);
        }
        b2t /= bw_sum; // normalize
        t2p = t2p * b2t;
    }

    pos = t2p * pos;

    world_frag_pos = M * pos; // now FragPos is in world
    view_frag_pos = V * world_frag_pos; // camera space coords
    world_normal = normalize(mat3x3(M) * a_normal);

    vert_color = vec3(a_bone_assoc[0], a_bone_assoc[1], a_bone_assoc[2]) / NUM_POSE_BONE; // ! This one looks pretty cool!
    gl_Position = K * view_frag_pos; // doing a perspective projection to clip space
}
)glsl";

const char* FRAG_SHADER_SRC =
    R"glsl(
#version 330
#pragma vscode_glsllint_stage: frag
precision highp float;

in lowp vec3 vert_color;
in vec4 world_frag_pos; // Note: this is in world space
in vec4 view_frag_pos; // Note: this is in camera space
in vec3 world_normal; // Note: this is in world space

uniform mat4 V; // this is currently useless due to MV
uniform bool unlit;
uniform bool use_face_normal;
uniform bool bone_association;
uniform vec3 world_cam_center;

layout(location = 0) out lowp vec4 frag_color;
layout(location = 1) out float depth;

void main()
{
    vec3 shade_normal = vec3(0, 0, 1);
    if (use_face_normal) {
        vec3 xTangent = dFdx(vec3(world_frag_pos));
        vec3 yTangent = dFdy(vec3(world_frag_pos));
        shade_normal = normalize(cross(xTangent, yTangent));
        shade_normal = vec3(transpose(inverse(V)) * vec4(shade_normal, 1.0));
    } else {
        shade_normal = vec3(transpose(inverse(V)) * vec4(world_normal, 1.0));
    }
    if (unlit) {
        if (bone_association) {
            frag_color = vec4(vert_color, 1);
        } else {
            frag_color = vec4(shade_normal * 0.5 + vec3(0.5), 0.5); // transform [-1,1] to [0, 1]
        }
    } else {
        // FIXME make these uniforms, whatever for now
        float ambient = 0.3;
        float specular_strength = 0.6;
        float diffuse_strength = 0.7;
        float diffuse2_strength = 0.2;
        vec3 lightdir = normalize(vec3(0.5, 0.2, 1));
        vec3 lightdir2 = normalize(vec3(-0.5, -1.0, -0.5));

        float diffuse = diffuse_strength * max(dot(lightdir, shade_normal), 0.0);
        float diffuse2 = diffuse2_strength * max(dot(lightdir2, shade_normal), 0.0);

        vec3 viewdir = normalize(world_cam_center - vec3(world_frag_pos)); // ! is this really ok? FragPos was in world coords
        vec3 reflectdir = reflect(-lightdir, shade_normal);
        float spec = pow(max(dot(viewdir, reflectdir), 0.0), 32.0);
        float specular = specular_strength * spec;

        frag_color = (ambient + diffuse + diffuse2 + specular) * vec4(vert_color, 1);
    }

    depth = length(view_frag_pos.xyz);
}
)glsl";

}  // namespace volrend

namespace volrend {
Human::Human(int n_verts, int n_faces)
    : verts(n_verts * DATA_SZ),
      faces(n_faces * FACE_SZ) {
    if (program == -1) {
        program = create_shader_program(VERT_SHADER_SRC, FRAG_SHADER_SRC);
        u.MV = glGetUniformLocation(program, "MV");
        u.M = glGetUniformLocation(program, "M");
        u.V = glGetUniformLocation(program, "V");
        u.K = glGetUniformLocation(program, "K");
        u.cam_pos = glGetUniformLocation(program, "world_cam_center");
        u.unlit = glGetUniformLocation(program, "unlit");
        u.rigid = glGetUniformLocation(program, "rigid");
        u.bigrigid = glGetUniformLocation(program, "bigrigid");
        u.bone_association = glGetUniformLocation(program, "bone_association");
        u.use_face_normal = glGetUniformLocation(program, "use_face_normal");
        u.bigpose_geometry = glGetUniformLocation(program, "bigpose_geometry");
    }

    bigposes[1][2] = (float)glm::radians(30.0);
    bigposes[2][2] = -(float)glm::radians(30.0);
    // for (int i = 0; i < rigid.size(); i++) {
    //     rigid[i] = glm::mat4(1.0f);
    // }
    // transform = glm::mat4(1.0f);
}

void Human::update() {
    glGenVertexArrays(1, &vao_);
    glGenBuffers(1, &vbo_);
    glGenBuffers(1, &ebo_);

    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(verts[0]), verts.data(), GL_DYNAMIC_DRAW);

    // prepare the vertex array
    // clang-format off
    glVertexAttribPointer(0, VERT_SZ, GL_FLOAT, GL_FALSE, 0, (void*)(sizeof(verts[0]) * 0                                                    ));
    glVertexAttribPointer(1, DEFO_SZ, GL_FLOAT, GL_FALSE, 0, (void*)(sizeof(verts[0]) * (n_verts() * (VERT_SZ))                              ));
    glVertexAttribPointer(2, NORM_SZ, GL_FLOAT, GL_FALSE, 0, (void*)(sizeof(verts[0]) * (n_verts() * (VERT_SZ + DEFO_SZ))                    ));
    glVertexAttribPointer(3, BWEI_SZ, GL_FLOAT, GL_FALSE, 0, (void*)(sizeof(verts[0]) * (n_verts() * (VERT_SZ + DEFO_SZ + NORM_SZ))          ));
    glVertexAttribPointer(4, BONE_SZ, GL_FLOAT, GL_FALSE, 0, (void*)(sizeof(verts[0]) * (n_verts() * (VERT_SZ + DEFO_SZ + NORM_SZ + BWEI_SZ))));
    // clang-format on

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    glEnableVertexAttribArray(3);
    glEnableVertexAttribArray(4);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, faces.size() * sizeof(faces[0]), faces.data(), GL_STATIC_DRAW);
    glBindVertexArray(0);
}

void Human::use_shader() { glUseProgram(program); }

void Human::draw(const glm::mat4x4& V, glm::mat4x4 K, const RenderOptions& options) const {
    glm::vec3 cam_pos = -glm::transpose(glm::mat3x3(V)) * glm::vec3(V[3]);
    glm::mat4x4 MV = V * transform;
    // clang-format off
    glUniformMatrix4fv(u.MV, 1, GL_FALSE, glm::value_ptr(MV));                         // o2c
    glUniformMatrix4fv(u.M, 1, GL_FALSE, glm::value_ptr(transform));                   // o2w
    glUniformMatrix4fv(u.V, 1, GL_FALSE, glm::value_ptr(V));                           // w2c
    glUniformMatrix4fv(u.K, 1, GL_FALSE, glm::value_ptr(K));                           // c2clip
    glUniformMatrix4fv(u.rigid, POSE_BONE_SZ, GL_FALSE, (GLfloat*)(&rigid[0]));        // rigid transformation
    glUniformMatrix4fv(u.bigrigid, POSE_BONE_SZ, GL_FALSE, (GLfloat*)(&bigrigid[0]));  // rigid transformation
    glUniform3fv      (u.cam_pos, 1, glm::value_ptr(cam_pos));
    glUniform1i       (u.unlit, true);
    glUniform1i       (u.use_face_normal, options.use_face_normal);
    glUniform1i       (u.bone_association, options.show_bone_association);
    glUniform1i       (u.bigpose_geometry, options.bigpose_geometry);
    // clang-format on

    glBindVertexArray(vao_);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_);
    glDrawElements(GL_TRIANGLES, (GLsizei)faces.size(), GL_UNSIGNED_INT, (void*)0);

    glBindVertexArray(0);
}

std::vector<glm::mat4> get_rigid_transform(const std::vector<glm::vec3>& tjoints, const std::vector<int>& parents, const std::vector<glm::vec3>& poses) {
    // we should've got the poses and tjoints

    // update size if is the first time
    auto rigid = std::vector<glm::mat4>(POSE_BONE_SZ);

    // prepare the first transform
    rigid[0] = glm::mat4(1.0f);                 // eye
    rigid[0][3] = glm::vec4(tjoints[0], 1.0f);  // assign translation

    // accumulate transformation
    glm::mat4 current;
    glm::vec3 r;
    float norm;
    glm::quat rot;
    glm::vec3 diff;
    // MARK: assumption, poses are already topologically sorted
    for (int i = 1; i < POSE_BONE_SZ; i++) {
        r = poses[i];           // axis angle
        norm = glm::length(r);  // angle
        if (norm < 1e-3) {
            current = glm::mat4(1.0);
        } else {
            rot = glm::angleAxis(norm, r / norm);  // axis and angle
            current = glm::mat4_cast(rot);         // ! this convert quaternion, not axis angle
        }
        // MARK: for speed we do this inside one for loop
        diff = tjoints[i] - tjoints[parents[i]];  // rel_joints
        current[3] = glm::vec4(diff, 1.0f);       // fill the rot_mat
        rigid[i] = rigid[parents[i]] * current;   // update the chain
    }

    // Now every tranform has been accumulated
    glm::vec3 rot_joint;
    for (int i = 0; i < POSE_BONE_SZ; i++) {
        rot_joint = -glm::mat3(rigid[i]) * tjoints[i];  // just the rotation
        rigid[i][3] += glm::vec4(rot_joint, 0.0f);      // add the translation back
    }

    return rigid;
}

void Human::select_pose(int index) {
    update_pose(index);
    update_verts(index);
}

void Human::update_pose(int index) {
    update_rigid(index);
    update_transform(index);
}

void Human::update_verts(int index) {
    size_t n_poses = blend_shapes.size() / (n_verts() * VERT_SZ);
    if (n_poses > index) {
        for (int i = 0; i < n_verts(); i++) {
            // fill vertices
            for (int j = 0; j < DEFO_SZ; ++j) {
                verts[(n_verts() * VERT_SZ) + i * DEFO_SZ + j] = blend_shapes[index * n_verts() * DEFO_SZ + i * DEFO_SZ + j];
            }
        }
        glBindBuffer(GL_ARRAY_BUFFER, vbo_);
        // For this example where the argument to sizeof() operator is called on a value type, the sizeof() operator is executed at compile time and so vecs[0] can never cause a segfault or crash.
        glBufferSubData(
            GL_ARRAY_BUFFER,
            (sizeof(verts[0]) * (n_verts() * VERT_SZ)),
            n_verts() * DEFO_SZ * sizeof(verts[0]),
            n_verts() * VERT_SZ + verts.data());  // the starting point shoule not come in bytes...
    }
}

void Human::update_transform(int index) {
    if (all_rs.size() > index && all_ts.size() > index) {
        glm::vec3 r = all_rs[index];
        glm::vec3 t = all_ts[index];
        float norm = glm::length(r);
        if (norm < 1e-3) {
            transform = glm::mat4(1.0);
        } else {
            glm::quat rot = glm::angleAxis(norm, r / norm);
            transform = glm::mat4_cast(rot);
        }
        transform[3] = glm::vec4(t, 1);
    }
}

void Human::update_rigid(int index) {
    if (all_poses.size() / POSE_BONE_SZ > index) {
        auto p_poses = &all_poses[index * POSE_BONE_SZ];
        auto poses = std::vector<glm::vec3>(p_poses, p_poses + POSE_BONE_SZ);
        rigid = get_rigid_transform(tjoints, parents, poses);
    }
}

void Human::update_bigrigid() {
    bigrigid = get_rigid_transform(tjoints, parents, bigposes);
}

std::vector<glm::mat4> Human::get_bigrigid() {
    return bigrigid;
}

void Human::load_poses_npz(const std::string& path) {
    auto npz = cnpy::npz_load(path);

    // Poses
    if (npz.count("all_poses") && npz.count("all_rs") && npz.count("all_ts")) {
        // TODO: if double, get's broken
        auto p_all_poses = npz["all_poses"].data<glm::vec3>();  // element viewed as one glm::vec3
        size_t n_poses = npz["all_poses"].shape[0];             // number of pose vec3
        size_t n_bones = npz["all_poses"].shape[1];             // number of pose vec3
        all_poses.resize(n_poses * POSE_BONE_SZ);
        for (int i = 0; i < n_poses; i++) {
            std::move(p_all_poses + i * n_bones,
                      p_all_poses + i * n_bones + POSE_BONE_SZ,
                      all_poses.begin() + i * POSE_BONE_SZ);
        }

        // TODO: check whether pose count and r/t count match
        auto p_all_rs = npz["all_rs"].data<glm::vec3>();  // element viewed as one glm::vec3
        all_rs.resize(n_poses);
        std::move(p_all_rs, p_all_rs + n_poses, all_rs.begin());

        auto p_all_ts = npz["all_ts"].data<glm::vec3>();  // element viewed as one glm::vec3
        all_ts.resize(n_poses);
        std::move(p_all_ts, p_all_ts + n_poses, all_ts.begin());
    }

    // Deformations
    if (npz.count("blend_shapes")) {
        // TODO: check if deformations count match && deformation size match with the actual vertices
        auto p_blend_shapes = npz["blend_shapes"].data<float>();  // element viewed as one glm::vec3
        size_t n_poses = npz["blend_shapes"].shape[0];            // number of pose vec3
        size_t n_verts = npz["blend_shapes"].shape[1];            // number of pose vec3
        size_t n_loaded_verts = verts.size() / DATA_SZ;
        blend_shapes.resize(n_poses * n_loaded_verts * VERT_SZ);
        for (int i = 0; i < n_poses; i++) {
            std::move(p_blend_shapes + i * n_verts * VERT_SZ,
                      p_blend_shapes + i * n_verts * VERT_SZ + n_loaded_verts * VERT_SZ,
                      blend_shapes.begin() + i * n_loaded_verts * VERT_SZ);
        }
    }

    select_pose(0);  // will update rigid transformation for bones and Model matrix
}

Human Human::load_human_npz(const std::string& path) {
    Human human;
    auto npz = cnpy::npz_load(path);

    // Vector will resize to have enough space for the objects. It will then iterate through the objects and call the default copy operator for every object. In this way, the copy of the vector is 'deep'.

    // This move() gives its target the value of its argument, but is not obliged to preserve the value of its source. So, for a vector, move() could reasonably be expected to leave its argument as a zero-capacity vector to avoid having to copy all the elements. In other words, move is a potentially destructive read.

    /** 
     *  foo(foo&& other)
        {
            this->length = other.length;
            this->ptr = other.ptr;
            other.length = 0;
            other.ptr = nullptr;
        }
     */

    // Basics: vertices, faces, normals, this will discard the initialized memory of triangles and vertices
    // FIXME: estimating normal instead of loading
    human.faces = npz["triangles"].as_vec<int>();    // load faces (int32)
    auto verts = npz["vertices"].as_vec<float>();    // load vertices
    auto normal = std::vector<float>(verts.size());  // allocate mem for normal
    estimate_normals(verts, normal, human.faces);    // estimate vertex normal from face normal

    // VAO data (VBO)
    auto blend_weights = npz["weights"].as_vec<float>();  // load blend weights
    auto bones = npz["bones"].as_vec<int>();              // load bone indices
    // resize the verts array to hold everything
    auto n_verts = verts.size() / VERT_SZ;
    human.verts.resize(n_verts * DATA_SZ);
    // fill the verts array
    // auto iter = human.verts.begin() + n_verts * VERT_SZ;
    // std::move(normal.begin(), normal.end(), iter);
    // iter += n_verts * NORM_SZ;
    // std::move(blend_weights.begin(), blend_weights.end(), iter);
    // iter += n_verts * BWEI_SZ;
    // std::move(bones.begin(), bones.end(), iter);
    // FIXME: CAN WE NOT REARRANGE THE MEMORY?
    for (int i = 0; i < n_verts; i++) {
        // fill vertices
        for (int j = 0; j < VERT_SZ; ++j) {
            human.verts[0 + i * VERT_SZ + j] = verts[i * VERT_SZ + j];
        }
        for (int j = 0; j < NORM_SZ; ++j) {
            human.verts[(human.n_verts() * (VERT_SZ + DEFO_SZ)) + i * NORM_SZ + j] = normal[i * NORM_SZ + j];
        }
        for (int j = 0; j < BWEI_SZ; ++j) {
            human.verts[(human.n_verts() * (VERT_SZ + DEFO_SZ + NORM_SZ)) + i * BWEI_SZ + j] = blend_weights[i * BWEI_SZ + j];
        }
        for (int j = 0; j < BONE_SZ; ++j) {
            human.verts[(human.n_verts() * (VERT_SZ + DEFO_SZ + NORM_SZ + BWEI_SZ)) + i * BONE_SZ + j] = (float)bones[i * BONE_SZ + j];  // FIXME: int to float
        }
    }

    // Joints / Bones
    human.parents = npz["parents"].as_vec<int>();
    auto tjoints = npz["tjoints"].data<glm::vec3>();
    human.tjoints = std::vector<glm::vec3>(tjoints, tjoints + POSE_BONE_SZ);

    // will update the bigpose components once the human is loaded
    human.update_bigrigid();

    return human;
}

}  // namespace volrend
