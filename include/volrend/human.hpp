#pragma once

#include <string>
#include <vector>

#ifdef __EMSCRIPTEN__
// WebGL
#include <GLES3/gl3.h>
#else
#include <GL/glew.h>
#endif

#include "glm/mat4x4.hpp"
#include "volrend/render_options.hpp"

const unsigned int POSE_BONE_SZ = 24;
const unsigned int VERT_SZ = 3;  // position[3], deformation[3], normal[3], bw[4], bone[4]
const unsigned int DEFO_SZ = 3;
const unsigned int NORM_SZ = 3;
const unsigned int BWEI_SZ = 4;
const unsigned int BONE_SZ = 4;
const unsigned int DATA_SZ = VERT_SZ + DEFO_SZ + NORM_SZ + BWEI_SZ + BONE_SZ;

const unsigned int FACE_SZ = 3;

const unsigned int ANGLE_AXIS_ROT_SZ = 3;  // angle axis rotation
const unsigned int JOINT_LOC_SZ = 3;       // similar to VERT_SZ, 3D location in world

namespace volrend {
struct _HumanUniforms {
    GLint
        K,
        MV,
        V,
        M,
        cam_pos,
        unlit,
        rigid,
        bigrigid,
        bone_association,
        use_face_normal,
        bigpose_geometry;
};
struct Human {
    Human(int n_verts = 0, int n_faces = 0);
    inline size_t Human::n_verts() { return verts.size() / DATA_SZ; }
    inline size_t Human::n_poses() {
        if (n_verts()) {
            return std::max(all_poses.size() / POSE_BONE_SZ, blend_shapes.size() / (n_verts() * VERT_SZ));
        } else {
            return 0;  // avoid division by zero exception
        }
    }

    // Upload to GPU
    void update();

    // Use human shader program (for shader render purposes only)
    void use_shader();

    // Draw the human
    void draw(const glm::mat4x4 &V, glm::mat4x4 K, const RenderOptions &options) const;

    // Create faces by grouping consecutive vertices (only for triangle human)
    void auto_faces();

    // Copy the vertices & faces in the human n times and offset the faces
    // appropriately
    void repeat(int n);

    // void mutable_select_pose(int *index);
    void load_poses_npz(const std::string &path);
    void select_pose(int index);   // load pose and c2w into transforma matrix, with blend shapes
    void update_pose(int index);   // load pose and c2w into transforma matrix
    void update_verts(int index);  // load blend shapes into specific

    void update_bigrigid();
    std::vector<glm::mat4> get_bigrigid();

    // Vertex positions
    std::vector<float> verts;
    // Triangle indices
    std::vector<int> faces;

    // Bones
    std::vector<int> parents;        // which bone is this one attached to
    std::vector<glm::vec3> tjoints;  // where is this bone

    // FIXME: constraint on number of poses and rs/ts
    std::vector<glm::vec3> all_poses;  // ! need to create a 2d view of this
    std::vector<glm::vec3> all_rs;     // pose specific c2w
    std::vector<glm::vec3> all_ts;     // pose specific c2w
    std::vector<float> blend_shapes;   // ! need to create a 2d view of this

    // // Model c2w, rotation is axis-angle
    // glm::vec3 rotation, translation;
    // float scale = 1.f;

    // Computed c2w and rigid c2w for bones
    // TODO: These should be initialized
    // Note: human will be created upon launching of the program
    // all data except "rigid" is 0
    // and "rigid" is 24 matrix of all zeros
    // Except that, they really should be initailized to eye matrices
    glm::mat4 c2w = glm::mat4(1.0f);
    std::vector<glm::mat4> rigid = std::vector<glm::mat4>(24, glm::mat4(1.0f));

    std::string name = "Human";

    // Load a basic OBJ file (triangles & optionally vertex colors)
    // static Human load_basic_obj(const std::string &path);
    // static Human load_mem_basic_obj(const std::string &str);
    static Human load_human_npz(const std::string &path);

    unsigned int vao_, vbo_, ebo_;

   private:
    void update_rigid(int index);
    void update_transform(int index);
    // bigpose definition
    // https://stackoverflow.com/questions/47735318/compiler-error-c2280-attempting-to-reference-a-deleted-function-operator

    /** bigposes definition
     * array([[ 0.       ,  0.       ,  0.       ],
        [ 0.       ,  0.       ,  0.5235988],
        [ 0.       ,  0.       , -0.5235988],
        [ 0.       ,  0.       ,  0.       ],
        [ 0.       ,  0.       ,  0.       ],
        [ 0.       ,  0.       ,  0.       ],
        [ 0.       ,  0.       ,  0.       ],
        [ 0.       ,  0.       ,  0.       ],
        [ 0.       ,  0.       ,  0.       ],
        [ 0.       ,  0.       ,  0.       ],
        [ 0.       ,  0.       ,  0.       ],
        [ 0.       ,  0.       ,  0.       ],
        [ 0.       ,  0.       ,  0.       ],
        [ 0.       ,  0.       ,  0.       ],
        [ 0.       ,  0.       ,  0.       ],
        [ 0.       ,  0.       ,  0.       ],
        [ 0.       ,  0.       ,  0.       ],
        [ 0.       ,  0.       ,  0.       ],
        [ 0.       ,  0.       ,  0.       ],
        [ 0.       ,  0.       ,  0.       ],
        [ 0.       ,  0.       ,  0.       ],
        [ 0.       ,  0.       ,  0.       ],
        [ 0.       ,  0.       ,  0.       ],
        [ 0.       ,  0.       ,  0.       ]], dtype=float32)
    * we need to first c2w points to tpose and then to bigpose to get the correct sampling location (octree lives in bigpose space)
    */
    std::vector<glm::vec3> bigposes = std::vector<glm::vec3>(24, glm::vec3(0.0f));
    std::vector<glm::mat4> bigrigid = std::vector<glm::mat4>(24, glm::mat4(1.0f));

    _HumanUniforms u;  // uniform value in the shader
    unsigned int program = -1;
};
}  // namespace volrend
