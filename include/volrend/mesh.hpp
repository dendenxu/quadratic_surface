#pragma once

#include <array>
#include <string>
#include <vector>

#include "glm/mat4x4.hpp"
#define MSH_PLY_IMPLEMENTATION
#include "volrend/msh_ply.hpp"

void estimate_normals(std::vector<float>& verts, const std::vector<unsigned int>& faces);

namespace volrend {
const int VERT_SZ = 9;

struct Mesh {
    explicit Mesh(int n_verts = 0, int n_faces = 0, int face_size = 3, bool unlit = true);

    // Upload to GPU
    void update();

    // Draw the mesh
    void draw(const glm::mat4x4& V, glm::mat4x4 K) const;

    // Create faces by grouping consecutive vertices (only for triangle mesh)
    void auto_faces();

    // Copy the vertices & faces in the mesh n times and offset the faces
    // appropriately
    void repeat(int n);

    // Apply affine c2w directly to the vertices (rotation is axis-angle)
    void apply_transform(glm::vec3 r, glm::vec3 t, int start = 0, int end = -1);
    void apply_transform(glm::mat4 c2w, int start = 0, int end = -1);

    // Vertex positions
    std::vector<float> verts;
    // Triangle indices
    std::vector<unsigned int> faces;

    size_t n_verts() { return verts.size() / VERT_SZ; }
    size_t n_faces() { return faces.size() / face_size; }

    // Model c2w, rotation is axis-angle
    glm::vec3 rotation{0.0};
    glm::vec3 translation{0.0};
    float scale = 1.f;

    // Computed c2w
    mutable glm::mat4 transform_;

    int face_size = 3;
    bool visible = true;
    bool unlit = true;

    std::string name = "Mesh";

    // * Preset meshes
    // Unit cube centered at 0
    static Mesh Cube(glm::vec3 color = glm::vec3(1.0f, 0.5f, 0.2f));

    // Unit UV sphere centered at 0
    static Mesh Sphere(int rings = 15, int sectors = 30,
                       glm::vec3 color = glm::vec3(1.0f, 0.5f, 0.2f));

    // Point lattice
    static Mesh Lattice(int reso = 8,
                        glm::vec3 color = glm::vec3(0.5f, 0.5f, 0.5f));

    // A single camera frustum
    static Mesh CameraFrustum(float focal_length, float image_width,
                              float image_height, float z = -0.3,
                              glm::vec3 color = glm::vec3(0.5f, 0.5f, 0.5f));

    // A single line from a to b
    static Mesh Line(glm::vec3 a, glm::vec3 b,
                     glm::vec3 color = glm::vec3(0.5f, 0.5f, 0.5f));

    // Consecutive lines (points: N * 3), each consecutive 3 numbers is a point
    static Mesh Lines(std::vector<float> points,
                      glm::vec3 color = glm::vec3(0.5f, 0.5f, 0.5f));

    // Point cloud (points: N * 3), each consecutive 3 numbers is a point
    static Mesh Points(std::vector<float> points,
                       glm::vec3 color = glm::vec3(0.5f, 0.5f, 0.5f));

    // Load a basic OBJ file (triangles & optionally vertex colors)
    static Mesh load_basic_obj(const std::string& path);
    static Mesh load_mem_basic_obj(const std::string& str);

    // Load series of meshes/lines/points from a npz file
    static std::vector<Mesh> open_drawlist(const std::string& path,
                                           bool default_visible = true);
    static std::vector<Mesh> open_drawlist_mem(const char* data, uint64_t size,
                                               bool default_visible = true);

    typedef struct TriMesh {
        std::array<float, 3>* vertices;
        std::array<unsigned int, 3>* faces;
        int32_t n_vertices;
        int32_t n_faces;
    } TriMesh_t;

    void save_ply(const std::string& path) {
        std::vector<std::array<float, 3>> trimesh_vertices;
        std::vector<std::array<unsigned int, 3>> trimesh_faces;
        for (int i = 0; i < n_verts(); i++) {
            trimesh_vertices.push_back({verts[i * VERT_SZ], verts[i * VERT_SZ + 1], verts[i * VERT_SZ + 2]});
        }
        for (int i = 0; i < n_faces(); i++) {
            trimesh_faces.push_back({faces[i * 3], faces[i * 3 + 1], faces[i * 3 + 2]});
        }

        TriMesh_t trimesh{0};
        trimesh.n_faces = n_faces();
        trimesh.n_vertices = n_verts();
        trimesh.vertices = trimesh_vertices.data();
        trimesh.faces = trimesh_faces.data();

        msh_ply_desc_t descriptors[2];
        const char* vertex_name[]{"x", "y", "z"};
        descriptors[0].element_name = "vertex";
        descriptors[0].property_names = vertex_name;
        descriptors[0].num_properties = 3;
        descriptors[0].data_type = MSH_PLY_FLOAT;
        descriptors[0].data = &trimesh.vertices;
        descriptors[0].data_count = &trimesh.n_vertices;

        const char* face_name[]{"vertex_indices"};
        descriptors[1].element_name = "face";
        descriptors[1].property_names = face_name;
        descriptors[1].num_properties = 1;
        descriptors[1].data_type = MSH_PLY_INT32;
        descriptors[1].list_type = MSH_PLY_UINT8;
        descriptors[1].data = &trimesh.faces;
        descriptors[1].data_count = &trimesh.n_faces;
        descriptors[1].list_size_hint = 3;

        // Create new ply file
        msh_ply_t* ply_file = msh_ply_open(path.c_str(), "wb");
        // Add descriptors to ply file
        msh_ply_add_descriptor(ply_file, &descriptors[0]);
        msh_ply_add_descriptor(ply_file, &descriptors[1]);

        // Write data to disk
        msh_ply_write(ply_file);

        // Close ply file
        msh_ply_close(ply_file);
    }

   private:
    unsigned int vao_, vbo_, ebo_;
};

}  // namespace volrend
