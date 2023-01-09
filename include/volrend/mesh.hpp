#pragma once

#include <array>
#include <string>
#include <vector>

#include "glm/mat4x4.hpp"
#include "volrend/happly.hpp"
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
        std::vector<std::array<double, 3>> trimesh_vertices;
        std::vector<std::vector<unsigned int>> trimesh_faces;
        for (int i = 0; i < n_verts(); i++) {
            trimesh_vertices.push_back({verts[i * VERT_SZ], verts[i * VERT_SZ + 1], verts[i * VERT_SZ + 2]});
        }
        for (int i = 0; i < n_faces(); i++) {
            trimesh_faces.push_back({faces[i * 3], faces[i * 3 + 1], faces[i * 3 + 2]});
        }

        // Create an empty object
        happly::PLYData plyOut;

        // Add mesh data (elements are created automatically)
        plyOut.addVertexPositions(trimesh_vertices);
        plyOut.addFaceIndices(trimesh_faces);

        // Write the object to file
        plyOut.write(path, happly::DataFormat::Binary);
    }

   private:
    unsigned int vao_, vbo_, ebo_;
};

}  // namespace volrend
