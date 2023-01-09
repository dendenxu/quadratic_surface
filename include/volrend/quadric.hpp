#pragma once

#include <glm/gtc/type_ptr.hpp>
#include <string>
#include <vector>

#include "volrend/marching_cubes.hpp"
#include "volrend/mesh.hpp"

namespace volrend {

// const mat4 hyperbolicParaboloid = mat4(
//     4.0, 0.0, 0.0, 0.0,
//     0.0, -4.0, 0.0, 0.0,
//     0.0, 0.0, 0.0, 1.0,
//     0.0, 0.0, 1.0, 0.0
// );

// \mathbf{Q} =
// \begin{bmatrix}
// A & B & C & D \\
// B & E & F & G \\
// C & F & H & I \\
// D & G & I & J
// \end{bmatrix}

struct Quadric {
    float A = 4.0f;
    float B = 0.0f;
    float C = 0.0f;
    float D = 0.0f;
    float E = -4.0f;
    float F = 0.0f;
    float G = 0.0f;
    float H = 0.0f;
    float I = 1.0f;
    float J = 0.0f;

    float eps = 0.000001f;
    float box_size = 0.5f;
    int samples = 4;
    int resolution = 512;  // marching cubes resolution

    glm::mat4 Q() const {
        return glm::mat4(
            A, B, C, D,
            B, E, F, G,
            C, F, H, I,
            D, G, I, J);
    }

    float evaluate(float x, float y, float z) const {
        // evaluate the quadratic function at the given point
        return A * x * x + 2 * B * x * y + 2 * C * x * z + 2 * D * x + E * y * y + 2 * F * y * z + G * y + H * z * z + 2 * I * z + J;
        // return glm::dot(glm::vec4(x, y, z, 1.0f), Q() * glm::vec4(x, y, z, 1.0f));
    }

    bool render_mesh = true;     // if render_mesh and loaded, render the triangle mesh instead
    std::unique_ptr<Mesh> mesh;  // once loaded, render this

    void marching_cubes() {
        // mesh = std::make_unique<Mesh>(Mesh::Sphere());
        // mesh->update();
        // return;

        // perform marching cubes on the quadratic surface
        mesh = std::make_unique<Mesh>();
        mesh->name = "Quadric Mesh";

        std::vector<double> verts;  // marching cubes requires double input
        std::vector<size_t> faces;  // marching cubes requires organized input
        std::vector<double> lower = {-box_size, -box_size, -box_size};
        std::vector<double> upper = {box_size, box_size, box_size};
        mc::marching_cubes(
            lower,
            upper,
            resolution, resolution, resolution,
            [this](float x, float y, float z) {
                return evaluate(x, y, z);
            },
            0.0,
            verts,
            faces);
        mesh->faces = std::vector<unsigned int>(faces.begin(), faces.end());
        mesh->verts.resize(verts.size() / 3 * VERT_SZ);
        for (int i = 0; i < verts.size() / 3; i++) {
            mesh->verts[i * VERT_SZ + 0] = verts[i * 3 + 0];
            mesh->verts[i * VERT_SZ + 1] = verts[i * 3 + 1];
            mesh->verts[i * VERT_SZ + 2] = verts[i * 3 + 2];
        }
        estimate_normals(mesh->verts, mesh->faces);
        mesh->update();
    }
};

}  // namespace volrend
