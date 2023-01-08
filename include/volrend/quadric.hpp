#pragma once

#include <glm/gtc/type_ptr.hpp>
#include <string>
#include <vector>

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

    glm::mat4 Q() const {
        return glm::mat4(
            A, B, C, D,
            B, E, F, G,
            C, F, H, I,
            D, G, I, J);
    }

    std::vector<Mesh> meshes;
};

}  // namespace volrend
