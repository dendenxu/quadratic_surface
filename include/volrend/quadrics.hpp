#pragma once
#define _CRT_SECURE_NO_WARNINGS  // ignore sprintf unsafe warnings on windows

#include <glm/gtc/type_ptr.hpp>
#include <string>
#include <vector>

#include "tinylogger.hpp"
#include "volrend/common.hpp"
#include "volrend/internal/shader.hpp"

namespace volrend {

// Quadrics primitive container
struct Quadrics {
    // Upload to GPU
    void update();

    // Draw the mesh
    void draw(const glm::mat4x4& V, glm::mat4x4 K) const;

    // Quadrics parameters
    std::vector<float[10]> params;  // N * 10 -> will be converted to N, 10 textures

    // Q matrices

    // OpenGL does not support varying length data, maybe we need to define a max length?
};

}  // namespace volrend