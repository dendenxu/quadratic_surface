#pragma once

#include <memory>

#include "volrend/camera.hpp"
#include "volrend/mesh.hpp"
#include "volrend/quadric.hpp"

namespace volrend {
// Volume renderer using CUDA or compute shader
struct VolumeRenderer {
    explicit VolumeRenderer();
    ~VolumeRenderer();

    // Render the currently set tree
    void render();

    // Resize the buffer
    void resize(int width, int height);

    // Get name identifying the renderer backend used e.g. CUDA
    const char* get_backend();

    // Camera instance
    Camera camera;

    // Quadric instance
    Quadric quadric;

   private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace volrend
