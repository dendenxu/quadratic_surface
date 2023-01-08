#pragma once

#include <array>
#include <memory>

#include "glm/mat4x3.hpp"
#include "glm/mat4x4.hpp"
#include "glm/vec2.hpp"
#include "glm/vec3.hpp"
#include "volrend/common.hpp"

namespace volrend {
static const float CAMERA_DEFAULT_FOCAL_LENGTH = 1111.11f;

struct Camera {
    Camera(int width = 960, int height = 540,
           float fx = CAMERA_DEFAULT_FOCAL_LENGTH, float fy = -1.f);
    ~Camera();

    /** Drag helpers **/
    void begin_drag(float x, float y, bool is_pan, bool about_origin);
    void drag_update(float x, float y);
    void end_drag();
    bool is_dragging() const;
    /** Move center by +=xyz, correctly handling drag **/
    void move(const glm::vec3& xyz);

    /** Camera params **/
    // Camera pose model, you can modify these
    glm::vec3 v_back, v_world_up, center;

    // Origin for about-origin rotation
    glm::vec3 origin;

    // Vectors below are automatically updated
    glm::vec3 v_up, v_right;

    // 4x3 C2W transform used for volume rendering, automatically updated
    glm::mat4x3 c2w;
    bool lock_trans = false;  // update from vector?

    // 4x4 projection matrix for triangle rendering
    glm::mat4x4 K;
    glm::mat4x4 inv_K;

    // 4x4 W2C transform
    glm::mat4x4 w2c;

    // Image size
    int width, height;

    // Focal length
    float fx, fy;

    // Principal point
    float cx, cy;

    // camera space clip near & far
    float clip_near = 1e-3f;

    // GUI movement speed
    float movement_speed = 1.f;

    // CUDA memory used in kernel
    struct {
        float* transform = nullptr;
    } device;

    // Update the transform after modifying v_right/v_forward/center
    // (internal)
    void _update(bool transform_from_vecs = true, bool copy_cuda = true);

   private:
    // For dragging
    struct DragState;
    std::unique_ptr<DragState> drag_state_;
};

}  // namespace volrend
