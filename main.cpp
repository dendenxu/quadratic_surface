#define _CRT_SECURE_NO_WARNINGS  // ignore sprintf unsafe warnings on windows
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <string>

#include "volrend/internal/imwrite.hpp"
#include "volrend/internal/opts.hpp"
#include "volrend/n3tree.hpp"
#include "volrend/renderer.hpp"

// clang-format off
// You know what? Just DO NOT CHANGE THIS INCLUDE ORDER
#include "imgui_impl_opengl3.h"
#include "imgui_impl_glfw.h"

#include "ImGuizmo.h"
// clang-format on

#ifndef __EMSCRIPTEN__
#include "imfilebrowser.h"
#endif

#ifdef VOLREND_CUDA
#include "volrend/cuda/common.cuh"
#endif

namespace volrend {

namespace {

#define GET_RENDERER(window) \
    (*((VolumeRenderer*)glfwGetWindowUserPointer(window)))

void glfw_update_title(GLFWwindow* window) {
    // static fps counters
    // Source: http://antongerdelan.net/opengl/glcontext2.html
    static const int moving_averge_cnt = 5;
    static int frame_count = 0;
    static double stamp_prev = 0.0;
    static int head = 0;
    static double moving_average[moving_averge_cnt] = {
        0,
    };

    const double stamp_curr = glfwGetTime();
    const double elapsed = stamp_curr - stamp_prev;
    const double fps = (double)frame_count / elapsed;

    // assign new value
    moving_average[head] = fps;

    double average = 0.0f;
    char tmp[128];

    for (int i = 0; i < moving_averge_cnt; i++) {
        average += moving_average[i];
    }
    average /= moving_averge_cnt;

    if (elapsed > 0.5) {
        stamp_prev = stamp_curr;

        sprintf(tmp, "dynamic real-time nerf - FPS: %.2f", average);
        glfwSetWindowTitle(window, tmp);
        frame_count = 0;
    }

    frame_count++;
    head = (head + 1) % moving_averge_cnt;
}

int gizmo_mesh_op = ImGuizmo::TRANSLATE;
int gizmo_mesh_space = ImGuizmo::LOCAL;
bool play_animation = false;
float fps = 60.0;
int selected_pose = 0;

void draw_imgui(VolumeRenderer& rend
                // N3Tree& tree,
                // Human& human
) {
    auto& cam = rend.camera;
    auto& quadric = rend.quadric;
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    ImGui::SetNextWindowPos(ImVec2(20.f, 20.f), ImGuiCond_Once);
    ImGui::SetNextWindowSize(ImVec2(400.f, 800.f), ImGuiCond_Once);

    static char title[128] = {0};
    if (title[0] == 0) {
        sprintf(title, "volrend backend: %s", rend.get_backend());
    }

    // Begin window
    ImGui::Begin(title);
    static ImGui::FileBrowser save_screenshot_dialog(ImGuiFileBrowserFlags_EnterNewFilename);
    if (save_screenshot_dialog.GetTitle().empty()) {
        save_screenshot_dialog.SetTypeFilters({".png"});
        save_screenshot_dialog.SetTitle("Save screenshot (png)");
    }

    save_screenshot_dialog.Display();
    if (save_screenshot_dialog.HasSelected()) {
        // Save screenshot
        std::string path = save_screenshot_dialog.GetSelected().string();
        save_screenshot_dialog.ClearSelected();
        int width = rend.camera.width, height = rend.camera.height;
        std::vector<unsigned char> windowPixels(4 * width * height);
        glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE,
                     &windowPixels[0]);

        std::vector<unsigned char> flippedPixels(4 * width * height);
        for (int row = 0; row < height; ++row)
            memcpy(&flippedPixels[row * width * 4],
                   &windowPixels[(height - row - 1) * width * 4], 4 * width);

        if (path.size() < 4 ||
            path.compare(path.size() - 4, 4, ".png", 0, 4) != 0) {
            path.append(".png");
        }
        if (internal::write_png_file(path, flippedPixels.data(), width,
                                     height)) {
            printf("Wrote %s", path.c_str());
        } else {
            printf("Failed to save screenshot\n");
        }
    }

    ImGui::SetNextTreeNodeOpen(true, ImGuiCond_Once);
    if (ImGui::CollapsingHeader("Camera")) {
        // Update vectors indirectly since we need to normalize on change
        // (press update button) and it would be too confusing to keep
        // normalizing
        static glm::vec3 world_up_tmp = rend.camera.v_world_up;
        static glm::vec3 world_down_prev = rend.camera.v_world_up;
        static glm::vec3 back_tmp = rend.camera.v_back;
        static glm::vec3 forward_prev = rend.camera.v_back;
        if (cam.v_world_up != world_down_prev)
            world_up_tmp = world_down_prev = cam.v_world_up;
        if (cam.v_back != forward_prev) back_tmp = forward_prev = cam.v_back;

        ImGui::InputFloat3("center", glm::value_ptr(cam.center));
        ImGui::InputFloat3("origin", glm::value_ptr(cam.origin));
        ImGui::SliderFloat("clip_near", &cam.clip_near, 0.0001f, 10.f);

        static bool lock_fx_fy = true;
        static float fx = cam.fx;  // gui state variable for duplication
        static float fy = cam.fy;  // gui state variable for duplication (like one slider + one input for same location)
        ImGui::Checkbox("fx=fy", &lock_fx_fy);
        if (lock_fx_fy) {
            if (ImGui::SliderFloat("focal", &cam.fx, 10.f, 50000.f)) {
                cam.fy = cam.fx;
            }
        } else {
            ImGui::SliderFloat("fx", &cam.fx, 10.f, 50000.f);
            ImGui::SliderFloat("fy", &cam.fy, 10.f, 50000.f);
            ImGui::InputFloat("fx", &cam.fx);
            ImGui::InputFloat("fy", &cam.fy);
        }

        ImGui::SetNextTreeNodeOpen(true, ImGuiCond_Once);
        if (ImGui::TreeNode("Directions")) {
            ImGui::InputFloat3("world_up", glm::value_ptr(world_up_tmp));
            ImGui::InputFloat3("back", glm::value_ptr(back_tmp));
            if (ImGui::Button("normalize & update dirs")) {
                cam.v_world_up = glm::normalize(world_up_tmp);
                cam.v_back = glm::normalize(back_tmp);
            }
            ImGui::TreePop();
        }
        ImGui::SetNextTreeNodeOpen(true, ImGuiCond_Once);
        if (ImGui::TreeNode("Transform")) {
            ImGui::Checkbox("Lock Transform", &cam.lock_trans);
            ImGui::InputFloat3("x", &cam.c2w[0][0]);
            ImGui::InputFloat3("y", &cam.c2w[1][0]);
            ImGui::InputFloat3("z", &cam.c2w[2][0]);
            ImGui::InputFloat3("t", &cam.c2w[3][0]);
            ImGui::TreePop();
        }
    }  // End camera node

    ImGui::SetNextTreeNodeOpen(true, ImGuiCond_Once);
    if (ImGui::CollapsingHeader("Quadric")) {
        ImGui::SetNextTreeNodeOpen(true, ImGuiCond_Once);
        if (ImGui::TreeNode("Parameters")) {
            ImGui::SliderFloat("A", &quadric.A, -10.0f, 10.f);
            ImGui::SliderFloat("B", &quadric.B, -10.0f, 10.f);
            ImGui::SliderFloat("C", &quadric.C, -10.0f, 10.f);
            ImGui::SliderFloat("D", &quadric.D, -10.0f, 10.f);
            ImGui::SliderFloat("E", &quadric.E, -10.0f, 10.f);
            ImGui::SliderFloat("F", &quadric.F, -10.0f, 10.f);
            ImGui::SliderFloat("G", &quadric.G, -10.0f, 10.f);
            ImGui::SliderFloat("H", &quadric.H, -10.0f, 10.f);
            ImGui::SliderFloat("I", &quadric.I, -10.0f, 10.f);
            ImGui::SliderFloat("J", &quadric.J, -10.0f, 10.f);
            ImGui::TreePop();
        }
        ImGui::SetNextTreeNodeOpen(true, ImGuiCond_Once);
        if (ImGui::TreeNode("Rendering")) {
            static float inv_eps = 1 / quadric.eps;
            ImGui::SliderFloat("box_size", &quadric.box_size, 0.001f, 10.f);
            ImGui::SliderInt("samples ^ 1/2", &quadric.samples, 1, 10);  // super sampling ratio
            if (ImGui::SliderFloat("1 / epsilon", &inv_eps, 0.0f, 10000000.f)) {
                quadric.eps = 1 / inv_eps;
            };
            ImGui::TreePop();
        }
    }

    ImGui::End();
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void glfw_error_callback(int error, const char* description) {
    fputs(description, stderr);
}

void glfw_key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);
    if (ImGui::GetIO().WantCaptureKeyboard) return;

    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
        auto& rend = GET_RENDERER(window);
        auto& cam = rend.camera;
        switch (key) {
        case GLFW_KEY_ESCAPE:
            glfwSetWindowShouldClose(window, GL_TRUE);
            break;
        case GLFW_KEY_SPACE:
            play_animation = !play_animation;
        case GLFW_KEY_W:
        case GLFW_KEY_S:
        case GLFW_KEY_A:
        case GLFW_KEY_D:
        case GLFW_KEY_E:
        case GLFW_KEY_Q: {
            // Camera movement
            float speed = 0.1f;
            if (mods & GLFW_MOD_SHIFT) speed *= 5.f;
            if (key == GLFW_KEY_S || key == GLFW_KEY_A || key == GLFW_KEY_E)
                speed = -speed;
            const auto& vec =
                (key == GLFW_KEY_A || key == GLFW_KEY_D)   ? cam.v_right
                : (key == GLFW_KEY_W || key == GLFW_KEY_S) ? -cam.v_back
                                                           : -cam.v_up;
            cam.move(vec * speed);
        } break;

        case GLFW_KEY_C: {
            // Print C2W matrix
            puts("C2W:\n");
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 4; ++j) {
                    if (j) puts(" ");
                    printf("%.10f", cam.c2w[j][i]);
                }
                puts("\n");
            }
            fflush(stdout);
        } break;

        case GLFW_KEY_Z: {
            // Cycle gizmo op
            if (gizmo_mesh_op == ImGuizmo::TRANSLATE)
                gizmo_mesh_op = ImGuizmo::ROTATE;
            else if (gizmo_mesh_op == ImGuizmo::ROTATE)
                gizmo_mesh_op = ImGuizmo::SCALE_Z;
            else
                gizmo_mesh_op = ImGuizmo::TRANSLATE;
        } break;

        case GLFW_KEY_X: {
            // Cycle gizmo space
            if (gizmo_mesh_space == ImGuizmo::LOCAL)
                gizmo_mesh_space = ImGuizmo::WORLD;
            else
                gizmo_mesh_space = ImGuizmo::LOCAL;
        } break;

        case GLFW_KEY_MINUS:
            cam.fx *= 0.99f;
            cam.fy *= 0.99f;
            break;

        case GLFW_KEY_EQUAL:
            cam.fx *= 1.01f;
            cam.fy *= 1.01f;
            break;

        case GLFW_KEY_0:
            cam.fx = CAMERA_DEFAULT_FOCAL_LENGTH;
            cam.fy = CAMERA_DEFAULT_FOCAL_LENGTH;
            break;

        case GLFW_KEY_1:
            cam.v_world_up = glm::vec3(0.f, 0.f, 1.f);
            break;

        case GLFW_KEY_2:
            cam.v_world_up = glm::vec3(0.f, 0.f, -1.f);
            break;

        case GLFW_KEY_3:
            cam.v_world_up = glm::vec3(0.f, 1.f, 0.f);
            break;

        case GLFW_KEY_4:
            cam.v_world_up = glm::vec3(0.f, -1.f, 0.f);
            break;

        case GLFW_KEY_5:
            cam.v_world_up = glm::vec3(1.f, 0.f, 0.f);
            break;

        case GLFW_KEY_6:
            cam.v_world_up = glm::vec3(-1.f, 0.f, 0.f);
            break;
        }
    }
}

void glfw_mouse_button_callback(GLFWwindow* window, int button, int action,
                                int mods) {
    ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);
    if (ImGui::GetIO().WantCaptureMouse) return;

    auto& rend = GET_RENDERER(window);
    auto& cam = rend.camera;
    double x, y;
    glfwGetCursorPos(window, &x, &y);
    if (action == GLFW_PRESS) {
        const bool SHIFT = mods & GLFW_MOD_SHIFT;
        cam.begin_drag((float)x, (float)y,
                       SHIFT || button == GLFW_MOUSE_BUTTON_MIDDLE,
                       button == GLFW_MOUSE_BUTTON_LEFT || (button == GLFW_MOUSE_BUTTON_MIDDLE && SHIFT));
    } else if (action == GLFW_RELEASE) {
        cam.end_drag();
    }
}

void glfw_cursor_pos_callback(GLFWwindow* window, double x, double y) {
    GET_RENDERER(window).camera.drag_update((float)x, (float)y);
}

void glfw_scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    ImGui_ImplGlfw_ScrollCallback(window, xoffset, yoffset);
    if (ImGui::GetIO().WantCaptureMouse) return;
    auto& cam = GET_RENDERER(window).camera;
    // Focal length adjusting was very annoying so changed it to movement in z
    // cam.focal *= (yoffset > 0.f) ? 1.01f : 0.99f;
    const float speed_fact = 1e-1f;
    cam.move(cam.v_back * ((yoffset < 0.f) ? speed_fact : -speed_fact));
}

GLFWwindow* glfw_init(const int width, const int height) {
    glfwSetErrorCallback(glfw_error_callback);

    if (!glfwInit()) std::exit(EXIT_FAILURE);

    glfwWindowHint(GLFW_DEPTH_BITS, GL_TRUE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    GLFWwindow* window =
        glfwCreateWindow(width, height, "volrend viewer", NULL, NULL);

    glClearDepth(1.0);
    glDepthFunc(GL_LESS);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    if (window == nullptr) {
        glfwTerminate();
        std::exit(EXIT_FAILURE);
    }

    glfwMakeContextCurrent(window);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        fputs("GLEW init failed\n", stderr);
        getchar();
        glfwTerminate();
        std::exit(EXIT_FAILURE);
    }

    // ignore vsync for now
    // in some G-Sync enabled systems, you may get a consistent 160 fps
    // when actually it should have been a few thousands
    glfwSwapInterval(0);

    // only copy r/g/b
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_FALSE);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGui_ImplGlfw_InitForOpenGL(window, false);
    char* glsl_version = NULL;
    ImGui_ImplOpenGL3_Init(glsl_version);
    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    ImGui::GetIO().IniFilename = nullptr;
    glfwSetCharCallback(window, ImGui_ImplGlfw_CharCallback);

    return window;
}

void glfw_window_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
    GET_RENDERER(window).resize(width, height);
}

}  // namespace
}  // namespace volrend

int main(int argc, char* argv[]) {
    using namespace volrend;

    cxxopts::Options cxxoptions(
        "volrend",
        "OpenGL Dynamic PlenOctree volume rendering (c)");

    internal::add_common_opts(cxxoptions);
    // clang-format off
    cxxoptions.add_options()
        ("nogui", "disable imgui", cxxopts::value<bool>())
        ("center", "camera center position (world); ignored for NDC",
                cxxopts::value<std::vector<float>>()->default_value("0,0,-10.0"))
        ("back", "camera's back direction unit vector (world) for orientation; ignored for NDC",
                cxxopts::value<std::vector<float>>()->default_value("0,0,1"))
        ("origin", "origin for right click rotation controls; ignored for NDC",
                cxxopts::value<std::vector<float>>()->default_value("0,0,0"))
        ("world_up", "world up direction for rotating controls e.g. "
                     "0,0,1=blender; ignored for NDC",
                cxxopts::value<std::vector<float>>()->default_value("0,1,0"))
        ("grid", "show grid with given max resolution (4 is reasonable)", cxxopts::value<int>())
        ("probe", "enable lumisphere_probe and place it at given x,y,z",
                   cxxopts::value<std::vector<float>>())
        ;
    // clang-format on

    cxxopts::ParseResult args = internal::parse_options(cxxoptions, argc, argv);
    int width = args["width"].as<int>(), height = args["height"].as<int>();
    float fx = args["fx"].as<float>();
    float fy = args["fy"].as<float>();
    bool nogui = args["nogui"].as<bool>();

    GLFWwindow* window = glfw_init(width, height);
    {
        VolumeRenderer rend;
        if (fx > 0.f) {
            rend.camera.fx = fx;
        }

        auto cen = args["center"].as<std::vector<float>>();
        rend.camera.center = glm::vec3(cen[0], cen[1], cen[2]);
        auto origin = args["origin"].as<std::vector<float>>();
        rend.camera.origin = glm::vec3(origin[0], origin[1], origin[2]);
        auto world_up = args["world_up"].as<std::vector<float>>();
        rend.camera.v_world_up = glm::vec3(world_up[0], world_up[1], world_up[2]);
        auto back = args["back"].as<std::vector<float>>();
        rend.camera.v_back = glm::vec3(back[0], back[1], back[2]);
        rend.camera.update();  // update transform matrix from these values
        if (fy <= 0.f) {
            rend.camera.fy = rend.camera.fx;
        }

        glfwGetFramebufferSize(window, &width, &height);
        rend.resize(width, height);

        // Set user pointer and callbacks
        glfwSetWindowUserPointer(window, &rend);
        glfwSetKeyCallback(window, glfw_key_callback);
        glfwSetMouseButtonCallback(window, glfw_mouse_button_callback);
        glfwSetCursorPosCallback(window, glfw_cursor_pos_callback);
        glfwSetScrollCallback(window, glfw_scroll_callback);
        glfwSetFramebufferSizeCallback(window, glfw_window_size_callback);

        while (!glfwWindowShouldClose(window)) {
            glEnable(GL_DEPTH_TEST);
            glfw_update_title(window);
            rend.render();
            if (!nogui) draw_imgui(rend);
            glfwSwapBuffers(window);
            glFinish();
            glfwPollEvents();
        }
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
}