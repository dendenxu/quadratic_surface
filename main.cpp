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
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // BEGIN gizmo handling
    // clang-format off
    static glm::mat4 camera_persp_prj(1.f, 0.f, 0.f, 0.f,
                                      0.f, 1.f, 0.f, 0.f,
                                      0.f, 0.f, -1.f, -1.f,
                                      0.f, 0.f, -0.001f, 0.f);
    // clang-format on
    ImGuiIO& io = ImGui::GetIO();

    camera_persp_prj[0][0] = cam.fx / cam.width * 2;
    camera_persp_prj[1][1] = cam.fy / cam.height * 2;
    ImGuizmo::SetOrthographic(false);
    ImGuizmo::SetGizmoSizeClipSpace(0.05f);

    ImGuizmo::BeginFrame();

    ImGuizmo::SetRect(0, 0, io.DisplaySize.x, io.DisplaySize.y);
    glm::mat4 w2c = glm::affineInverse(glm::mat4(cam.transform));
    // END gizmo handling

    ImGui::SetNextWindowPos(ImVec2(20.f, 20.f), ImGuiCond_Once);
    ImGui::SetNextWindowSize(ImVec2(340.f, 400.f), ImGuiCond_Once);

    static char title[128] = {0};
    if (title[0] == 0) {
        sprintf(title, "volrend backend: %s", rend.get_backend());
    }

    // Begin window
    ImGui::Begin(title);
    // static ImGui::FileBrowser open_npz_guide_dialog(
    //     ImGuiFileBrowserFlags_MultipleSelection);
    // if (open_npz_guide_dialog.GetTitle().empty()) {
    //     open_npz_guide_dialog.SetTypeFilters({".npz"});
    //     open_npz_guide_dialog.SetTitle("Load guide mesh NPZ");
    // }
    // static ImGui::FileBrowser open_npz_poses_dialog(
    //     ImGuiFileBrowserFlags_MultipleSelection);
    // if (open_npz_poses_dialog.GetTitle().empty()) {
    //     open_npz_poses_dialog.SetTypeFilters({".npz"});
    //     open_npz_poses_dialog.SetTitle("Load human poses NPZ");
    // }
    // static ImGui::FileBrowser open_obj_mesh_dialog(
    //     ImGuiFileBrowserFlags_MultipleSelection);
    // if (open_obj_mesh_dialog.GetTitle().empty()) {
    //     open_obj_mesh_dialog.SetTypeFilters({".obj"});
    //     open_obj_mesh_dialog.SetTitle("Load basic triangle OBJ");
    // }
    static ImGui::FileBrowser save_screenshot_dialog(ImGuiFileBrowserFlags_EnterNewFilename);
    // if (open_tree_dialog.GetTitle().empty()) {
    //     open_tree_dialog.SetTypeFilters({".npz"});
    //     open_tree_dialog.SetTitle("Load N3Tree npz from svox");
    // }
    if (save_screenshot_dialog.GetTitle().empty()) {
        save_screenshot_dialog.SetTypeFilters({".png"});
        save_screenshot_dialog.SetTitle("Save screenshot (png)");
    }

    // if (ImGui::Button("Open Tree")) {
    //     open_tree_dialog.Open();
    // }
    // ImGui::SameLine();
    // if (ImGui::Button("Load Guide Mesh")) {
    //     open_npz_guide_dialog.Open();
    // }
    // if (human.verts.size()) {
    //     ImGui::SameLine();
    //     if (ImGui::Button("Load Human Poses")) {
    //         open_npz_poses_dialog.Open();
    //     }
    // }
    // if (ImGui::Button("Save Screenshot")) {
    //     save_screenshot_dialog.Open();
    // }

    // open_tree_dialog.Display();
    // if (open_tree_dialog.HasSelected()) {
    //     // Load octree
    //     std::string path = open_tree_dialog.GetSelected().string();
    //     printf("Load N3Tree npz: %s\n", path.c_str());
    //     tree.open(path);
    //     rend.set(tree);
    //     if (human.verts.size()) {
    //         rend.options.render_guide_mesh = false;  // clear the guide mesh
    //     }
    //     open_tree_dialog.ClearSelected();
    // }

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
        // ImGui::InputFloat("clip_near", &cam.clip_near);

        static bool lock_fx_fy = true;
        static float fx = cam.fx;  // gui state variable for duplication
        static float fy = cam.fy;  // gui state variable for duplication (like one slider + one input for same location)
        ImGui::Checkbox("fx=fy", &lock_fx_fy);
        if (lock_fx_fy) {
            if (ImGui::SliderFloat("focal", &cam.fx, 10.f, 50000.f)) {
                cam.fy = cam.fx;
            }
            // if (ImGui::InputFloat("focal", &cam.fx)) {
            //     cam.fy = cam.fx;
            // }
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
            ImGui::InputFloat3("x", &cam.transform[0][0]);
            ImGui::InputFloat3("y", &cam.transform[1][0]);
            ImGui::InputFloat3("z", &cam.transform[2][0]);
            ImGui::InputFloat3("t", &cam.transform[3][0]);
            ImGui::TreePop();
        }
    }  // End camera node

    // ImGui::SetNextTreeNodeOpen(true, ImGuiCond_Once);
    // if (ImGui::CollapsingHeader("Render")) {
    //     static float inv_step_size = 1.0f / rend.options.step_size;
    //     if (ImGui::SliderFloat("1/overshoot", &inv_step_size, 128.f, 20000.f)) {
    //         rend.options.step_size = 1.f / inv_step_size;
    //     }
    //     ImGui::SliderFloat("sigma_thresh", &rend.options.sigma_thresh, 0.f, 100.0f);
    //     ImGui::SliderFloat("stop_thresh", &rend.options.stop_thresh, 0.001f, 0.4f);
    //     ImGui::SliderFloat("bg_brightness", &rend.options.background_brightness, 0.f, 1.0f);
    //     if (!rend.options.show_template) {
    //         ImGui::SliderFloat("t_off_in", &rend.options.t_off_in, -1.0f, 3.0f);
    //         ImGui::SliderFloat("t_off_out", &rend.options.t_off_out, -1.0f, 3.0f);
    //     }
    //     if ((human.n_poses()) && !rend.options.show_template) {
    //         if (ImGui::SliderInt("pose", &selected_pose, 0, GLsizei(human.n_poses() - 1))) {
    //             human.select_pose(selected_pose);
    //         }
    //     }

    // }  // End render node
    //     ImGui::SetNextTreeNodeOpen(true, ImGuiCond_Once);
    //     if (ImGui::CollapsingHeader("Visualization")) {
    //         ImGui::PushItemWidth(230);
    //         ImGui::SliderFloat3("bb_min", rend.options.render_bbox, 0.0, 1.0);
    //         ImGui::SliderFloat3("bb_max", rend.options.render_bbox + 3, 0.0, 1.0);
    //         ImGui::SliderInt2("decomp", rend.options.basis_minmax, 0, std::max(tree.data_format.basis_dim - 1, 0));
    //         ImGui::SliderFloat3("viewdir shift", rend.options.rot_dirs, float(-M_PI / 4), float(M_PI / 4));
    //         ImGui::PopItemWidth();
    //         if (ImGui::Button("Reset Viewdir Shift")) {
    //             for (int i = 0; i < 3; ++i) rend.options.rot_dirs[i] = 0.f;
    //         }

    //         ImGui::SliderFloat("Vis Color Multi", &rend.options.vis_color_multiplier, 0.0f, 10.0f);
    //         ImGui::SliderFloat("Vis Color Off", &rend.options.vis_color_offset, -1.0f, 1.0f);

    //         ImGui::Checkbox("Apply Sigmoid", &rend.options.apply_sigmoid);
    //         ImGui::Checkbox("Big-Pose Geometry", &rend.options.bigpose_geometry);
    //         ImGui::Checkbox("Use Face Normal", &rend.options.use_face_normal);

    //         ImGui::Checkbox("Show Grid", &rend.options.show_grid);
    //         ImGui::Checkbox("Visualize Unseen", &rend.options.visualize_unseen);
    //         ImGui::Checkbox("Visualize Intensity", &rend.options.visualize_intensity);
    //         if (tree.extra_slot == 3) {
    //             ImGui::Checkbox("Use Offset", &rend.options.use_offset);
    //             if (rend.options.use_offset) {
    //                 ImGui::Checkbox("Visualize Offset", &rend.options.visualize_offset);
    //                 if (rend.options.visualize_offset) {
    //                     ImGui::SliderFloat("Vis Offset Multi", &rend.options.vis_offset_multiplier, 1.0f, 50.0f);
    //                 }
    //             }
    //         }
    //         if (tree.capacity && !rend.options.show_template) {
    //             ImGui::Checkbox("Render Dynamic NeRF", &rend.options.render_dynamic_nerf);
    //         }

    //         if (human.verts.size() && !rend.options.show_template) {
    //             ImGui::Checkbox("Render Guide Mesh", &rend.options.render_guide_mesh);
    //             if (rend.options.render_guide_mesh) {
    //                 ImGui::Checkbox("Show Bone Association", &rend.options.show_bone_association);
    //             }
    //         }
    //         if (human.all_poses.size() && !rend.options.show_template) {
    //             ImGui::Checkbox("Play Animation", &play_animation);
    //         }
    // #ifdef VOLREND_CUDA
    //         ImGui::SameLine();
    //         ImGui::Checkbox("Render Depth", &rend.options.render_depth);
    // #endif
    //         if (rend.options.show_grid) {
    //             ImGui::SliderInt("grid max depth", &rend.options.grid_max_depth, 0, 7);
    //         }
    //     }

    // ImGui::SetNextTreeNodeOpen(true, ImGuiCond_Once);
    // if (ImGui::CollapsingHeader("Manipulation")) {
    //     static std::vector<glm::mat4> gizmo_mesh_trans;
    //     gizmo_mesh_trans.resize(rend.meshes.size());

    //     ImGui::TextUnformatted("gizmo op");
    //     ImGui::SameLine();
    //     ImGui::RadioButton("trans##giztrans", &gizmo_mesh_op,
    //                        ImGuizmo::TRANSLATE);
    //     ImGui::SameLine();
    //     ImGui::RadioButton("rot##gizrot", &gizmo_mesh_op, ImGuizmo::ROTATE);
    //     ImGui::SameLine();
    //     ImGui::RadioButton("scale##gizscale", &gizmo_mesh_op,
    //                        ImGuizmo::SCALE_Z);

    //     ImGui::TextUnformatted("gizmo space");
    //     ImGui::SameLine();
    //     ImGui::RadioButton("local##gizlocal", &gizmo_mesh_space,
    //                        ImGuizmo::LOCAL);
    //     ImGui::SameLine();
    //     ImGui::RadioButton("world##gizworld", &gizmo_mesh_space,
    //                        ImGuizmo::WORLD);

    // ImGui::BeginGroup();
    // std::vector<int> meshes_to_del;
    // for (int i = 0; i < (int)rend.meshes.size(); ++i) {
    //     auto& mesh = rend.meshes[i];
    //     if (ImGui::TreeNode(mesh.name.c_str())) {
    //         if (mesh.visible) {
    //             glm::mat4& gizmo_trans = gizmo_mesh_trans[i];
    //             gizmo_trans = mesh.transform_;
    //             if (gizmo_mesh_op == ImGuizmo::SCALE_Z) {
    //                 glm::mat4 tmp(1);
    //                 tmp[3] = gizmo_trans[3];
    //                 gizmo_trans = tmp;
    //             }
    //             ImGuizmo::SetID(i + 1);
    //             if (ImGuizmo::Manipulate(glm::value_ptr(w2c),
    //                                      glm::value_ptr(camera_persp_prj),
    //                                      (ImGuizmo::OPERATION)gizmo_mesh_op,
    //                                      (ImGuizmo::MODE)gizmo_mesh_space,
    //                                      glm::value_ptr(gizmo_trans), NULL,
    //                                      NULL, NULL, NULL)) {
    //                 if (gizmo_mesh_op == ImGuizmo::ROTATE) {
    //                     glm::quat rot_q = glm::quat_cast(
    //                         glm::mat3(gizmo_trans) / mesh.scale);
    //                     mesh.rotation =
    //                         glm::axis(rot_q) * glm::angle(rot_q);
    //                 } else if (gizmo_mesh_op == ImGuizmo::SCALE_Z) {
    //                     mesh.scale *= gizmo_trans[2][2] /
    //                                   mesh.transform_[2][2];  // max_scale;
    //                 }
    //                 mesh.translation = gizmo_trans[3];
    //             }
    //         }
    //         ImGui::PushItemWidth(230);
    //         ImGui::InputFloat3("trans", glm::value_ptr(mesh.translation));
    //         ImGui::InputFloat3("rot", glm::value_ptr(mesh.rotation));
    //         ImGui::InputFloat("scale", &mesh.scale);
    //         ImGui::PopItemWidth();
    //         ImGui::Checkbox("visible", &mesh.visible);
    //         ImGui::SameLine();
    //         ImGui::Checkbox("unlit", &mesh.unlit);
    //         ImGui::SameLine();
    //         if (ImGui::Button("delete")) meshes_to_del.push_back(i);

    //         ImGui::TreePop();
    //     }
    // }

    // if (meshes_to_del.size()) {
    //     int j = 0;
    //     std::vector<Mesh> tmp;
    //     tmp.reserve(rend.meshes.size() - meshes_to_del.size());
    //     for (int i = 0; i < rend.meshes.size(); ++i) {
    //         if (i == meshes_to_del[j]) {
    //             ++j;
    //             continue;
    //         }
    //         tmp.push_back(std::move(rend.meshes[i]));
    //     }
    //     rend.meshes.swap(tmp);
    // }
    // ImGui::EndGroup();
    // if (ImGui::Button("Sphere##addsphere")) {
    //     static int sphereid = 0;
    //     {
    //         Mesh sph = Mesh::Sphere();
    //         sph.scale = 0.1f;
    //         sph.translation[2] = 1.0f;
    //         sph.update();
    //         if (sphereid) sph.name = sph.name + std::to_string(sphereid);
    //         ++sphereid;
    //         rend.meshes.push_back(std::move(sph));
    //     }
    // }
    // ImGui::SameLine();
    // if (ImGui::Button("Cube##addcube")) {
    //     static int cubeid = 0;
    //     {
    //         Mesh cube = Mesh::Cube();
    //         cube.scale = 0.2f;
    //         cube.translation[2] = 1.0f;
    //         cube.update();
    //         if (cubeid) cube.name = cube.name + std::to_string(cubeid);
    //         ++cubeid;
    //         rend.meshes.push_back(std::move(cube));
    //     }
    // }
    // ImGui::SameLine();
    // if (ImGui::Button("Latti##addlattice")) {
    //     static int lattid = 0;
    //     {
    //         Mesh latt = Mesh::Lattice();
    //         if (tree.N > 0) {
    //             latt.scale =
    //                 1.f / std::min(std::min(tree.scale[0], tree.scale[1]),
    //                                tree.scale[2]);
    //             for (int i = 0; i < 3; ++i) {
    //                 latt.translation[i] =
    //                     -1.f / tree.scale[0] * tree.offset[0];
    //             }
    //         }
    //         latt.update();
    //         if (lattid) latt.name = latt.name + std::to_string(lattid);
    //         ++lattid;
    //         rend.meshes.push_back(std::move(latt));
    //     }
    // }
    // ImGui::SameLine();
    // if (ImGui::Button("Load OBJ")) {
    //     open_obj_mesh_dialog.Open();
    // }
    // ImGui::SameLine();
    // if (ImGui::Button("Clear All")) {
    //     rend.meshes.clear();
    // }

    // // #ifdef VOLREND_CUDA
    // if (tree.capacity) {
    //     ImGui::BeginGroup();
    //     ImGui::Checkbox("Show Template NeRF",
    //                     &rend.options.show_template);
    //     ImGui::Checkbox("Enable Lumisphere Probe",
    //                     &rend.options.enable_probe);
    //     if (rend.options.enable_probe) {
    //         ImGui::SetNextTreeNodeOpen(true, ImGuiCond_Once);
    //         if (ImGui::TreeNode("Probe")) {
    //             static glm::mat4 probe_trans;
    //             static bool show_probe_gizmo = true;
    //             float* probe = rend.options.probe;
    //             probe_trans =
    //                 glm::translate(glm::mat4(1.f),
    //                                glm::vec3(probe[0], probe[1], probe[2]));

    //             ImGui::Checkbox("Show gizmo", &show_probe_gizmo);
    //             if (show_probe_gizmo) {
    //                 ImGuizmo::SetID(0);
    //                 if (ImGuizmo::Manipulate(
    //                         glm::value_ptr(w2c),
    //                         glm::value_ptr(camera_persp_prj),
    //                         ImGuizmo::TRANSLATE, ImGuizmo::LOCAL,
    //                         glm::value_ptr(probe_trans), NULL, NULL, NULL,
    //                         NULL)) {
    //                     for (int i = 0; i < 3; ++i)
    //                         probe[i] = probe_trans[3][i];
    //                 }
    //             }
    //             ImGui::InputFloat3("probe", probe);
    //             ImGui::SliderInt("probe_win_sz",
    //                              &rend.options.probe_disp_size, 50, 800);
    //             ImGui::TreePop();
    //         }
    //     }
    //     ImGui::EndGroup();
    // }
    // // #endif
    // }
    // open_obj_mesh_dialog.Display();
    // if (open_obj_mesh_dialog.HasSelected()) {
    //     // Load mesh
    //     auto sels = open_obj_mesh_dialog.GetMultiSelected();
    //     for (auto& fpath : sels) {
    //         const std::string path = fpath.string();
    //         printf("Load OBJ: %s\n", path.c_str());
    //         Mesh tmp = Mesh::load_basic_obj(path);
    //         if (tmp.vert.size()) {
    //             // Auto offset
    //             std::ifstream ifs(path + ".offs");
    //             if (ifs) {
    //                 ifs >> tmp.translation.x >> tmp.translation.y >>
    //                     tmp.translation.z;
    //                 if (ifs) {
    //                     ifs >> tmp.scale;
    //                 }
    //             }
    //             tmp.update();
    //             rend.meshes.push_back(std::move(tmp));
    //             puts("Load success\n");
    //         } else {
    //             puts("Load failed\n");
    //         }
    //     }
    //     open_obj_mesh_dialog.ClearSelected();
    // }

    // open_npz_guide_dialog.Display();
    // if (open_npz_guide_dialog.HasSelected()) {
    //     // Load mesh
    //     auto sels = open_npz_guide_dialog.GetMultiSelected();
    //     for (auto& fpath : sels) {
    //         const std::string path = fpath.string();
    //         printf("Load NPZ Guide Mesh: %s\n", path.c_str());
    //         human = Human::load_human_npz(path);
    //         if (human.verts.size()) {
    //             human.update();
    //             rend.set(human);
    //             puts("Load success\n");
    //             rend.options.show_template = false;
    //             rend.options.render_guide_mesh = !tree.capacity;  // show guide mesh is tree hasn't been loaded
    //         } else {
    //             puts("Load failed\n");
    //         }
    //     }
    //     open_npz_guide_dialog.ClearSelected();
    // }

    // open_npz_poses_dialog.Display();
    // if (open_npz_poses_dialog.HasSelected()) {
    //     // Load mesh
    //     auto sels = open_npz_poses_dialog.GetMultiSelected();
    //     for (auto& fpath : sels) {
    //         const std::string path = fpath.string();
    //         printf("Load NPZ Human Poses: %s\n", path.c_str());
    //         human.load_poses_npz(path);  // FIXME: will discard all loaded poses if selecting another guide mesh
    //         if (human.all_poses.size()) {
    //             puts("Load success\n");
    //         } else {
    //             puts("Load failed\n");
    //         }
    //     }
    //     open_npz_poses_dialog.ClearSelected();
    // }

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
            float speed = 0.002f;
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
                    printf("%.10f", cam.transform[j][i]);
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

        case GLFW_KEY_I:
        case GLFW_KEY_J:
        case GLFW_KEY_K:
        case GLFW_KEY_L:
        case GLFW_KEY_U:
        case GLFW_KEY_O:
            if (rend.options.enable_probe) {
                // Probe movement
                float speed = 0.002f;
                if (mods & GLFW_MOD_SHIFT) speed *= 5.f;
                if (key == GLFW_KEY_J || key == GLFW_KEY_K ||
                    key == GLFW_KEY_U)
                    speed = -speed;
                int dim = (key == GLFW_KEY_J || key == GLFW_KEY_L)   ? 0
                          : (key == GLFW_KEY_I || key == GLFW_KEY_K) ? 1
                                                                     : 2;
                rend.options.probe[dim] += speed;
            }
            break;

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
                       button == GLFW_MOUSE_BUTTON_RIGHT || (button == GLFW_MOUSE_BUTTON_MIDDLE && SHIFT));
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
#ifdef VOLREND_CUDA
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
#else
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
#endif
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
    // when it should actually have been a few thousands
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
                cxxopts::value<std::vector<float>>()->default_value("-3.5,0,3.5"))
        ("back", "camera's back direction unit vector (world) for orientation; ignored for NDC",
                cxxopts::value<std::vector<float>>()->default_value("-0.7071068,0,0.7071068"))
        ("origin", "origin for right click rotation controls; ignored for NDC",
                cxxopts::value<std::vector<float>>()->default_value("0,0,0"))
        ("world_up", "world up direction for rotating controls e.g. "
                     "0,0,1=blender; ignored for NDC",
                cxxopts::value<std::vector<float>>()->default_value("0,0,1"))
        ("grid", "show grid with given max resolution (4 is reasonable)", cxxopts::value<int>())
        ("probe", "enable lumisphere_probe and place it at given x,y,z",
                   cxxopts::value<std::vector<float>>())
        ;
    // clang-format on

    cxxoptions.positional_help("npz_file");

    cxxopts::ParseResult args = internal::parse_options(cxxoptions, argc, argv);

#ifdef VOLREND_CUDA
    const int device_id = args["gpu"].as<int>();
    if (~device_id) {
        cuda(SetDevice(device_id));
    }
#endif

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

        rend.options = internal::render_options_from_args(args);
        {
            auto cen = args["center"].as<std::vector<float>>();
            rend.camera.center = glm::vec3(cen[0], cen[1], cen[2]);
            auto origin = args["origin"].as<std::vector<float>>();
            rend.camera.origin = glm::vec3(origin[0], origin[1], origin[2]);
            auto world_up = args["world_up"].as<std::vector<float>>();
            rend.camera.v_world_up = glm::vec3(world_up[0], world_up[1], world_up[2]);
            auto back = args["back"].as<std::vector<float>>();
            rend.camera.v_back = glm::vec3(back[0], back[1], back[2]);
        }
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