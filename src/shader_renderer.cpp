#include "volrend/common.hpp"

// Shader backend only enabled when build with VOLREND_USE_CUDA=OFF
#ifndef VOLREND_CUDA
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <glm/gtc/type_ptr.hpp>
#include <string>

#include "volrend/mesh.hpp"
#include "volrend/renderer.hpp"

#ifdef __EMSCRIPTEN__
// WebGL
#include <GLES3/gl3.h>
#else
#include <GL/glew.h>
#endif

#include "volrend/internal/rt_frag.inl"
#include "volrend/internal/shader.hpp"

namespace volrend {

namespace {

const char* GUIDE_MESH_VERT_SHADER_SRC =
    R"glsl(
#version 330
#pragma vscode_glsllint_stage: vert
#define NUM_POSE_BONE 24
#define NUM_BONE 4

// Computer vision style camera
struct Camera {
    mat4x3 transform;
    mat4x4 w2c;
    mat4x4 K;
    vec2 reso;
    vec2 focal;
};


// FIXME: HEADACHE, should define these in the same place?
// Store render options
struct RenderOptions {
    // Epsilon added to each step
    float step_size;
    // If remaining light intensity/alpha < this amount stop marching
    float stop_thresh;
    // If sigma < this, skips
    float sigma_thresh;
    // Background brightness
    float background_brightness;

    // Rendering bounding box (relative to outer tree bounding box [0, 1])
    // [minx, miny, minz, maxx, maxy, maxz]
    float render_bbox[6];
    // Range of basis functions to use
    int basis_minmax[2];
    // Rotation applied to viewdirs for all rays
    vec3 rot_dirs;

    float t_off_out;  // how much to render outside the guide mesh
    float t_off_in;   // how much to render inside the mesh

    bool use_offset;  // FIXME: should check whether extra_slot is available
    bool visualize_offset;

    bool visualize_unseen;
    bool visualize_intensity;

    float vis_offset_multiplier;

    float vis_color_multiplier;
    float vis_color_offset;

    bool apply_sigmoid;

    vec3 probe;
    float probe_disp_size;

    bool show_template;
    bool bigpose_geometry;
};

uniform bool drawing_probe; // THIS IS AN IMPORTANT SWITCH FOR THE SHADER

uniform Camera cam;
uniform RenderOptions opt; // THE NAME IS OPT NOT OPTS

uniform mat4x4 M; // human object space transformation
uniform mat4x4 rigid[NUM_POSE_BONE]; // pose should be a set of 24 affine transfromation
uniform mat4x4 bigrigid[NUM_POSE_BONE]; // pose should be a set of 24 affine transfromation
// this transformation should map a point in tpose to world

layout(location=0) in vec3 a_pos; // location 0
layout(location=1) in vec3 a_deform;
layout(location=2) in vec3 a_normal;
layout(location=3) in vec4 a_blend_weights; // blending weight
layout(location=4) in vec4 a_bone_assoc; // bone index for blending weight

out highp vec4 tpose_cam_center;
out highp vec4 tpose_frag_pos;
out highp vec4 deform_frag_pos;
out highp vec4 world_frag_pos;
out mat4 p2t; // inverse the blended rigid transformation matrix

void main()
{
    if (drawing_probe) {
        // // Scale to where the probe should live in
        vec2 pos = vec2(a_pos.x, a_pos.y);
        pos = (1 - (pos + 1) / 2 * (opt.probe_disp_size / cam.reso) - 5 / cam.reso) * 2 - 1;
        // // PASSING THROUGH
        gl_Position = vec4(pos, a_pos.z, 1.0);
    } else if (opt.show_template) {
        // ! Depth buffer test goes on here
        gl_Position = vec4(a_pos.x, a_pos.y, 0.75, 1.0); // lumisphere should be draw to the front
    } else {
        // linear blend skinning
        mat4 t2p = mat4(0);
        mat4 p2t = mat4(0);
        float bw_sum = 0.f;

        {
        // from tpose points to world pose points
            for (int i = 0; i < NUM_BONE; i++) {
                bw_sum += a_blend_weights[i]; // if this doesn't sums up to one, we normalize it
                t2p += a_blend_weights[i] * rigid[int(a_bone_assoc[i])]; // FIXME: is this OK, int should be smaller than float right...
            }
            t2p /= bw_sum; // normalize
        }

        if (opt.bigpose_geometry) {
            mat4 b2t = mat4(0);
            bw_sum = 0;
            for (int i = 0; i < NUM_BONE; i++) {
                bw_sum += a_blend_weights[i]; // if this doesn't sums up to one, we normalize it
                b2t += a_blend_weights[i] * inverse(bigrigid[int(a_bone_assoc[i])]);
            }
            b2t /= bw_sum; // normalize
            t2p = t2p * b2t;

        }

        t2p = M * t2p; // add model transformation

        p2t = inverse(t2p);

        tpose_cam_center = vec4(cam.transform[3], 1.f); // This camera center should do a inverse transformations
        tpose_cam_center = p2t * tpose_cam_center; // go from world cam position to pose cam position

        tpose_frag_pos = vec4(a_pos, 1.f); // in object space
        deform_frag_pos = vec4(a_pos + a_deform, 1.f); // in deformed space
        world_frag_pos = t2p * deform_frag_pos; // apply animation

        gl_Position = cam.K * cam.w2c * world_frag_pos; // projected onto the screen
    }
}
)glsl";

const float quad_verts[] = {
    -1.f,
    -1.f,
    0.5f,
    1.f,
    -1.f,
    0.5f,
    -1.f,
    1.f,
    0.5f,
    1.f,
    1.f,
    0.5f,
};

struct _RenderUniforms {
    GLint cam_transform, cam_focal, cam_reso, cam_K, cam_w2c;
    GLint opt_step_size, opt_backgrond_brightness, opt_stop_thresh,
        opt_sigma_thresh, opt_render_bbox, opt_basis_minmax, opt_rot_dirs, opt_t_off_in, opt_t_off_out, opt_use_offset, opt_visualize_offset, opt_visualize_unseen, opt_visualize_intensity, opt_vis_offset_multiplier, opt_vis_color_multiplier, opt_vis_color_offset,
        opt_apply_sigmoid,
        opt_probe,
        opt_probe_disp_size,
        opt_show_template;
    GLint tree_data_tex, tree_child_tex;
    GLint mesh_depth_tex, mesh_color_tex;
    GLint rigid;             // LBS
    GLint bigrigid;          // LBS
    GLint M;                 // Model transform
    GLint drawing_probe;     // to avoid loading texture once again
    GLint bigpose_geometry;  // to avoid loading texture once again
};

}  // namespace

struct VolumeRenderer::Impl {
    Impl(Camera& camera, RenderOptions& options, std::vector<Mesh>& meshes,
         int max_tries = 4)
        : camera(camera), options(options), meshes(meshes) {
        probe_ = Mesh::Cube(glm::vec3(0.0));
        probe_.name = "_probe_cube";
        probe_.visible = false;
        probe_.scale = 0.05f;
        // Make face colors
        for (int i = 0; i < 3; ++i) {
            int off = i * 12 * 9;
            for (int j = 0; j < 12; ++j) {
                int soff = off + 9 * j + 3;
                probe_.vert[soff + 2 - i] = 1.f;
            }
        }
        probe_.unlit = true;
        probe_.update();
        wire_.face_size = 2;
        wire_.unlit = true;
    }

    ~Impl() {
        glDeleteProgram(program);
        glDeleteFramebuffers(1, &fb);
        glDeleteTextures(1, &tex_tree_data);
        glDeleteTextures(1, &tex_tree_child);
        glDeleteTextures(1, &tex_tree_extra);
        glDeleteTextures(1, &tex_mesh_color);
        glDeleteTextures(1, &tex_mesh_depth);

        // TODO: use a render buffer
        glDeleteTextures(1, &tex_mesh_depth_buf);
    }

    void start() {
        if (started_) return;
        glGetIntegerv(GL_MAX_TEXTURE_SIZE, &tex_max_size);
        // int tex_3d_max_size;
        // glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &tex_3d_max_size);

        /** 
         * n
         * Specifies the number of texture names to be generated.

         * textures
         * Specifies an array in which the generated texture names are stored.
         */
        glGenTextures(1, &tex_tree_data);  // generate 1 texture, store reference in tex_tree_data
        glGenTextures(1, &tex_tree_child);
        glGenTextures(1, &tex_tree_extra);

        glGenTextures(1, &tex_mesh_color);
        glGenTextures(1, &tex_mesh_depth);

        // TODO: use a render buffer
        glGenTextures(1, &tex_mesh_depth_buf);

        glGenFramebuffers(1, &fb);  // generate 1 framebuffer, storereference in fb

        // Put some dummy information to suppress browser warnings
        glBindTexture(GL_TEXTURE_2D, tex_tree_data);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R16F, 1, 1, 0, GL_RED, GL_HALF_FLOAT,
                     nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glBindTexture(GL_TEXTURE_2D, tex_tree_child);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32I, 1, 1, 0, GL_RED_INTEGER, GL_INT,
                     nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glBindTexture(GL_TEXTURE_2D, 0);

        resize(800, 800);

        /** 
         * An attachment is a memory location that can act as a buffer for the framebuffer, think of it as an image. When creating an attachment we have two options to take: textures or renderbuffer objects.
         */
        glBindFramebuffer(GL_FRAMEBUFFER, fb);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, tex_mesh_color, 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1,
                               GL_TEXTURE_2D, tex_mesh_depth, 0);

        // TODO: use a render buffer
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                               GL_TEXTURE_2D, tex_mesh_depth_buf, 0);
        const GLenum attach_buffers[]{GL_COLOR_ATTACHMENT0,
                                      GL_COLOR_ATTACHMENT1};
        glDrawBuffers(2, attach_buffers);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) !=
            GL_FRAMEBUFFER_COMPLETE) {
            fprintf(stderr, "Framebuffer not complete\n");
            std::exit(1);
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        quad_init();
        shader_init();
        started_ = true;
    }

    void render() {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);  // ? what is bound, 0 ?
        if (!started_) return;

        probe_.visible = options.enable_probe;
        for (int i = 0; i < 3; ++i) probe_.translation[i] = options.probe[i];

        camera._update();
        if (options.show_grid) {
            maybe_gen_wire(options.grid_max_depth);
        }

        GLfloat clear_color[] = {options.background_brightness,
                                 options.background_brightness,
                                 options.background_brightness, 1.f};
        GLfloat depth_inf = 1e9, zero = 0;

        /** 
         * Since our framebuffer is not the default framebuffer, the rendering commands will have no impact on the visual output of your window. For this reason it is called off-screen rendering when rendering to a different framebuffer.
         * It is also possible to bind a framebuffer to a read or write target specifically by binding to GL_READ_FRAMEBUFFER or GL_DRAW_FRAMEBUFFER respectively. The framebuffer bound to GL_READ_FRAMEBUFFER is then used for all read operations like glReadPixels and the framebuffer bound to GL_DRAW_FRAMEBUFFER is used as the destination for rendering, clearing and other write operations. Most of the times you won't need to make this distinction though and you generally bind to both with GL_FRAMEBUFFER.
         */
        glBindFramebuffer(GL_FRAMEBUFFER, fb);

        glDepthMask(GL_TRUE);  // this asks OpenGL to write depth info to depth buffer
#ifdef __EMSCRIPTEN__
        // GLES 3
        glClearDepthf(1.f);
#else
        glClearDepth(1.f);
#endif
        glClearBufferfv(GL_COLOR, 0, clear_color);
        glClearBufferfv(GL_COLOR, 1, &depth_inf);
        glClearBufferfv(GL_DEPTH, 0, &depth_inf);

        if (human->verts.size() && options.render_guide_mesh) {
            human->use_shader();                           // using human shader
            (*human).draw(camera.w2c, camera.K, options);  // ! And this is why you never omit params
        }

        Mesh::use_shader();  // using mesh shader
        for (const Mesh& mesh : meshes) {
            mesh.draw(camera.w2c, camera.K);
        }
        probe_.draw(camera.w2c, camera.K);  // TODO: What's this? The probe is drawn manually
        if (options.show_grid) {
            wire_.draw(camera.w2c, camera.K);
        }

        glBindFramebuffer(GL_READ_FRAMEBUFFER, fb);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        glBlitFramebuffer(0, 0, camera.width, camera.height, 0, 0, camera.width, camera.height,
                          GL_COLOR_BUFFER_BIT, GL_NEAREST);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        glUseProgram(program);  // using tree shader

        // FIXME reduce uniform transfers?
        glUniformMatrix4x3fv(u.cam_transform, 1, GL_FALSE, glm::value_ptr(camera.transform));
        glUniformMatrix4fv(u.cam_w2c, 1, GL_FALSE, glm::value_ptr(camera.w2c));                         // loading camera w2c to GPU
        glUniformMatrix4fv(u.cam_K, 1, GL_FALSE, glm::value_ptr(camera.K));                             // loading camera K
        glUniformMatrix4fv(u.rigid, POSE_BONE_SZ, GL_FALSE, (GLfloat*)(&human->rigid[0]));              // rigid transformation for linear blend skinning
        glUniformMatrix4fv(u.bigrigid, POSE_BONE_SZ, GL_FALSE, (GLfloat*)(&human->get_bigrigid()[0]));  // rigid transformation for linear blend skinning
        glUniformMatrix4fv(u.M, 1, GL_FALSE, glm::value_ptr(human->transform));                         // rigid transformation for linear blend skinning

        glUniform2f(u.cam_focal, camera.fx, camera.fy);
        glUniform2f(u.cam_reso, (float)camera.width, (float)camera.height);
        glUniform1f(u.opt_step_size, options.step_size);  // TODO: poor naming
        glUniform1f(u.opt_backgrond_brightness, options.background_brightness);
        glUniform1f(u.opt_stop_thresh, options.stop_thresh);
        glUniform1f(u.opt_sigma_thresh, options.sigma_thresh);
        glUniform1fv(u.opt_render_bbox, 6, options.render_bbox);
        glUniform1iv(u.opt_basis_minmax, 2, options.basis_minmax);
        glUniform3fv(u.opt_rot_dirs, 1, options.rot_dirs);
        glUniform1f(u.opt_t_off_in, options.t_off_in);
        glUniform1f(u.opt_t_off_out, options.t_off_out);
        glUniform1i(u.opt_use_offset, options.use_offset);
        glUniform1i(u.opt_visualize_offset, options.visualize_offset);
        glUniform1i(u.opt_visualize_unseen, options.visualize_unseen);
        glUniform1i(u.opt_visualize_intensity, options.visualize_intensity);
        glUniform1f(u.opt_vis_offset_multiplier, options.vis_offset_multiplier);
        glUniform1f(u.opt_vis_color_multiplier, options.vis_color_multiplier);
        glUniform1f(u.opt_vis_color_offset, options.vis_color_offset);
        glUniform1f(u.opt_apply_sigmoid, options.apply_sigmoid);
        glUniform1i(u.opt_show_template, options.show_template);
        glUniform3fv(u.opt_probe, 1, options.probe);
        glUniform1f(u.opt_probe_disp_size, (float)options.probe_disp_size);
        glUniform1i(u.drawing_probe, false);
        glUniform1i(u.bigpose_geometry, options.bigpose_geometry);

        // FIXME Probably can be done only once
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex_tree_child);

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, tex_tree_data);

        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, tex_mesh_depth);  // tex_mesh_depth_buf is never used...

        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_2D, tex_mesh_color);

        // TODO: implement LBS first, using quad for now
        if (options.render_dynamic_nerf) {
            if (human->verts.size() && !options.show_template) {
                // When the guide mesh exists, load its vertex/face data and render from it
                glBindVertexArray(human->vao_);
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, human->ebo_);
                glDrawElements(GL_TRIANGLES, (GLsizei)human->faces.size(), GL_UNSIGNED_INT, (void*)0);
            } else {
                // When it doesn't exist use a quad to render the whole screen
                // TODO: current rendering the quad as you know, a quad, that transforms as the plane shifts
                // Make this deal with the whole screen
                glBindVertexArray(vao_quad);
                glDrawArrays(GL_TRIANGLE_STRIP, 0, (GLsizei)4);
            }
        }

        if (options.enable_probe) {
            // * Draw lumisphere probe, using the same shader to avoid reloading texture and resources
            glUniform1i(u.drawing_probe, true);
            glBindVertexArray(vao_quad);
            glDrawArrays(GL_TRIANGLE_STRIP, 0, (GLsizei)4);
        }

        glBindVertexArray(0);
    }

    void set(N3Tree& tree) {
        start();
        if (tree.capacity > 0) {
            this->tree = &tree;
            upload_data();
            upload_child_links();
            upload_tree_spec();
        }
        options.basis_minmax[0] = 0;
        options.basis_minmax[1] = std::max(tree.data_format.basis_dim - 1, 0);
        probe_.scale = 0.02f / tree.scale[0];
    }

    void set(Human& human) {
        this->human = &human;
    }

    void maybe_gen_wire(int depth) {
        if (last_wire_depth_ != depth) {
            wire_.vert = tree->gen_wireframe(depth);
            wire_.update();
            last_wire_depth_ = depth;
        }
    }

    void clear() { this->tree = nullptr; }

    void resize(const int width, const int height) {
        if (camera.width == width && camera.height == height) return;
        if (width <= 0 || height <= 0) return;
        camera.width = width;
        camera.height = height;

        // Re-allocate memory for textures used in mesh-volume compositing
        // process
        glBindTexture(GL_TEXTURE_2D, tex_mesh_color);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

        glBindTexture(GL_TEXTURE_2D, tex_mesh_depth);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, 0);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

        // TODO: use a render buffer
        glBindTexture(GL_TEXTURE_2D, tex_mesh_depth_buf);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

        glViewport(0, 0, width, height);
    }

    Human* get_human() { return human; }

   private:
    void auto_size_2d(size_t size, size_t& width, size_t& height, int base_dim = 1) {
        if (size == 0) {
            width = height = 0;
            return;
        }
        width = (size_t)std::sqrt(size);
        if (width % base_dim) {
            width += base_dim - width % base_dim;
        }
        height = (size - 1) / width + 1;  // ceiling
        if (height > tex_max_size || width > tex_max_size) {
            throw std::runtime_error(
                "Octree data exceeds your OpenGL driver's 2D texture limit.\n"
                "Please try the CUDA renderer or another device.");
        }
    }

    void upload_data() {
        const GLint data_size = tree->capacity * tree->N * tree->N * tree->N * tree->data_dim;
        size_t width, height;
        auto_size_2d(data_size, width, height, tree->data_dim);
        const size_t pad = width * height - data_size;

        glUseProgram(program);  // ? do we need this line?

        glUniform1i(glGetUniformLocation(program, "tree_data_dim"), (GLsizei)width);  // y axis align

#ifdef __EMSCRIPTEN__
        tree->data_.data_holder.resize((data_size + pad) * sizeof(half));
        glBindTexture(GL_TEXTURE_2D, tex_tree_data);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R16F, width, height, 0, GL_RED,
                     GL_HALF_FLOAT, tree->data_.data<half>());
#else
        // FIXME: there seems to be some weird bug in the NVIDIA OpenGL
        // implementation where GL_HALF_FLOAT is sometimes ignored, and we have
        // to use float32 for uploads
        std::vector<float> tmp(data_size + pad);
        std::copy(tree->data_.data<half>(),
                  tree->data_.data<half>() + data_size, tmp.begin());
        glBindTexture(GL_TEXTURE_2D, tex_tree_data);

        /** 
         * The format (7th argument), together with the type argument, describes the data you pass in as the last argument. So the format/type combination defines the memory layout of the data you pass in.

         * internalFormat (2nd argument) defines the format that OpenGL should use to store the data internally.
         */
        glTexImage2D(
            GL_TEXTURE_2D,     // Specifies the target texture.
            0,                 // Level n is the nth mipmap reduction image
            GL_R16F,           // Specifies the number of color components in the texture. (internalformat)
            (GLsizei)width,    // Specifies the width of the texture image.
            (GLsizei)height,   // Specifies the width of the texture image.
            0,                 // This value must be 0.
            GL_RED,            // Specifies the format of the pixel data. Should mean one float for this
            GL_FLOAT,          // Specifies the data type of the pixel data.
            (void*)tmp.data()  // Specifies a pointer to the image data in memory.
        );
#endif
        // specify minify/magnify interpolation method
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        glBindTexture(GL_TEXTURE_2D, 0);
    }

    void upload_child_links() {
        const size_t child_size = size_t(tree->capacity) * tree->N * tree->N * tree->N;
        size_t width, height;
        auto_size_2d(child_size, width, height);

        const size_t pad = width * height - child_size;
        tree->child_.data_holder.resize((child_size + pad) * sizeof(int32_t));
        glUniform1i(glGetUniformLocation(program, "tree_child_dim"), (GLsizei)width);  // y axis align

        glBindTexture(GL_TEXTURE_2D, tex_tree_child);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32I, (GLsizei)width, (GLsizei)height, 0, GL_RED_INTEGER, GL_INT, tree->child_.data<int32_t>());
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        glBindTexture(GL_TEXTURE_2D, 0);
    }

    void upload_tree_spec() {
        glUniform1i(glGetUniformLocation(program, "tree.N"), tree->N);
        glUniform1i(glGetUniformLocation(program, "tree.data_dim"), tree->data_dim);
        glUniform1i(glGetUniformLocation(program, "tree.extra_slot"), tree->extra_slot);
        glUniform1i(glGetUniformLocation(program, "tree.format"), (int)tree->data_format.format);
        glUniform1i(glGetUniformLocation(program, "tree.basis_dim"), tree->data_format.format == DataFormat::RGBA ? 1 : tree->data_format.basis_dim);
        glUniform3f(glGetUniformLocation(program, "tree.center"), tree->offset[0], tree->offset[1], tree->offset[2]);
        glUniform3f(glGetUniformLocation(program, "tree.scale"), tree->scale[0], tree->scale[1], tree->scale[2]);

        // We do not use NDC but [0,1]^3
        glUniform1f(glGetUniformLocation(program, "tree.ndc_width"), -1.f);
        // // TODO: make this configurable
        // glUniform1i(glGetUniformLocation(program, "tree.use_offset"), true);
    }

    void shader_init() {
        program = create_shader_program(GUIDE_MESH_VERT_SHADER_SRC, RT_FRAG_SRC);

        u.cam_transform = glGetUniformLocation(program, "cam.transform");
        u.cam_focal = glGetUniformLocation(program, "cam.focal");
        u.cam_reso = glGetUniformLocation(program, "cam.reso");
        u.cam_K = glGetUniformLocation(program, "cam.K");
        u.cam_w2c = glGetUniformLocation(program, "cam.w2c");
        u.rigid = glGetUniformLocation(program, "rigid");
        u.bigrigid = glGetUniformLocation(program, "bigrigid");
        u.M = glGetUniformLocation(program, "M");
        u.opt_step_size = glGetUniformLocation(program, "opt.step_size");
        u.opt_backgrond_brightness = glGetUniformLocation(program, "opt.background_brightness");
        u.opt_stop_thresh = glGetUniformLocation(program, "opt.stop_thresh");
        u.opt_sigma_thresh = glGetUniformLocation(program, "opt.sigma_thresh");
        u.opt_render_bbox = glGetUniformLocation(program, "opt.render_bbox");
        u.opt_basis_minmax = glGetUniformLocation(program, "opt.basis_minmax");
        u.opt_rot_dirs = glGetUniformLocation(program, "opt.rot_dirs");
        u.opt_t_off_in = glGetUniformLocation(program, "opt.t_off_in");
        u.opt_t_off_out = glGetUniformLocation(program, "opt.t_off_out");
        u.opt_use_offset = glGetUniformLocation(program, "opt.use_offset");
        u.opt_visualize_offset = glGetUniformLocation(program, "opt.visualize_offset");
        u.opt_visualize_unseen = glGetUniformLocation(program, "opt.visualize_unseen");
        u.opt_visualize_intensity = glGetUniformLocation(program, "opt.visualize_intensity");
        u.opt_vis_offset_multiplier = glGetUniformLocation(program, "opt.vis_offset_multiplier");
        u.opt_vis_color_multiplier = glGetUniformLocation(program, "opt.vis_color_multiplier");
        u.opt_vis_color_offset = glGetUniformLocation(program, "opt.vis_color_offset");
        u.opt_apply_sigmoid = glGetUniformLocation(program, "opt.apply_sigmoid");
        u.opt_probe = glGetUniformLocation(program, "opt.probe");
        u.opt_probe_disp_size = glGetUniformLocation(program, "opt.probe_disp_size");
        u.opt_show_template = glGetUniformLocation(program, "opt.show_template");
        u.bigpose_geometry = glGetUniformLocation(program, "opt.bigpose_geometry");
        u.drawing_probe = glGetUniformLocation(program, "drawing_probe");
        u.tree_data_tex = glGetUniformLocation(program, "tree_data_tex");
        u.tree_child_tex = glGetUniformLocation(program, "tree_child_tex");
        u.mesh_depth_tex = glGetUniformLocation(program, "mesh_depth_tex");
        u.mesh_color_tex = glGetUniformLocation(program, "mesh_color_tex");
        glUniform1i(u.tree_child_tex, 0);  // this index should later used in glActiveTexture
        glUniform1i(u.tree_data_tex, 1);   // it means setting this uniform to an int, which will be interpreted as texture location
        glUniform1i(u.mesh_depth_tex, 2);
        glUniform1i(u.mesh_color_tex, 3);
        glUniform1i(glGetUniformLocation(program, "tree_data_dim"), 0);
    }

    void quad_init() {
        glGenBuffers(1, &vbo_quad);       // the buffer to hold the actual vertex data
        glGenVertexArrays(1, &vao_quad);  // the vertex array buffer

        glBindVertexArray(vao_quad);              // tell GL we're modifying this vao
        glBindBuffer(GL_ARRAY_BUFFER, vbo_quad);  // tell GL we're modifying this vbo
        glBufferData(GL_ARRAY_BUFFER, sizeof(quad_verts), (GLvoid*)quad_verts, GL_STATIC_DRAW);

        glVertexAttribPointer(
            0,                  // index of the generic vertex attribute
            3,                  // number of components per generic vertex attribute. Must be 1, 2, 3, 4.
            GL_FLOAT,           // data type of each component in the array
            GL_FALSE,           // whether fixed-point data values should be normalized
            3 * sizeof(float),  // byte offset between consecutive generic vertex attributes. 0 means tightly packed
            (GLvoid*)0          //  offset of the first component of the first generic vertex attribute
        );

        glEnableVertexAttribArray(0);  // defaults to disabled
        glBindVertexArray(0);          // reset to vao number 0
    }

    Camera& camera;
    RenderOptions& options;

    N3Tree* tree;               // the actual volume rendering tree
    Human* human;               // the guide mesh for the volume renderer
    std::vector<Mesh>& meshes;  // meshed to render on screen

    GLuint program = -1;                                        // volume rendering ray casting shader program
    GLuint tex_tree_data = -1, tex_tree_child, tex_tree_extra;  // index of tree data texture location
    GLuint vao_quad;
    GLuint vbo_quad;
    GLint tex_max_size;

    Mesh probe_, wire_;         // special meshes
    int last_wire_depth_ = -1;  // The depth level of the octree wireframe; -1 = not yet generated

    // TODO: use a render buffer for depth buffer
    // TODO: rename tex_mesh_depth to a more consistent name
    GLuint
        fb,                  // frame buffer
        tex_mesh_color,      // mesh pass color buffer
        tex_mesh_depth,      // mesh pass camera t buffer
        tex_mesh_depth_buf;  // mesh pass depth buffer

    std::string shader_fname = "shaders/rt.frag";

    _RenderUniforms u;      // uniform value in the shader
    bool started_ = false;  // whether the program (globally) has been started, can render now
};

VolumeRenderer::VolumeRenderer() : impl_(std::make_unique<Impl>(camera, options, meshes)) {}

VolumeRenderer::~VolumeRenderer() {}

void VolumeRenderer::render() { impl_->render(); }

void VolumeRenderer::set(N3Tree& tree) { impl_->set(tree); }
void VolumeRenderer::set(Human& human) { impl_->set(human); }
Human* VolumeRenderer::get_human() { return impl_->get_human(); }
void VolumeRenderer::clear() { impl_->clear(); }

void VolumeRenderer::resize(int width, int height) {
    impl_->resize(width, height);
}
const char* VolumeRenderer::get_backend() { return "Shader"; }

}  // namespace volrend

#endif
