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

// Computer vision style camera
struct Camera {
    mat4x3 c2w;
    mat4x4 w2c;
    mat4x4 K;
    mat4x4 inv_K;
    vec2 reso;
    vec2 focal;
    vec3 center;
};
uniform Camera cam;
layout(location=0) in vec3 a_pos; // location 0

// pass through shader
void main()
{
    gl_Position = vec4(a_pos.x, a_pos.y, a_pos.z, 1.0);
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

struct RenderUniforms {
    struct {
        GLint c2w;
        GLint w2c;
        GLint K;
        GLint inv_K;
        GLint focal;
        GLint reso;
        GLint center;
    } cam;
};

}  // namespace

struct VolumeRenderer::Impl {
    Impl(Camera& camera, RenderOptions& options, std::vector<Mesh>& meshes,
         int max_tries = 4)
        : camera(camera), options(options), meshes(meshes) {
        start();
    }

    ~Impl() {
        glDeleteProgram(program);
    }

    void start() {
        if (started_) return;
        quad_init();
        shader_init();
        started_ = true;
    }

    void render() {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);  // ? what is bound, 0 ?
        if (!started_) return;
        camera.update();
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glUseProgram(program);  // using tree shader

        // FIXME reduce uniform transfers?
        glUniformMatrix4x3fv(u.cam.c2w, 1, GL_FALSE, glm::value_ptr(camera.c2w));
        glUniformMatrix4fv(u.cam.w2c, 1, GL_FALSE, glm::value_ptr(camera.w2c));        // loading camera w2c to GPU
        glUniformMatrix4fv(u.cam.K, 1, GL_FALSE, glm::value_ptr(camera.K));            // loading camera K
        glUniformMatrix4fv(u.cam.inv_K, 1, GL_FALSE, glm::value_ptr(camera.inv_K));    // loading camera inv_K
        glUniformMatrix4fv(u.cam.center, 1, GL_FALSE, glm::value_ptr(camera.center));  // loading camera inv_K

        glUniform2f(u.cam.focal, camera.fx, camera.fy);
        glUniform2f(u.cam.reso, (float)camera.width, (float)camera.height);
        glBindVertexArray(vao_quad);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, (GLsizei)4);
        glBindVertexArray(0);
    }

    void set(N3Tree& tree) {}

    void set(Human& human) {}

    void maybe_gen_wire(int depth) {}

    void clear() { this->tree = nullptr; }

    void resize(const int width, const int height) {
        if (camera.width == width && camera.height == height) return;
        if (width <= 0 || height <= 0) return;
        camera.width = width;
        camera.height = height;
        glViewport(0, 0, width, height);
    }

    Human* get_human() { return human; }

   private:
    void auto_size_2d(size_t size, size_t& width, size_t& height, int base_dim = 1) {}

    void upload_data() {}

    void upload_child_links() {}

    void upload_tree_spec() {}

    void shader_init() {
        program = create_shader_program(GUIDE_MESH_VERT_SHADER_SRC, RT_FRAG_SRC);
        u.cam.c2w = glGetUniformLocation(program, "cam.c2w");
        u.cam.w2c = glGetUniformLocation(program, "cam.w2c");
        u.cam.K = glGetUniformLocation(program, "cam.K");
        u.cam.inv_K = glGetUniformLocation(program, "cam.inv_K");
        u.cam.center = glGetUniformLocation(program, "cam.center");
        u.cam.focal = glGetUniformLocation(program, "cam.focal");
        u.cam.reso = glGetUniformLocation(program, "cam.reso");
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

    RenderUniforms u;       // uniform value in the shader
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
