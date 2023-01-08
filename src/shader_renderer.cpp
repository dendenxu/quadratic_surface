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

    struct {
        GLint shape;
        GLint eps;
        GLint samples;
        GLint box_size;
    } qua;
};

}  // namespace

struct VolumeRenderer::Impl {
    Impl(Camera& camera, Quadric& quadric)
        : camera(camera), quadric(quadric) {
        start();
    }

    ~Impl() {
        glDeleteProgram(program);
    }

    void start() {
        if (started) return;
        quad_init();
        shader_init();
        started = true;
    }

    void render() {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);  // ? what is bound, 0 ?
        if (!started) return;
        camera.update();
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glUseProgram(program);  // using tree shader

        glUniformMatrix4x3fv(u.cam.c2w, 1, GL_FALSE, glm::value_ptr(camera.c2w));
        glUniformMatrix4fv(u.cam.w2c, 1, GL_FALSE, glm::value_ptr(camera.w2c));      // camera w2c to GPU
        glUniformMatrix4fv(u.cam.K, 1, GL_FALSE, glm::value_ptr(camera.K));          // camera intrinsics
        glUniformMatrix4fv(u.cam.inv_K, 1, GL_FALSE, glm::value_ptr(camera.inv_K));  // camera inverse intrinsics
        glUniform3fv(u.cam.center, 1, glm::value_ptr(camera.center));                // camera center
        glUniform2f(u.cam.focal, camera.fx, camera.fy);
        glUniform2f(u.cam.reso, (float)camera.width, (float)camera.height);

        glUniformMatrix4fv(u.qua.shape, 1, GL_FALSE, glm::value_ptr(quadric.Q()));  // loading camera inv_K
        glUniform1f(u.qua.eps, quadric.eps);
        glUniform1f(u.qua.box_size, quadric.box_size);
        glUniform1i(u.qua.samples, quadric.samples);

        glBindVertexArray(vao_quad);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, (GLsizei)4);
        glBindVertexArray(0);
    }

    void resize(const int width, const int height) {
        if (camera.width == width && camera.height == height) return;
        if (width <= 0 || height <= 0) return;
        camera.width = width;
        camera.height = height;
        glViewport(0, 0, width, height);
    }

   private:
    void shader_init() {
        program = create_shader_program(GUIDE_MESH_VERT_SHADER_SRC, RT_FRAG_SRC);
        u.cam.c2w = glGetUniformLocation(program, "cam.c2w");
        u.cam.w2c = glGetUniformLocation(program, "cam.w2c");
        u.cam.K = glGetUniformLocation(program, "cam.K");
        u.cam.inv_K = glGetUniformLocation(program, "cam.inv_K");
        u.cam.center = glGetUniformLocation(program, "cam.center");
        u.cam.focal = glGetUniformLocation(program, "cam.focal");
        u.cam.reso = glGetUniformLocation(program, "cam.reso");

        u.qua.shape = glGetUniformLocation(program, "qua.shape");
        u.qua.eps = glGetUniformLocation(program, "qua.eps");
        u.qua.box_size = glGetUniformLocation(program, "qua.box_size");
        u.qua.samples = glGetUniformLocation(program, "qua.samples");
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
    Quadric& quadric;
    GLuint program = -1;  // volume rendering ray casting shader program
    GLuint vao_quad;
    GLuint vbo_quad;

    std::string shader_fname = "shaders/rt.frag";

    RenderUniforms u;      // uniform value in the shader
    bool started = false;  // whether the program (globally) has been started, can render now
};

VolumeRenderer::VolumeRenderer() : impl(std::make_unique<Impl>(camera, quadric)) {}
VolumeRenderer::~VolumeRenderer() {}
void VolumeRenderer::render() { impl->render(); }
void VolumeRenderer::resize(int width, int height) { impl->resize(width, height); }
const char* VolumeRenderer::get_backend() { return "Shader"; }

}  // namespace volrend

#endif
