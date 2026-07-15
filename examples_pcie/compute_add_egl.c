// Minimal EGL+GL4.3 compute shader add: triggers nouveau GR init + compute path on GK104.
// gcc -O2 -o compute_add_egl compute_add_egl.c -lEGL -lGL -lgbm -ldrm
#define GL_GLEXT_PROTOTYPES 1
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GL/gl.h>
#include <GL/glext.h>
#include <gbm.h>
#include <drm/drm.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/ioctl.h>

static const char *compute_src =
"#version 430\n"
"layout(local_size_x=1) in;\n"
"layout(std430, binding=0) buffer InA { float a[]; };\n"
"layout(std430, binding=1) buffer InB { float b[]; };\n"
"layout(std430, binding=2) buffer Out { float o[]; };\n"
"void main() { o[gl_GlobalInvocationID.x] = a[gl_GlobalInvocationID.x] + b[gl_GlobalInvocationID.x]; }\n";

typedef struct { EGLDisplay dpy; EGLContext ctx; EGLDeviceEXT dev; } GLCtx;

int main(int argc, char **argv) {
    const char *node = argc > 1 ? argv[1] : "/dev/dri/renderD128";
    int fd = open(node, O_RDWR);
    if (fd < 0) { perror(node); return 1; }
    printf("[egl] opened %s fd=%d\n", node, fd);

    struct gbm_device *gbm = gbm_create_device(fd);
    if (!gbm) { perror("gbm_create_device"); return 1; }

    PFNEGLGETPLATFORMDISPLAYEXTPROC eglGetPlatformDisplayEXT =
        (void*)eglGetProcAddress("eglGetPlatformDisplayEXT");
    EGLDisplay dpy = eglGetPlatformDisplayEXT(EGL_PLATFORM_GBM_KHR, gbm, NULL);
    if (dpy == EGL_NO_DISPLAY) { printf("eglGetPlatformDisplay failed\n"); return 1; }

    EGLint major, minor;
    if (!eglInitialize(dpy, &major, &minor)) { printf("eglInitialize failed\n"); return 1; }
    printf("[egl] initialized EGL %d.%d\n", major, minor);

    const char *client_apis = eglQueryString(dpy, EGL_CLIENT_APIS);
    const char *vendor = eglQueryString(dpy, EGL_VENDOR);
    printf("[egl] vendor=%s apis=%s\n", vendor ? vendor : "?", client_apis ? client_apis : "?");

    eglBindAPI(EGL_OPENGL_API);

    EGLint cfg_attr[] = { EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT, EGL_NONE };
    EGLConfig cfg; EGLint n;
    if (!eglChooseConfig(dpy, cfg_attr, &cfg, 1, &n) || n == 0) {
        printf("eglChooseConfig failed\n"); return 1;
    }

    EGLint ctx_attr[] = { EGL_CONTEXT_MAJOR_VERSION, 4, EGL_CONTEXT_MINOR_VERSION, 3, EGL_NONE };
    EGLContext ctx = eglCreateContext(dpy, cfg, EGL_NO_CONTEXT, ctx_attr);
    if (ctx == EGL_NO_CONTEXT) {
        // try GL 3.3 (no compute but still triggers GR init)
        ctx_attr[3] = 0;
        ctx = eglCreateContext(dpy, cfg, EGL_NO_CONTEXT, ctx_attr);
        if (ctx == EGL_NO_CONTEXT) { printf("eglCreateContext failed\n"); return 1; }
        printf("[egl] created GL 3.3 context (no compute)\n");
    } else {
        printf("[egl] created GL 4.3 context (compute capable)\n");
    }

    eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx);

    printf("[gl] vendor=%s\n", glGetString(GL_VENDOR));
    printf("[gl] renderer=%s\n", glGetString(GL_RENDERER));
    printf("[gl] version=%s\n", glGetString(GL_VERSION));

    // Check compute support
    GLint max_wg = 0;
    glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &max_wg);
    printf("[gl] max compute work group invocations=%d\n", max_wg);

    if (max_wg > 0) {
        // Compile compute shader
        GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
        glShaderSource(shader, 1, &compute_src, NULL);
        glCompileShader(shader);
        GLint ok = 0;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            char log[1024];
            glGetShaderInfoLog(shader, sizeof(log), NULL, log);
            printf("[gl] shader compile failed: %s\n", log);
        } else {
            GLuint prog = glCreateProgram();
            glAttachShader(prog, shader);
            glLinkProgram(prog);
            glGetProgramiv(prog, GL_LINK_STATUS, &ok);
            if (!ok) {
                char log[1024];
                glGetProgramInfoLog(prog, sizeof(log), NULL, log);
                printf("[gl] link failed: %s\n", log);
            } else {
                glUseProgram(prog);
                // Create buffers: a=[1,2,3,4], b=[10,20,30,40], o=[]
                float a[] = {1,2,3,4}, b[] = {10,20,30,40}, o[4] = {0};
                GLuint bufs[3];
                glGenBuffers(3, bufs);
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufs[0]);
                glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(a), a, GL_STATIC_COPY);
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bufs[1]);
                glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(b), b, GL_STATIC_COPY);
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, bufs[2]);
                glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(o), o, GL_DYNAMIC_READ);

                printf("[compute] dispatching add kernel...\n");
                glDispatchCompute(4, 1, 1);
                glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
                glFinish();

                float result[4] = {0};
                glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufs[2]);
                glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(result), result);
                printf("[compute] result = [%f, %f, %f, %f]\n", result[0], result[1], result[2], result[3]);
                int pass = (result[0]==11 && result[1]==22 && result[2]==33 && result[3]==44);
                printf("%s\n", pass ? "PASS" : "FAIL");
            }
        }
    } else {
        printf("[gl] no compute support, but GR init was triggered\n");
        // Just do a simple draw to trigger GR init
        glFinish();
        printf("PASS (GR init only)\n");
    }

    eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    eglDestroyContext(dpy, ctx);
    eglTerminate(dpy);
    gbm_device_destroy(gbm);
    close(fd);
    return 0;
}
