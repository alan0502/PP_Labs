#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <unistd.h>
#include <immintrin.h>
#include <cpuid.h>
#include <nvtx3/nvToolsExt.h>

typedef struct {
    int tid;
    int num_threads;
    int iters;
    double left, right, lower, upper;
    int width, height;
    int *image;
    double x_scale, y_scale;
} ThreadData;

/* ==============================================================
   SSE2 kernel: 128-bit / 2 doubles
   ============================================================== */
static inline void mandelbrot_row_sse2(
    double left, double x_scale, double y0,
    int iters, int width, int *row_out)
{
    const __m128d v_left = _mm_set1_pd(left);
    const __m128d v_dx   = _mm_set1_pd(x_scale);
    const __m128d v_y0   = _mm_set1_pd(y0);
    const __m128d v_four = _mm_set1_pd(4.0);
    const __m128d v_one  = _mm_set1_pd(1.0);
    const __m128d v_two  = _mm_set1_pd(2.0);

    int i = 0;
    for (; i + 1 < width; i += 2) {
        __m128d idx = _mm_set_pd(i + 1, i);
        __m128d cx  = idx * v_dx + v_left;
        __m128d cy  = v_y0;

        __m128d x   = _mm_setzero_pd();
        __m128d y   = _mm_setzero_pd();
        __m128d itv = _mm_setzero_pd();

        for (int k = 0; k < iters; ++k) {
            __m128d x2 = x*x;
            __m128d y2 = y*y;
            __m128d r2 = x2 + y2;

            __m128d mask = r2 < v_four;
            int mask_bits = _mm_movemask_pd(mask);
            if (mask_bits == 0) break;

            __m128d xy = x * y;
            x = x2 - y2 + cx;
            y = v_two * xy + cy;

            itv += _mm_and_pd(mask, v_one);
        }

        double tmp[2];
        _mm_storeu_pd(tmp, itv);
        row_out[i]   = (int)(tmp[0] + 0.5);
        row_out[i+1] = (int)(tmp[1] + 0.5);
    }

    // scalar tail
    for (; i < width; ++i) {
        double x0 = i * x_scale + left;
        double x = 0.0, y = 0.0;
        int repeats = 0;
        while (repeats < iters && (x*x + y*y) <= 4.0) {
            double xt = x*x - y*y + x0;
            y = 2.0*x*y + y0;
            x = xt;
            ++repeats;
        }
        row_out[i] = repeats;
    }
}

/* ==============================================================
   PNG writer
   ============================================================== */
void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);

    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_compression_level(png_ptr, 0);
    png_write_info(png_ptr, info_ptr);

    size_t row_size = 3 * width;
    png_bytep image_data = (png_bytep)malloc(row_size * height);
    png_bytep* rows = (png_bytep*)malloc(sizeof(png_bytep) * height);

    for (int y = 0; y < height; ++y) {
        png_bytep row = image_data + y * row_size;
        rows[y] = row;
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = (p % 16) * 16;
                } else {
                    color[0] = (p % 16) * 16;
                    color[1] = color[2] = 0;
                }
            } else {
                color[0] = color[1] = color[2] = 0;
            }
        }
    }

    png_write_image(png_ptr, rows);
    png_write_end(png_ptr, NULL);
    free(rows);
    free(image_data);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

/* ==============================================================
   pthread worker
   ============================================================== */
void* mandelbrot_worker(void* arg) {
    ThreadData* t = (ThreadData*)arg;
    int tid = t->tid;
    int num_threads = t->num_threads;
    double left = t->left, lower = t->lower;
    double x_scale = t->x_scale, y_scale = t->y_scale;
    int width = t->width, height = t->height, iters = t->iters;
    int* image = t->image;

    for (int j = tid; j < height; j += num_threads) {
        double y0 = j * y_scale + lower;
        int* row_ptr = &image[j * width];
        mandelbrot_row_sse2(left, x_scale, y0, iters, width, row_ptr);
    }

    pthread_exit(NULL);
}

/* ==============================================================
   main
   ============================================================== */
int main(int argc, char** argv) {
    nvtxRangePush("CPU");
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    int num_cpus = CPU_COUNT(&cpu_set);
    printf("%d cpus available\n", num_cpus);

    assert(argc == 9);
    const char* filename = argv[1];
    int iters = strtol(argv[2], 0, 10);
    double left = strtod(argv[3], 0);
    double right = strtod(argv[4], 0);
    double lower = strtod(argv[5], 0);
    double upper = strtod(argv[6], 0);
    int width = strtol(argv[7], 0, 10);
    int height = strtol(argv[8], 0, 10);

    double x_scale = (right - left) / width;
    double y_scale = (upper - lower) / height;

    int num_threads = num_cpus;
    int* image;
    posix_memalign((void**)&image, 64, width * height * sizeof(int));

    pthread_t* threads = (pthread_t*)malloc(num_threads * sizeof(pthread_t));
    ThreadData* tdata = (ThreadData*)malloc(num_threads * sizeof(ThreadData));

    for (int t = 0; t < num_threads; ++t) {
        tdata[t] = (ThreadData){t, num_threads, iters, left, right, lower, upper,
                                width, height, image, x_scale, y_scale};
        pthread_create(&threads[t], NULL, mandelbrot_worker, &tdata[t]);
    }

    for (int t = 0; t < num_threads; ++t)
        pthread_join(threads[t], NULL);

    nvtxRangePop(); nvtxRangePush("IO");
    write_png(filename, iters, width, height, image);
    nvtxRangePop(); nvtxRangePush("CPU");

    free(image);
    free(threads);
    free(tdata);
    nvtxRangePop();
    return 0;
}
