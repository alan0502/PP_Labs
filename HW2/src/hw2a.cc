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

//SSE2 kernel: 128-bit / 2 doubles
static inline void mandelbrot_row_sse2(
    double left, double x_scale, double y0,
    int iters, int width, int *row_out)
{
    const __m128d v_left = _mm_set1_pd(left); // The left boundary
    const __m128d v_dx   = _mm_set1_pd(x_scale); // The delta x per pixel
    const __m128d v_y0   = _mm_set1_pd(y0); // The y0 for this row，because y0 is constant in a row
    const __m128d v_four = _mm_set1_pd(4.0); // 4.0
    const __m128d v_one  = _mm_set1_pd(1.0); // 1.0
    const __m128d v_two  = _mm_set1_pd(2.0); // 2.0

    int i = 0;
    for (; i + 5 < width; i += 6) {
        // 3 groups of indices (6 pixels)
        __m128d idx1 = _mm_set_pd(i + 1, i);
        __m128d idx2 = _mm_set_pd(i + 3, i + 2);
        __m128d idx3 = _mm_set_pd(i + 5, i + 4);

        // Compute the initial complex numbers c = x + iy
        __m128d cx1 = idx1 * v_dx + v_left;
        __m128d cx2 = idx2 * v_dx + v_left;
        __m128d cx3 = idx3 * v_dx + v_left;
        __m128d cy1 = v_y0, cy2 = v_y0, cy3 = v_y0;

        __m128d x1 = _mm_setzero_pd(), y1 = _mm_setzero_pd(), itv1 = _mm_setzero_pd();
        __m128d x2 = _mm_setzero_pd(), y2 = _mm_setzero_pd(), itv2 = _mm_setzero_pd();
        __m128d x3 = _mm_setzero_pd(), y3 = _mm_setzero_pd(), itv3 = _mm_setzero_pd();

        int maskb1 = 3, maskb2 = 3, maskb3 = 3; // initially all active (11 in binary)

        __m128d mask1 = _mm_set1_pd(-1LL);
        __m128d mask2 = _mm_set1_pd(-1LL);
        __m128d mask3 = _mm_set1_pd(-1LL);

        for (int k = 0; k < iters; ++k) {
            // Group 1
            __m128d x2_1 = x1 * x1;
            __m128d y2_1 = y1 * y1;
            __m128d r2_1 = x2_1 + y2_1;

            // Considering the mask into left and right parts, if the value < 4.0, then set the sub-mask to all 1s, else all 0s
            // For example, if r2_1 = [3.0, 5.0], then mask1 = [all 1s, all 0s]
            mask1 = r2_1 < v_four;

            // If mask1 is not zero in last iteration, continue the iteration
            // If maskb1 != 0, it means at least one of the two pixels in this group is still active
            if (maskb1 != 0) {
                __m128d xy1 = x1 * y1;
                x1 = x2_1 - y2_1 + cx1;
                y1 = v_two * xy1 + cy1;
                itv1 += _mm_and_pd(mask1, v_one); // Increment the iteration count only for active pixels, [+1 or +0, +1 or +0]
                // If some pixels change from active to inactive in this iteration, they will not be counted in next iterations
            }

            // Group 2
            __m128d x2_2 = x2 * x2;
            __m128d y2_2 = y2 * y2;
            __m128d r2_2 = x2_2 + y2_2;

            // If the value < 4.0, then set mask2 to all 1s, else all 0s
            mask2 = r2_2 < v_four;

            // If mask2 is not zero in last iteration, continue the iteration
            if (maskb2 != 0) {
                __m128d xy2 = x2 * y2;
                x2 = x2_2 - y2_2 + cx2;
                y2 = v_two * xy2 + cy2;
                itv2 += _mm_and_pd(mask2, v_one);
            }

            // Group 3
            __m128d x2_3 = x3 * x3;
            __m128d y2_3 = y3 * y3;
            __m128d r2_3 = x2_3 + y2_3;

            // If the value < 4.0, then set mask3 to all 1s, else all 0s
            mask3 = r2_3 < v_four;

            // If mask3 is not zero in last iteration, continue the iteration
            if (maskb3 != 0) {
                __m128d xy3 = x3 * y3;
                x3 = x2_3 - y2_3 + cx3;
                y3 = v_two * xy3 + cy3;
                //itv3 = _mm_add_pd(itv3, _mm_and_pd(mask3, v_one));
                itv3 += _mm_and_pd(mask3, v_one);
            }

            // Determine the new masks for next iteration, transform from __m128d to int
            maskb1 = _mm_movemask_pd(mask1);
            maskb2 = _mm_movemask_pd(mask2);
            maskb3 = _mm_movemask_pd(mask3);

            // If all masks are zero, break the loop
            if ((maskb1 | maskb2 | maskb3) == 0)
                break;
        }

        // Store the iteration counts back to memory
        double tmp1[2], tmp2[2], tmp3[2];
        _mm_storeu_pd(tmp1, itv1);
        _mm_storeu_pd(tmp2, itv2);
        _mm_storeu_pd(tmp3, itv3);

        // Store 6 pixels
        row_out[i]   = (int)(tmp1[0] + 0.5);
        row_out[i+1] = (int)(tmp1[1] + 0.5);
        row_out[i+2] = (int)(tmp2[0] + 0.5);
        row_out[i+3] = (int)(tmp2[1] + 0.5);
        row_out[i+4] = (int)(tmp3[0] + 0.5);
        row_out[i+5] = (int)(tmp3[1] + 0.5);
    }

    // scalar tail
    // If the width is not a multiple of 6, handle the remaining pixels
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

void* mandelbrot_worker(void* arg) {
    ThreadData* t = (ThreadData*)arg;
    int tid = t->tid;
    int num_threads = t->num_threads;
    double left = t->left, lower = t->lower;
    double x_scale = t->x_scale, y_scale = t->y_scale;
    int width = t->width, height = t->height, iters = t->iters;
    int* image = t->image;

    // Interleaved row assignment
    // For example, thread 0 does rows 0, num_threads, 2*num_threads
    // thread 1 does rows 1, num_threads+1, 2*num_threads+1, etc.
    // The critical computation is in mandelbrot_row_sse2()
    for (int j = tid; j < height; j += num_threads) {
        double y0 = j * y_scale + lower;
        int* row_ptr = &image[j * width];
        mandelbrot_row_sse2(left, x_scale, y0, iters, width, row_ptr);
    }

    pthread_exit(NULL);
}

int main(int argc, char** argv) {
    nvtxRangePush("CPU");
    // Detect number of CPUs (threads)
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    int num_cpus = CPU_COUNT(&cpu_set);
    printf("%d cpus available\n", num_cpus);

    // Parse arguments
    assert(argc == 9);
    const char* filename = argv[1];
    int iters = strtol(argv[2], 0, 10);
    double left = strtod(argv[3], 0);
    double right = strtod(argv[4], 0);
    double lower = strtod(argv[5], 0);
    double upper = strtod(argv[6], 0);
    int width = strtol(argv[7], 0, 10);
    int height = strtol(argv[8], 0, 10);

    // Compute the length per pixel
    double x_scale = (right - left) / width;
    double y_scale = (upper - lower) / height;

    int num_threads = num_cpus;
    int* image;
    posix_memalign((void**)&image, 64, width * height * sizeof(int));

    pthread_t* threads = (pthread_t*)malloc(num_threads * sizeof(pthread_t));
    ThreadData* tdata = (ThreadData*)malloc(num_threads * sizeof(ThreadData));

    // Create threads
    for (int t = 0; t < num_threads; ++t) {
        tdata[t] = (ThreadData){t, num_threads, iters, left, right, lower, upper,
                                width, height, image, x_scale, y_scale};
        pthread_create(&threads[t], NULL, mandelbrot_worker, &tdata[t]);
    }

    // Join threads
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