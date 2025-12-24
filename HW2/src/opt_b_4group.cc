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
#include <mpi.h>
#include <omp.h>
#include <nvtx3/nvToolsExt.h>
#include <immintrin.h>
#include <cpuid.h>

// SSE2 kernel: 128-bit / 2 doubles
// ILP = 4 groups (8 pixels per outer-iter)
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
    for (; i + 7 < width; i += 8) {
        // 4 groups of indices (8 pixels)
        __m128d idx1 = _mm_set_pd(i + 1, i);
        __m128d idx2 = _mm_set_pd(i + 3, i + 2);
        __m128d idx3 = _mm_set_pd(i + 5, i + 4);
        __m128d idx4 = _mm_set_pd(i + 7, i + 6);

        __m128d cx1 = idx1 * v_dx + v_left;
        __m128d cx2 = idx2 * v_dx + v_left;
        __m128d cx3 = idx3 * v_dx + v_left;
        __m128d cx4 = idx4 * v_dx + v_left;

        __m128d cy1 = v_y0, cy2 = v_y0, cy3 = v_y0, cy4 = v_y0;

        __m128d x1 = _mm_setzero_pd(), y1 = _mm_setzero_pd(), itv1 = _mm_setzero_pd();
        __m128d x2 = _mm_setzero_pd(), y2 = _mm_setzero_pd(), itv2 = _mm_setzero_pd();
        __m128d x3 = _mm_setzero_pd(), y3 = _mm_setzero_pd(), itv3 = _mm_setzero_pd();
        __m128d x4 = _mm_setzero_pd(), y4 = _mm_setzero_pd(), itv4 = _mm_setzero_pd();

        int maskb1 = 3, maskb2 = 3, maskb3 = 3, maskb4 = 3;

        __m128d mask1 = _mm_set1_pd(-1LL);
        __m128d mask2 = _mm_set1_pd(-1LL);
        __m128d mask3 = _mm_set1_pd(-1LL);
        __m128d mask4 = _mm_set1_pd(-1LL);

        for (int k = 0; k < iters; ++k) {
            // Group 1
            __m128d x2_1 = x1 * x1;
            __m128d y2_1 = y1 * y1;
            __m128d r2_1 = x2_1 + y2_1;
            mask1 = r2_1 < v_four;
            if (maskb1 != 0) {
                __m128d xy1 = x1 * y1;
                x1 = x2_1 - y2_1 + cx1;
                y1 = v_two * xy1 + cy1;
                itv1 += _mm_and_pd(mask1, v_one);
            }

            // Group 2
            __m128d x2_2 = x2 * x2;
            __m128d y2_2 = y2 * y2;
            __m128d r2_2 = x2_2 + y2_2;
            mask2 = r2_2 < v_four;
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
            mask3 = r2_3 < v_four;
            if (maskb3 != 0) {
                __m128d xy3 = x3 * y3;
                x3 = x2_3 - y2_3 + cx3;
                y3 = v_two * xy3 + cy3;
                itv3 += _mm_and_pd(mask3, v_one);
            }

            // Group 4
            __m128d x2_4 = x4 * x4;
            __m128d y2_4 = y4 * y4;
            __m128d r2_4 = x2_4 + y2_4;
            mask4 = r2_4 < v_four;
            if (maskb4 != 0) {
                __m128d xy4 = x4 * y4;
                x4 = x2_4 - y2_4 + cx4;
                y4 = v_two * xy4 + cy4;
                itv4 += _mm_and_pd(mask4, v_one);
            }

            maskb1 = _mm_movemask_pd(mask1);
            maskb2 = _mm_movemask_pd(mask2);
            maskb3 = _mm_movemask_pd(mask3);
            maskb4 = _mm_movemask_pd(mask4);

            // 全部都超界即可提早結束
            if ((maskb1 | maskb2 | maskb3 | maskb4) == 0)
                break;
        }

        double tmp1[2], tmp2[2], tmp3[2], tmp4[2];
        _mm_storeu_pd(tmp1, itv1);
        _mm_storeu_pd(tmp2, itv2);
        _mm_storeu_pd(tmp3, itv3);
        _mm_storeu_pd(tmp4, itv4);

        row_out[i    ] = (int)(tmp1[0] + 0.5);
        row_out[i + 1] = (int)(tmp1[1] + 0.5);
        row_out[i + 2] = (int)(tmp2[0] + 0.5);
        row_out[i + 3] = (int)(tmp2[1] + 0.5);
        row_out[i + 4] = (int)(tmp3[0] + 0.5);
        row_out[i + 5] = (int)(tmp3[1] + 0.5);
        row_out[i + 6] = (int)(tmp4[0] + 0.5);
        row_out[i + 7] = (int)(tmp4[1] + 0.5);
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
    png_bytep row;
    # pragma omp parallel for linear(row:row_size) schedule(static)
    for (int y = 0; y < height; ++y) {
        row = image_data + y * row_size;
        rows[y] = row;
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;

            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = (p & 15) * 16;
                } else {
                    color[0] = (p & 15) * 16;
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
   Main
   ============================================================== */
int main(int argc, char** argv) {
    nvtxRangePush("CPU");
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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

    // Interleaved row distribution
    int local_height = 0;
    for (int y = rank; y < height; y += size) local_height++;

    int* local_image = (int*)malloc(width * local_height * sizeof(int));
    assert(local_image);

    #pragma omp parallel for schedule(dynamic, 8)
    for (int j = 0; j < local_height; ++j) {
        int global_y = rank + j * size;
        double y0 = global_y * y_scale + lower;
        int* row_ptr = &local_image[j * width];
        mandelbrot_row_sse2(left, x_scale, y0, iters, width, row_ptr);
    }

    // === Gather using MPI_Gatherv ===
    nvtxRangePop(); nvtxRangePush("Comm");
    int local_count = width * local_height;
    int *recvcounts = NULL, *displs = NULL;

    int total_count = 0;
    if (rank == 0) {
        recvcounts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));

        for (int p = 0; p < size; ++p) {
            int cnt = 0;
            for (int y = p; y < height; y += size) cnt++;
            recvcounts[p] = cnt * width;
        }

        displs[0] = 0;
        for (int p = 1; p < size; ++p)
            displs[p] = displs[p - 1] + recvcounts[p - 1];

        total_count = displs[size - 1] + recvcounts[size - 1];
    }

    int* gathered = NULL;
    if (rank == 0)
        gathered = (int*)malloc(total_count * sizeof(int));

    MPI_Gatherv(local_image, local_count, MPI_INT,
                gathered, recvcounts, displs, MPI_INT,
                0, MPI_COMM_WORLD);

    // === Reassemble final image ===
    int* full_image = NULL;
    if (rank == 0) {
        full_image = (int*)malloc(width * height * sizeof(int));
        int offset = 0;
        for (int p = 0; p < size; ++p) {
            int cnt = recvcounts[p] / width;
            for (int j = 0; j < cnt; ++j) {
                int global_y = p + j * size;
                if (global_y < height) {
                    memcpy(&full_image[global_y * width],
                           &gathered[offset + j * width],
                           width * sizeof(int));
                }
            }
            offset += recvcounts[p];
        }
        free(gathered);
        free(recvcounts);
        free(displs);
    }

    // === Write PNG ===
    nvtxRangePop(); nvtxRangePush("IO");
    if (rank == 0) {
        write_png(filename, iters, width, height, full_image);
        free(full_image);
    }

    nvtxRangePop(); nvtxRangePush("CPU");
    free(local_image);
    MPI_Finalize();
    nvtxRangePop();
    return 0;
}
