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
#include <mpi.h>
#include <omp.h>
#include <nvtx3/nvToolsExt.h>

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
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_set_compression_level(png_ptr, 1);
    png_write_info(png_ptr, info_ptr);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = (p % 16) * 16;
                } else {
                    color[0] = (p % 16) * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

/* ============================================================== */
/* Main                                                           */
/* ============================================================== */
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

    /* 本 rank 需計算的列數（interleaved：rank, rank+size, rank+2*size, ...） */
    int local_height = 0;
    for (int y = rank; y < height; y += size) local_height++;

    /* 本 rank 的局部影像緩衝區（依 local row 連續擺放） */
    int* local_image = (int*)malloc((size_t)width * (size_t)local_height * sizeof(int));
    assert(local_image);

    /* 並行區：每個 rank 計算自己 interleaved 的行 */
    #pragma omp parallel for schedule(dynamic, 8)
    for (int j = 0; j < local_height; ++j) {
        int global_y = rank + j * size;
        double y0 = global_y * ((upper - lower) / (double)height) + lower;
        for (int i = 0; i < width; ++i) {
            double x0 = i * ((right - left) / (double)width) + left;
            int repeats = 0;
            double x = 0, y = 0;
            double length_squared = 0;
            while (repeats < iters && length_squared < 4.0) {
                double temp = x * x - y * y + x0;
                y = 2.0 * x * y + y0;
                x = temp;
                length_squared = x * x + y * y;
                ++repeats;
            }
            local_image[j * width + i] = repeats;
        }
    }

    /* === Comm: 統一為第二版寫法（收集到 gathered，再重建 full_image） === */
    nvtxRangePop(); nvtxRangePush("Comm");

    int local_count = width * local_height;
    int *recvcounts = NULL, *displs = NULL;
    if (rank == 0) {
        recvcounts = (int*)malloc(size * sizeof(int));
        displs     = (int*)malloc(size * sizeof(int));
        for (int p = 0; p < size; ++p) {
            int cnt = 0;
            for (int y = p; y < height; y += size) cnt++;
            recvcounts[p] = cnt * width;
        }
        displs[0] = 0;
        for (int p = 1; p < size; ++p)
            displs[p] = displs[p - 1] + recvcounts[p - 1];
    }

    int* gathered = NULL;
    if (rank == 0)
        gathered = (int*)malloc((size_t)width * (size_t)height * sizeof(int));

    MPI_Gatherv(local_image, local_count, MPI_INT,
                gathered, recvcounts, displs, MPI_INT,
                0, MPI_COMM_WORLD);

    /* 重建成連續列順序的 full_image（可直接輸出 PNG） */
    int* full_image = NULL;
    if (rank == 0) {
        full_image = (int*)malloc((size_t)width * (size_t)height * sizeof(int));
        int offset = 0;
        for (int p = 0; p < size; ++p) {
            int cnt_rows = recvcounts[p] / width;
            for (int j = 0; j < cnt_rows; ++j) {
                int global_y = p + j * size;
                if (global_y < height) {
                    memcpy(&full_image[global_y * width],
                           &gathered[offset + j * width],
                           (size_t)width * sizeof(int));
                }
            }
            offset += recvcounts[p];
        }
        free(gathered);
        free(recvcounts);
        free(displs);
    }

    /* === IO === */
    nvtxRangePop(); nvtxRangePush("IO");
    if (rank == 0) {
        write_png(filename, iters, width, height, full_image);
        free(full_image);
    }

    /* === Finalize === */
    nvtxRangePop(); nvtxRangePush("CPU");
    free(local_image);
    MPI_Finalize();
    nvtxRangePop();
    return 0;
}
