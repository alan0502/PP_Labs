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
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
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
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
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

    // Split rows among processes
    int rows_per_proc = height / size;
    int start_y = rank * rows_per_proc;
    int end_y = (rank == size - 1) ? height : start_y + rows_per_proc;
    int local_height = end_y - start_y;

    int* local_image = (int*)malloc(width * local_height * sizeof(int));

    #pragma omp parallel for schedule(dynamic, 8)
    for (int j = 0; j < local_height; ++j) {
        double y0 = (j + start_y) * ((upper - lower) / height) + lower;
        for (int i = 0; i < width; ++i) {
            double x0 = i * ((right - left) / width) + left;

            int repeats = 0;
            double x = 0, y = 0;
            double length_squared = 0;
            while (repeats < iters && length_squared < 4) {
                double temp = x * x - y * y + x0;
                y = 2 * x * y + y0;
                x = temp;
                length_squared = x * x + y * y;
                ++repeats;
            }
            local_image[j * width + i] = repeats;
        }
    }

    // Gather results
    int* full_image = NULL;
    if (rank == 0) full_image = (int*)malloc(width * height * sizeof(int));

    int* recvcounts = (int*)malloc(size * sizeof(int));
    int* displs = (int*)malloc(size * sizeof(int));
    for (int p = 0; p < size; ++p) {
        int start = p * rows_per_proc;
        int end = (p == size - 1) ? height : start + rows_per_proc;
        recvcounts[p] = (end - start) * width;
        displs[p] = start * width;
    }

    nvtxRangePop(); nvtxRangePush("Comm");
    MPI_Gatherv(local_image, local_height * width, MPI_INT,
                full_image, recvcounts, displs, MPI_INT,
                0, MPI_COMM_WORLD);

    
    nvtxRangePop(); nvtxRangePush("IO");

    if (rank == 0) {
        write_png(filename, iters, width, height, full_image);
        free(full_image);
    }
    nvtxRangePop(); nvtxRangePush("CPU");
    free(local_image);
    free(recvcounts);
    free(displs);

    MPI_Finalize();
    nvtxRangePop();
    return 0;
}