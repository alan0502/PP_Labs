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
#include <pthread.h>
#include <nvtx3/nvToolsExt.h>

typedef struct {
    int tid;              // thread id
    int num_threads;      // total number of threads
    int iters;            // max iterations
    double left, right;   // real axis range
    double lower, upper;  // imaginary axis range
    int width, height;    // image dimensions
    int *image;           // shared output buffer
} ThreadData;

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
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);

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

void* mandelbrot_worker(void* arg) {
    ThreadData* t = (ThreadData*)arg;

    int height = t->height;
    int width = t->width;
    int iters = t->iters;
    double left = t->left;
    double right = t->right;
    double lower = t->lower;
    double upper = t->upper;
    int num_threads = t->num_threads;
    int tid = t->tid;
    int *image = t->image;

    // Divide rows among threads
    int rows_per_thread = height / num_threads;
    int start_row = tid * rows_per_thread;
    int end_row = (tid == num_threads - 1) ? height : start_row + rows_per_thread;

    for (int j = start_row; j < end_row; ++j) {
        double y0 = j * ((upper - lower) / height) + lower;
        for (int i = 0; i < width; ++i) {
            double x0 = i * ((right - left) / width) + left;

            int repeats = 0;
            double x = 0, y = 0;
            double length_squared = 0;
            while (repeats < iters && length_squared < 4.0) {
                double temp = x * x - y * y + x0;
                y = 2 * x * y + y0;
                x = temp;
                length_squared = x * x + y * y;
                ++repeats;
            }
            image[j * width + i] = repeats;
        }
    }

    pthread_exit(NULL);
}

int main(int argc, char** argv) {
    nvtxRangePush("CPU");
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    int num_cpus = CPU_COUNT(&cpu_set);
    printf("%d cpus available\n", num_cpus);

    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    int iters = strtol(argv[2], 0, 10);
    double left = strtod(argv[3], 0);
    double right = strtod(argv[4], 0);
    double lower = strtod(argv[5], 0);
    double upper = strtod(argv[6], 0);
    int width = strtol(argv[7], 0, 10);
    int height = strtol(argv[8], 0, 10);
    int num_threads = CPU_COUNT(&cpu_set);

    /* allocate memory for image */
    int* image = (int*)malloc(width * height * sizeof(int));
    assert(image);

    pthread_t* threads = (pthread_t*)malloc(num_threads * sizeof(pthread_t));
    ThreadData* tdata = (ThreadData*)malloc(num_threads * sizeof(ThreadData));

    /* launch threads */
    for (int t = 0; t < num_threads; ++t) {
        tdata[t].tid = t;
        tdata[t].num_threads = num_threads;
        tdata[t].iters = iters;
        tdata[t].left = left;
        tdata[t].right = right;
        tdata[t].lower = lower;
        tdata[t].upper = upper;
        tdata[t].width = width;
        tdata[t].height = height;
        tdata[t].image = image;
        pthread_create(&threads[t], NULL, mandelbrot_worker, &tdata[t]);
    }

    /* wait for all threads */
    for (int t = 0; t < num_threads; ++t)
        pthread_join(threads[t], NULL);

    /* draw and cleanup */
    nvtxRangePop(); nvtxRangePush("IO");
    write_png(filename, iters, width, height, image);
    nvtxRangePop(); nvtxRangePush("CPU");
    free(image);
    free(threads);
    free(tdata);
    nvtxRangePop();
    return 0;
}