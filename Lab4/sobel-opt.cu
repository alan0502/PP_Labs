#include <iostream>
#include <cstdlib>
#include <cassert>
#include <zlib.h>
#include <png.h>
#include <cuda_runtime.h>
#include <math.h>

#define Z 2
#define Y 5
#define X 5
#define xBound (X / 2)
#define yBound (Y / 2)
#define SCALE 8

int read_png(const char* filename, unsigned char** image, unsigned* height, 
             unsigned* width, unsigned* channels) {

    unsigned char sig[8];
    FILE* infile;
    infile = fopen(filename, "rb");
    if (!infile) {
        std::cerr << "Cannot open input file\n";
        return 1;
    }

    fread(sig, 1, 8, infile);
    if (!png_check_sig(sig, 8))
        return 1;   /* bad signature */

    png_structp png_ptr;
    png_infop info_ptr;

    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr)
        return 4;   /* out of memory */
  
    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        fclose(infile);
        return 4;   /* out of memory */
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(infile);
        return 2;
    }

    png_init_io(png_ptr, infile);
    png_set_sig_bytes(png_ptr, 8);
    png_read_info(png_ptr, info_ptr);
    int bit_depth, color_type;
    png_get_IHDR(png_ptr, info_ptr, width, height, &bit_depth, &color_type, NULL, NULL, NULL);

    png_uint_32  i, rowbytes;
    png_read_update_info(png_ptr, info_ptr);
    rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    *channels = (int) png_get_channels(png_ptr, info_ptr);

    png_bytep* row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * (*height));
    if (!row_pointers) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(infile);
        return 3;
    }

    if ((*image = (unsigned char *) malloc(rowbytes * (*height))) == NULL) {
        free(row_pointers);
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(infile);
        return 3;
    }

    for (i = 0;  i < *height;  ++i)
        row_pointers[i] = *image + i * rowbytes;
    png_read_image(png_ptr, row_pointers);
    png_read_end(png_ptr, NULL);

    free(row_pointers);
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    fclose(infile);
    return 0;
}

void write_png(const char* filename, png_bytep image, const unsigned height, const unsigned width, 
               const unsigned channels) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        std::cerr << "Cannot open output file\n";
        return;
    }

    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) {
        fclose(fp);
        return;
    }
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
        fclose(fp);
        return;
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(fp);
        return;
    }

    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8,
                 PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);

    png_bytep* row_ptr = (png_bytep*)malloc(sizeof(png_bytep) * height);
    for (unsigned i = 0; i < height; ++ i) {
        row_ptr[i] = image + i * width * channels * sizeof(unsigned char);
    }
    png_write_image(png_ptr, row_ptr);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
    free(row_ptr);
}

// Same masks as your code, stored in constant memory
__constant__ char mask[Z][Y][X] = { 
    { { -1, -4, -6, -4, -1 },
      { -2, -8, -12, -8, -2 },
      { 0, 0, 0, 0, 0 },
      { 2, 8, 12, 8, 2 },
      { 1, 4, 6, 4, 1 } },
    { { -1, -2, 0, 2, 1 },
      { -4, -8, 0, 8, 4 },
      { -6, -12, 0, 12, 6 },
      { -4, -8, 0, 8, 4 },
      { -1, -2, 0, 2, 1 } } 
};

inline __device__ int clamp_int(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

/*
 * Shared-memory Sobel:
 *  - Each block processes a tile of size blockDim.x x blockDim.y
 *  - We allocate halo of radius xBound/yBound around it in shared memory.
 *  - We clamp global coordinates when loading into shared memory (no OOB).
 *  - Threads outside image bounds return immediately.
 */
__global__ void sobel(const unsigned char * __restrict__ s,
                             unsigned char * __restrict__ t,
                             unsigned height, unsigned width, unsigned channels) {

    const int RADIUS_X = xBound;  // 2
    const int RADIUS_Y = yBound;  // 2

    const int TILE_W = blockDim.x + 2 * RADIUS_X;
    const int TILE_H = blockDim.y + 2 * RADIUS_Y;

    // Global pixel this thread is responsible for
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Shared memory layout: [B][G][R], each size TILE_W * TILE_H
    extern __shared__ unsigned char shmem[];
    unsigned char* tileB = shmem;
    unsigned char* tileG = tileB + TILE_W * TILE_H;
    unsigned char* tileR = tileG + TILE_W * TILE_H;

    // Load tile + halo into shared memory
    // Each thread cooperatively loads multiple positions to cover full TILE_H x TILE_W
    for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < TILE_W * TILE_H; i += blockDim.x * blockDim.y) {
        int dy = i / TILE_W;
        int dx = i % TILE_W;
        
        int global_y = blockIdx.y * blockDim.y + dy - RADIUS_Y;
        global_y = clamp_int(global_y, 0, (int)height - 1);

        int global_x = blockIdx.x * blockDim.x + dx - RADIUS_X;
        global_x = clamp_int(global_x, 0, (int)width - 1);

        int srcIdx = channels * (global_y * (int)width + global_x);
        int tileIdx = dy * TILE_W + dx;

        // assuming channels >= 3; we're using B,G,R in that order
        unsigned char B = s[srcIdx + 0];
        unsigned char G = s[srcIdx + 1];
        unsigned char R = s[srcIdx + 2];

        tileB[tileIdx] = B;
        tileG[tileIdx] = G;
        tileR[tileIdx] = R;
    }

    __syncthreads();

    // If this thread's pixel is outside image bounds, nothing to do
    if (x >= (int)width || y >= (int)height) {
        return;
    }

    // Position of this thread's pixel in the shared tile
    int tx = threadIdx.x + RADIUS_X;
    int ty = threadIdx.y + RADIUS_Y;

    short val[Z][3] = {0}; // [mask direction][channel: 0=B, 1=G, 2=R]

    // Convolution using shared memory (no OOB because tile fully covers halo)
    for (int i = 0; i < Z; ++i) {
        for (int v = -RADIUS_Y; v <= RADIUS_Y; ++v) {
            int ty2 = ty + v;  // always in [0, TILE_H-1]
            for (int u = -RADIUS_X; u <= RADIUS_X; ++u) {
                int tx2 = tx + u;  // always in [0, TILE_W-1]
                int tIdx = ty2 * TILE_W + tx2;

                unsigned char B = tileB[tIdx];
                unsigned char G = tileG[tIdx];
                unsigned char R = tileR[tIdx];

                char m = mask[i][v + RADIUS_Y][u + RADIUS_X];

                val[i][0] += B * m;
                val[i][1] += G * m;
                val[i][2] += R * m;
            }
        }
    }

    float totalB = 0.0, totalG = 0.0, totalR = 0.0;
    for (int i = 0; i < Z; ++i) {
        totalB += val[i][0] * val[i][0];
        totalG += val[i][1] * val[i][1];
        totalR += val[i][2] * val[i][2];
    }

    totalB = sqrtf(totalB) / SCALE;
    totalG = sqrtf(totalG) / SCALE;
    totalR = sqrtf(totalR) / SCALE;

    unsigned char cB = (totalB > 255.0) ? 255 : (unsigned char)totalB;
    unsigned char cG = (totalG > 255.0) ? 255 : (unsigned char)totalG;
    unsigned char cR = (totalR > 255.0) ? 255 : (unsigned char)totalR;

    int outIdx = channels * (y * (int)width + x);
    t[outIdx + 0] = cB;
    t[outIdx + 1] = cG;
    t[outIdx + 2] = cR;
}

int main(int argc, char **argv) {
    assert(argc == 3);
    unsigned height, width, channels;
    unsigned char *src = NULL, *dst = NULL;
    unsigned char *dsrc = NULL, *ddst = NULL;

    /* read the image to src, and get height, width, channels */
    if (read_png(argv[1], &src, &height, &width, &channels)) {
        std::cerr << "Error in read png" << std::endl;
        return -1;
    }

    size_t numBytes = (size_t)height * (size_t)width * (size_t)channels * sizeof(unsigned char);

    dst = (unsigned char *)malloc(numBytes);
    if (!dst) {
        std::cerr << "Failed to allocate dst\n";
        free(src);
        return -1;
    }

    // Optional pinned memory for src
    cudaError_t err = cudaHostRegister(src, numBytes, cudaHostRegisterDefault);
    if (err != cudaSuccess) {
        std::cerr << "cudaHostRegister failed: " << cudaGetErrorString(err) << "\n";
        // Not fatal; we can still continue without pinned memory
    }

    // cudaMalloc(...) for device src and device dst
    cudaMalloc(&dsrc, numBytes);
    cudaMalloc(&ddst, numBytes);

    // copy source image to device
    cudaMemcpy(dsrc, src, numBytes, cudaMemcpyHostToDevice);

    // choose 2D block / grid
    dim3 block(32, 16);  // 512 threads/block, reasonable for shared memory
    dim3 grid((width  + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    // shared memory size: 3 channels * TILE_W * TILE_H
    int TILE_W = block.x + 2 * xBound;
    int TILE_H = block.y + 2 * yBound;
    size_t sharedBytes = 3 * TILE_W * TILE_H * sizeof(unsigned char);

    // launch kernel
    sobel<<<grid, block, sharedBytes>>>(dsrc, ddst, height, width, channels);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(dsrc);
        cudaFree(ddst);
        if (src) {
            cudaHostUnregister(src); // only if registered successfully (harmless if not)
        }
        free(src);
        free(dst);
        return -1;
    }

    cudaDeviceSynchronize();

    // copy result image back
    cudaMemcpy(dst, ddst, numBytes, cudaMemcpyDeviceToHost);

    write_png(argv[2], dst, height, width, channels);

    if (src) cudaHostUnregister(src);
    free(src);
    free(dst);
    cudaFree(dsrc);
    cudaFree(ddst);

    return 0;
}
