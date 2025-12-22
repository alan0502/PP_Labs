#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include <omp.h>

#ifndef TILE_SIZE
#define TILE_SIZE 64
#endif

#define WORK_PER_THREAD 4
#define BLOCK_DIM (TILE_SIZE / WORK_PER_THREAD)

constexpr unsigned short INF = 65535 / 2; // Use half of max to prevent overflow

// ====== HIP error check macro ======
#define CHECK_HIP(call)                                                       \
    do {                                                                      \
        hipError_t err__ = (call);                                            \
        if (err__ != hipSuccess) {                                            \
            fprintf(stderr, "HIP error %s at %s:%d\n",                        \
                    hipGetErrorString(err__), __FILE__, __LINE__);           \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

void input(char* inFileName);
void output(char* outFileName, unsigned short* buffer, int n, int padded_n);
int ceil_int(int a, int b);

int n, m;
int padded_n;
unsigned short* Dist;


// ==================== Kernels ====================

__global__ void phase1_kernel(unsigned short* __restrict__ d_Dist, int n, int Round) {
    __shared__ unsigned short tile[TILE_SIZE][TILE_SIZE + 1];

    int tx = threadIdx.x; 
    int ty = threadIdx.y; 

    int row_start = ty * WORK_PER_THREAD;
    int col_start = tx * WORK_PER_THREAD;

    int global_row_start = Round * TILE_SIZE + row_start;
    int global_col_start = Round * TILE_SIZE + col_start;

    #pragma unroll
    for (int i = 0; i < WORK_PER_THREAD; ++i) {
        int r = global_row_start + i;
        #pragma unroll
        for (int j = 0; j < WORK_PER_THREAD; j += 4) {
            int c = global_col_start + j;
            if (r < n && c < n) {
                ushort4 val = *reinterpret_cast<ushort4*>(&d_Dist[r * n + c]);
                tile[row_start + i][col_start + j + 0] = val.x;
                tile[row_start + i][col_start + j + 1] = val.y;
                tile[row_start + i][col_start + j + 2] = val.z;
                tile[row_start + i][col_start + j + 3] = val.w;
            }
        }
    }

    __syncthreads();

    #pragma unroll
    for (int k = 0; k < TILE_SIZE; ++k) {
        int a[WORK_PER_THREAD];
        int b[WORK_PER_THREAD];
        
        #pragma unroll
        for(int i = 0; i < WORK_PER_THREAD; ++i) a[i] = tile[row_start + i][k];
        #pragma unroll
        for(int j = 0; j < WORK_PER_THREAD; ++j) b[j] = tile[k][col_start + j];
        
        #pragma unroll
        for(int i = 0; i < WORK_PER_THREAD; ++i) {
            #pragma unroll
            for(int j = 0; j < WORK_PER_THREAD; ++j) {
                tile[row_start + i][col_start + j] =
                    min((int)tile[row_start + i][col_start + j], a[i] + b[j]);
            }
        }
        __syncthreads();
    }
    
    #pragma unroll
    for (int i = 0; i < WORK_PER_THREAD; ++i) {
        int r = global_row_start + i;
        #pragma unroll
        for (int j = 0; j < WORK_PER_THREAD; j += 4) {
            int c = global_col_start + j;
            if (r < n && c < n) {
                ushort4 val;
                val.x = tile[row_start + i][col_start + j + 0];
                val.y = tile[row_start + i][col_start + j + 1];
                val.z = tile[row_start + i][col_start + j + 2];
                val.w = tile[row_start + i][col_start + j + 3];
                *reinterpret_cast<ushort4*>(&d_Dist[r * n + c]) = val;
            }
        }
    }
}

__global__ void phase2_kernel(unsigned short* __restrict__ d_Dist, int n, int Round, int mode, int offset) {
    __shared__ unsigned short pivot[TILE_SIZE][TILE_SIZE + 1];
    __shared__ unsigned short target[TILE_SIZE][TILE_SIZE + 1];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_start = ty * WORK_PER_THREAD;
    int col_start = tx * WORK_PER_THREAD;
    
    int blk_idx = blockIdx.x + offset; // Add offset for multi-gpu splitting
    if (blk_idx == Round) return;
    
    int pivot_row_start = Round * TILE_SIZE + row_start;
    int pivot_col_start = Round * TILE_SIZE + col_start;
    
    int target_row_start, target_col_start;
    
    if (mode == 0) { // Row Block
        target_row_start = Round * TILE_SIZE + row_start;
        target_col_start = blk_idx * TILE_SIZE + col_start;
    } else { // Col Block
        target_row_start = blk_idx * TILE_SIZE + row_start;
        target_col_start = Round * TILE_SIZE + col_start;
    }
    
    #pragma unroll
    for (int i = 0; i < WORK_PER_THREAD; ++i) {
        int r = pivot_row_start + i;
        #pragma unroll
        for (int j = 0; j < WORK_PER_THREAD; j += 4) {
            int c = pivot_col_start + j;
            ushort4 val = *reinterpret_cast<ushort4*>(&d_Dist[r * n + c]);
            pivot[row_start + i][col_start + j + 0] = val.x;
            pivot[row_start + i][col_start + j + 1] = val.y;
            pivot[row_start + i][col_start + j + 2] = val.z;
            pivot[row_start + i][col_start + j + 3] = val.w;
        }
    }
    
    #pragma unroll
    for (int i = 0; i < WORK_PER_THREAD; ++i) {
        int r = target_row_start + i;
        #pragma unroll
        for (int j = 0; j < WORK_PER_THREAD; j += 4) {
            int c = target_col_start + j;
            ushort4 val = *reinterpret_cast<ushort4*>(&d_Dist[r * n + c]);
            target[row_start + i][col_start + j + 0] = val.x;
            target[row_start + i][col_start + j + 1] = val.y;
            target[row_start + i][col_start + j + 2] = val.z;
            target[row_start + i][col_start + j + 3] = val.w;
        }
    }
    
    __syncthreads();
    
    int t_val[WORK_PER_THREAD][WORK_PER_THREAD];
    #pragma unroll
    for(int i = 0; i < WORK_PER_THREAD; ++i) {
        #pragma unroll
        for(int j = 0; j < WORK_PER_THREAD; ++j) {
            t_val[i][j] = target[row_start + i][col_start + j];
        }
    }
    
    #pragma unroll
    for (int k = 0; k < TILE_SIZE; ++k) {
        int a[WORK_PER_THREAD]; 
        int b[WORK_PER_THREAD]; 
        
        if (mode == 0) { 
            #pragma unroll
            for(int i = 0; i < WORK_PER_THREAD; ++i) a[i] = pivot[row_start + i][k];
            #pragma unroll
            for(int j = 0; j < WORK_PER_THREAD; ++j) b[j] = target[k][col_start + j];
        } else { 
            #pragma unroll
            for(int i = 0; i < WORK_PER_THREAD; ++i) a[i] = target[row_start + i][k];
            #pragma unroll
            for(int j = 0; j < WORK_PER_THREAD; ++j) b[j] = pivot[k][col_start + j];
        }
        
        #pragma unroll
        for (int i = 0; i < WORK_PER_THREAD; ++i) {
            #pragma unroll
            for (int j = 0; j < WORK_PER_THREAD; ++j) {
                t_val[i][j] = min(t_val[i][j], a[i] + b[j]);
            }
        }
    }
    
    #pragma unroll
    for (int i = 0; i < WORK_PER_THREAD; ++i) {
        int r = target_row_start + i;
        #pragma unroll
        for (int j = 0; j < WORK_PER_THREAD; j += 4) {
            int c = target_col_start + j;
            ushort4 val;
            val.x = t_val[i][j + 0];
            val.y = t_val[i][j + 1];
            val.z = t_val[i][j + 2];
            val.w = t_val[i][j + 3];
            *reinterpret_cast<ushort4*>(&d_Dist[r * n + c]) = val;
        }
    }
}

__global__ void phase3_kernel(unsigned short* __restrict__ d_Dist, int n, int Round, int offset_y) {
    int bx = blockIdx.x;
    int by = blockIdx.y + offset_y; // Add offset for multi-gpu
    
    if (bx == Round || by == Round) return;
    
    __shared__ unsigned short row_tile[TILE_SIZE][TILE_SIZE + 1]; 
    __shared__ unsigned short col_tile[TILE_SIZE][TILE_SIZE + 1]; 
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_start = ty * WORK_PER_THREAD;
    int col_start = tx * WORK_PER_THREAD;
    
    int row_row_start = by * TILE_SIZE + row_start;
    int row_col_start = Round * TILE_SIZE + col_start;
    
    #pragma unroll
    for (int i = 0; i < WORK_PER_THREAD; ++i) {
        int r = row_row_start + i;
        #pragma unroll
        for (int j = 0; j < WORK_PER_THREAD; j += 4) {
            int c = row_col_start + j;
            ushort4 val = *reinterpret_cast<ushort4*>(&d_Dist[r * n + c]);
            row_tile[row_start + i][col_start + j + 0] = val.x;
            row_tile[row_start + i][col_start + j + 1] = val.y;
            row_tile[row_start + i][col_start + j + 2] = val.z;
            row_tile[row_start + i][col_start + j + 3] = val.w;
        }
    }
    
    int c_row_start = Round * TILE_SIZE + row_start;
    int c_col_start = bx * TILE_SIZE + col_start;
    
    #pragma unroll
    for (int i = 0; i < WORK_PER_THREAD; ++i) {
        int r = c_row_start + i;
        #pragma unroll
        for (int j = 0; j < WORK_PER_THREAD; j += 4) {
            int c = c_col_start + j;
            ushort4 val = *reinterpret_cast<ushort4*>(&d_Dist[r * n + c]);
            col_tile[row_start + i][col_start + j + 0] = val.x;
            col_tile[row_start + i][col_start + j + 1] = val.y;
            col_tile[row_start + i][col_start + j + 2] = val.z;
            col_tile[row_start + i][col_start + j + 3] = val.w;
        }
    }
    
    __syncthreads();
    
    int my_row_start = by * TILE_SIZE + row_start;
    int my_col_start = bx * TILE_SIZE + col_start;
    
    int c_val[WORK_PER_THREAD][WORK_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < WORK_PER_THREAD; ++i) {
        int r = my_row_start + i;
        #pragma unroll
        for (int j = 0; j < WORK_PER_THREAD; j += 4) {
            int c = my_col_start + j;
            ushort4 val = *reinterpret_cast<ushort4*>(&d_Dist[r * n + c]);
            c_val[i][j + 0] = val.x;
            c_val[i][j + 1] = val.y;
            c_val[i][j + 2] = val.z;
            c_val[i][j + 3] = val.w;
        }
    }
    
    #pragma unroll
    for (int k = 0; k < TILE_SIZE; ++k) {
        int a[WORK_PER_THREAD];
        int b[WORK_PER_THREAD];
        
        #pragma unroll
        for(int i = 0; i < WORK_PER_THREAD; ++i) a[i] = row_tile[row_start + i][k];
        #pragma unroll
        for(int j = 0; j < WORK_PER_THREAD; ++j) b[j] = col_tile[k][col_start + j];
        
        #pragma unroll
        for (int i = 0; i < WORK_PER_THREAD; ++i) {
            #pragma unroll
            for (int j = 0; j < WORK_PER_THREAD; ++j) {
                c_val[i][j] = min(c_val[i][j], a[i] + b[j]);
            }
        }
    }
    
    #pragma unroll
    for (int i = 0; i < WORK_PER_THREAD; ++i) {
        int r = my_row_start + i;
        #pragma unroll
        for (int j = 0; j < WORK_PER_THREAD; j += 4) {
            int c = my_col_start + j;
            ushort4 val;
            val.x = c_val[i][j + 0];
            val.y = c_val[i][j + 1];
            val.z = c_val[i][j + 2];
            val.w = c_val[i][j + 3];
            *reinterpret_cast<ushort4*>(&d_Dist[r * n + c]) = val;
        }
    }
}


// ==================== main ====================

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s input.bin output.bin\n", argv[0]);
        return 1;
    }

    input(argv[1]);

    int B = TILE_SIZE;
    padded_n = ceil_int(n, B);  // number of tiles
    padded_n *= B;              // padded size
    int round = padded_n / B;

    unsigned short* h_padded_Dist =
        (unsigned short*)malloc(sizeof(unsigned short) * padded_n * padded_n);
    
    // Initialize with INF
    for (size_t i = 0; i < (size_t)padded_n * padded_n; ++i)
        h_padded_Dist[i] = INF;
    
    // Copy Dist to h_padded_Dist
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            h_padded_Dist[i * padded_n + j] = Dist[i * n + j];
        }
    }
    // Set diagonal to 0 for padded area (if any)
    for (int i = n; i < padded_n; ++i) {
        h_padded_Dist[i * padded_n + i] = 0;
    }
    
    free(Dist);
    Dist = NULL;

    size_t totalBytes = sizeof(unsigned short) * (size_t)padded_n * (size_t)padded_n;

    // pin host memory
    CHECK_HIP(hipHostRegister(h_padded_Dist, totalBytes, hipHostRegisterDefault));

    // enable peer access
    int devCount = 0;
    CHECK_HIP(hipGetDeviceCount(&devCount));
    if (devCount >= 2) {
        int can01 = 0, can10 = 0;
        CHECK_HIP(hipDeviceCanAccessPeer(&can01, 0, 1));
        CHECK_HIP(hipDeviceCanAccessPeer(&can10, 1, 0));
        if (can01) {
            CHECK_HIP(hipSetDevice(0));
            hipDeviceEnablePeerAccess(1, 0);
        }
        if (can10) {
            CHECK_HIP(hipSetDevice(1));
            hipDeviceEnablePeerAccess(0, 0);
        }
    }

    omp_set_num_threads(2);

    int error_flag = 0;
    unsigned short* dev_ptrs[2];
    hipEvent_t pivot_ready[2][32];
    hipEvent_t row_ready[2][32];

#pragma omp parallel shared(error_flag, dev_ptrs, pivot_ready, row_ready)
    {
        int tid = omp_get_thread_num();      // 0 or 1
        int peer = 1 - tid;
        CHECK_HIP(hipSetDevice(tid));        // GPU 0 / GPU 1

        unsigned short* d_Dist = NULL;
        hipStream_t stream;
        CHECK_HIP(hipStreamCreate(&stream));
        hipStream_t stream_copy;
        CHECK_HIP(hipStreamCreate(&stream_copy));

        hipEvent_t event_p1[32];
        hipEvent_t event_p2_row[32];
        
        for (int i = 0; i < 32; ++i) {
            CHECK_HIP(hipEventCreateWithFlags(&pivot_ready[tid][i],
                                              hipEventDisableTiming));
            CHECK_HIP(hipEventCreateWithFlags(&row_ready[tid][i],
                                              hipEventDisableTiming));
            CHECK_HIP(hipEventCreateWithFlags(&event_p1[i],
                                              hipEventDisableTiming));
            CHECK_HIP(hipEventCreateWithFlags(&event_p2_row[i],
                                              hipEventDisableTiming));
        }

        // alloc
        if (!error_flag) {
            if (hipMalloc(&d_Dist, totalBytes) != hipSuccess) {
                printf("hipMalloc failed on device %d\n", tid);
                error_flag = 1;
            }
        }
        
        dev_ptrs[tid] = d_Dist;
        #pragma omp barrier

        // tid 0: upper half, tid 1: lower half
        int start_block_y = (tid == 0) ? 0 : round / 2; 
        int end_block_y   = (tid == 0) ? round / 2 : round;
        int num_block_y   = end_block_y - start_block_y;
        
        int start_row = start_block_y * B;
        int end_row   = end_block_y * B;
        int num_rows  = end_row - start_row;

        // copy H2D (partial)
        if (!error_flag) {
            if (num_rows > 0) {
                CHECK_HIP(hipMemcpyAsync(
                    d_Dist + (size_t)start_row * padded_n, 
                    h_padded_Dist + (size_t)start_row * padded_n, 
                    sizeof(unsigned short) * (size_t)num_rows * padded_n,
                    hipMemcpyHostToDevice, stream));
            }
            CHECK_HIP(hipStreamSynchronize(stream));
        }

        dim3 block(BLOCK_DIM, BLOCK_DIM);
        dim3 grid_row(round, 1);
        dim3 grid_col(num_block_y, 1);
        dim3 grid_all(round, num_block_y);

#pragma omp barrier
        if (!error_flag) {

            for (int r = 0; r < round; ++r) {
                int owner = (r < round / 2) ? 0 : 1;
                int buf = r % 32;
                
                // Phase 1
                if (tid == owner) {
                    phase1_kernel<<<dim3(1,1), block, 0, stream>>>(d_Dist, padded_n, r);
                    
                    CHECK_HIP(hipEventRecord(event_p1[buf], stream));
                    
                    CHECK_HIP(hipStreamWaitEvent(stream_copy, event_p1[buf], 0));
                    
                    int r_start = r * B;
                    int cur_B = B;

                    size_t offset = (size_t)r_start * padded_n + (size_t)r_start;
                    CHECK_HIP(hipMemcpy2DAsync(
                        dev_ptrs[peer] + offset, padded_n * sizeof(unsigned short),
                        d_Dist + offset,       padded_n * sizeof(unsigned short),
                        cur_B * sizeof(unsigned short), cur_B,
                        hipMemcpyDeviceToDevice, stream_copy));
                        
                    CHECK_HIP(hipEventRecord(pivot_ready[tid][buf], stream_copy));
                }
                
                #pragma omp barrier

                // Phase 2
                if (tid == owner) {
                    // row
                    phase2_kernel<<<grid_row, block, 0, stream>>>(d_Dist, padded_n, r, 0, 0);
                    CHECK_HIP(hipEventRecord(event_p2_row[buf], stream));

                    // copy updated pivot row
                    CHECK_HIP(hipStreamWaitEvent(stream_copy, event_p2_row[buf], 0));
                    
                    int r_start = r * B;
                    int cur_B = B;
                    size_t row_offset = (size_t)r_start * padded_n;
                    
                    CHECK_HIP(hipMemcpyAsync(
                        dev_ptrs[peer] + row_offset,
                        d_Dist + row_offset,
                        sizeof(unsigned short) * (size_t)cur_B * padded_n,
                        hipMemcpyDeviceToDevice,
                        stream_copy));
                        
                    CHECK_HIP(hipEventRecord(row_ready[tid][buf], stream_copy));
                } else {
                    // peer: wait pivot block then col
                    CHECK_HIP(hipStreamWaitEvent(stream, pivot_ready[owner][buf], 0));
                    phase2_kernel<<<grid_col, block, 0, stream>>>(
                        d_Dist, padded_n, r, 1, start_block_y);
                }
                
                if (tid == owner) {
                    // owner also does phase2 col
                    phase2_kernel<<<grid_col, block, 0, stream>>>(
                        d_Dist, padded_n, r, 1, start_block_y);
                }
                
                #pragma omp barrier

                if (tid != owner) {
                    CHECK_HIP(hipStreamWaitEvent(stream, row_ready[owner][buf], 0));
                }

                // Phase 3
                phase3_kernel<<<grid_all, block, 0, stream>>>(
                    d_Dist, padded_n, r, start_block_y);
                
                if ((r + 1) % 32 == 0) {
                    CHECK_HIP(hipStreamSynchronize(stream));
                }
                #pragma omp barrier
            }

            if (num_rows > 0) {
                CHECK_HIP(hipMemcpyAsync(
                    h_padded_Dist + (size_t)start_row * padded_n,
                    d_Dist + (size_t)start_row * padded_n,
                    sizeof(unsigned short) * (size_t)num_rows * padded_n,
                    hipMemcpyDeviceToHost, stream));
                CHECK_HIP(hipStreamSynchronize(stream));
            }
        }

        if (d_Dist) hipFree(d_Dist);
        hipStreamDestroy(stream);
        hipStreamDestroy(stream_copy);
        for (int i = 0; i < 32; ++i) {
            hipEventDestroy(pivot_ready[tid][i]);
            hipEventDestroy(row_ready[tid][i]);
            hipEventDestroy(event_p1[i]);
            hipEventDestroy(event_p2_row[i]);
        }
    } // parallel

    if (error_flag) {
        fprintf(stderr, "Aborting due to HIP errors.\n");
        hipHostUnregister(h_padded_Dist);
        free(h_padded_Dist);
        return 1;
    }

    output(argv[2], h_padded_Dist, n, padded_n);

    hipHostUnregister(h_padded_Dist);
    free(h_padded_Dist);
    return 0;
}


// ==================== I/O & helpers ====================

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    if (!file) {
        fprintf(stderr, "Error: Cannot open input file %s\n", infile);
        exit(1);
    }
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    Dist = (unsigned short*)malloc(sizeof(unsigned short) * n * n);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                Dist[i * (size_t)n + j] = 0;
            } else {
                Dist[i * (size_t)n + j] = INF;
            }
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0] * (size_t)n + pair[1]] = (unsigned short)pair[2];
    }
    fclose(file);
}

void output(char* outFileName, unsigned short* buffer, int n, int padded_n) {
    FILE* outfile = fopen(outFileName, "wb");
    if (!outfile) {
        fprintf(stderr, "Error: Cannot open output file %s\n", outFileName);
        exit(1);
    }
    int* rowBuf = (int*)malloc(sizeof(int) * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            unsigned short val = buffer[i * (size_t)padded_n + j];
            if (val >= INF) rowBuf[j] = ((1 << 30) - 1);
            else rowBuf[j] = (int)val;
        }
        fwrite(rowBuf, sizeof(int), n, outfile);
    }
    free(rowBuf);
    fclose(outfile);
}

int ceil_int(int a, int b) { return (a + b - 1) / b; }
