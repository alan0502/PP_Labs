#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <omp.h>

#ifndef TILE_SIZE
#define TILE_SIZE 64
#endif

#define WORK_PER_THREAD 4
#define BLOCK_DIM (TILE_SIZE / WORK_PER_THREAD)

constexpr unsigned short INF = 65535/2; // Use half of max to prevent overflow

void input(char* inFileName);
void output(char* outFileName, unsigned short* buffer, int n, int padded_n);
int ceil_int(int a, int b);

int n, m;
int padded_n;
unsigned short* Dist;

// ----------------- CUDA error check -----------------
#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(err__));          \
        }                                                                    \
    } while (0)

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
        for(int i=0; i<WORK_PER_THREAD; ++i) a[i] = tile[row_start + i][k];
        #pragma unroll
        for(int j=0; j<WORK_PER_THREAD; ++j) b[j] = tile[k][col_start + j];
        
        #pragma unroll
        for(int i=0; i<WORK_PER_THREAD; ++i) {
            #pragma unroll
            for(int j=0; j<WORK_PER_THREAD; ++j) {
                tile[row_start + i][col_start + j] = min((int)tile[row_start + i][col_start + j], a[i] + b[j]);
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
    for(int i=0; i<WORK_PER_THREAD; ++i) {
        #pragma unroll
        for(int j=0; j<WORK_PER_THREAD; ++j) {
            t_val[i][j] = target[row_start + i][col_start + j];
        }
    }
    
    #pragma unroll
    for (int k = 0; k < TILE_SIZE; ++k) {
        int a[WORK_PER_THREAD]; 
        int b[WORK_PER_THREAD]; 
        
        if (mode == 0) { 
            #pragma unroll
            for(int i=0; i<WORK_PER_THREAD; ++i) a[i] = pivot[row_start + i][k];
            #pragma unroll
            for(int j=0; j<WORK_PER_THREAD; ++j) b[j] = target[k][col_start + j];
        } else { 
            #pragma unroll
            for(int i=0; i<WORK_PER_THREAD; ++i) a[i] = target[row_start + i][k];
            #pragma unroll
            for(int j=0; j<WORK_PER_THREAD; ++j) b[j] = pivot[k][col_start + j];
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
        for(int i=0; i<WORK_PER_THREAD; ++i) a[i] = row_tile[row_start + i][k];
        #pragma unroll
        for(int j=0; j<WORK_PER_THREAD; ++j) b[j] = col_tile[k][col_start + j];
        
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

// Tile = 64x64 → 設成 64
#ifndef TILE_SIZE
#define TILE_SIZE 64
#endif

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s input.bin output.bin\n", argv[0]);
        return 1;
    }

    input(argv[1]);

    int B = TILE_SIZE;
    padded_n = ceil_int(n, B) * B; // padded size
    int round = padded_n / B;

    unsigned short* h_padded_Dist = (unsigned short*)malloc(sizeof(unsigned short) * padded_n * padded_n);
    
    // Initialize with INF
    for (size_t i = 0; i < (size_t)padded_n * padded_n; ++i) h_padded_Dist[i] = INF;
    
    // Copy Dist to h_padded_Dist
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            h_padded_Dist[i * padded_n + j] = Dist[i * n + j];
        }
    }
    // Set diagonal to 0 for padded area (if any)，因為這些是新增的節點，自己到自己的距離應該是 0
    for (int i = n; i < padded_n; ++i) {
        h_padded_Dist[i * padded_n + i] = 0;
    }
    
    free(Dist); // Free original input buffer to save memory
    Dist = NULL;

    size_t totalBytes = sizeof(unsigned short) * (size_t)padded_n * (size_t)padded_n;

    // pin host memory 加速 H2D / D2H
    CHECK_CUDA(cudaHostRegister(h_padded_Dist, totalBytes, cudaHostRegisterDefault));

    // enable peer access (之後要做真 multi-GPU 的必要條件)
    int devCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&devCount));
    if (devCount >= 2) {
        int can01 = 0, can10 = 0;
        CHECK_CUDA(cudaDeviceCanAccessPeer(&can01, 0, 1)); // dev0 access dev1
        CHECK_CUDA(cudaDeviceCanAccessPeer(&can10, 1, 0)); // dev1 access dev0
        if (can01) { // enable peer access from dev0 to dev1
            CHECK_CUDA(cudaSetDevice(0));
            cudaDeviceEnablePeerAccess(1, 0);
        }
        if (can10) { // enable peer access from dev1 to dev0
            CHECK_CUDA(cudaSetDevice(1));
            cudaDeviceEnablePeerAccess(0, 0);
        }
    }

    omp_set_num_threads(2); // Use 2 threads for 2 GPUs

    int error_flag = 0;
    unsigned short* dev_ptrs[2]; // Shared array for device pointers
    cudaEvent_t pivot_ready[2][32];
    cudaEvent_t row_ready[2][32];

// Thread parallel region
#pragma omp parallel shared(error_flag, dev_ptrs, pivot_ready, row_ready)
    {
        int tid = omp_get_thread_num();      // 0 or 1
        int peer = 1 - tid;
        CHECK_CUDA(cudaSetDevice(tid));      // GPU 0 / GPU 1

        unsigned short* d_Dist = NULL;
        cudaStream_t stream;
        CHECK_CUDA(cudaStreamCreate(&stream));
        cudaStream_t stream_copy;
        CHECK_CUDA(cudaStreamCreate(&stream_copy));

        cudaEvent_t event_p1[32];
        cudaEvent_t event_p2_row[32];
        
        for (int i = 0; i < 32; ++i) {
            CHECK_CUDA(cudaEventCreateWithFlags(&pivot_ready[tid][i], cudaEventDisableTiming));
            CHECK_CUDA(cudaEventCreateWithFlags(&row_ready[tid][i], cudaEventDisableTiming));
            CHECK_CUDA(cudaEventCreateWithFlags(&event_p1[i], cudaEventDisableTiming));
            CHECK_CUDA(cudaEventCreateWithFlags(&event_p2_row[i], cudaEventDisableTiming));
        }

        // alloc
        if (!error_flag) {
            if (cudaMalloc(&d_Dist, totalBytes) != cudaSuccess) {
                printf("cudaMalloc failed on device %d\n", tid);
                error_flag = 1;
            }
        }
        
        // Share pointer
        dev_ptrs[tid] = d_Dist;
        #pragma omp barrier

        // Calculate ranges
        // tid 0 負責上半部，tid 1 負責下半部
        int start_block_y = (tid == 0) ? 0 : round / 2; 
        int end_block_y   = (tid == 0) ? round / 2 : round;
        int num_block_y   = end_block_y - start_block_y;
        
        int start_row = start_block_y * B;
        int end_row   = end_block_y * B;
        // if (end_row > n) end_row = n; // No need, padded
        int num_rows  = end_row - start_row;

        // copy H2D (Partial)
        if (!error_flag) {
            if (num_rows > 0) {
                CHECK_CUDA(cudaMemcpyAsync(
                    d_Dist + (size_t)start_row * padded_n, 
                    h_padded_Dist + (size_t)start_row * padded_n, 
                    sizeof(unsigned short) * (size_t)num_rows * padded_n,
                    cudaMemcpyHostToDevice, stream));
            }
            CHECK_CUDA(cudaStreamSynchronize(stream));
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
                    // Phase 1 (in stream)
                    phase1_kernel<<<dim3(1,1), block, 0, stream>>>(d_Dist, padded_n, r);
                    
                    // Record Phase 1 done
                    CHECK_CUDA(cudaEventRecord(event_p1[buf], stream));
                    
                    // Copy Pivot Block to Peer (in stream_copy)
                    CHECK_CUDA(cudaStreamWaitEvent(stream_copy, event_p1[buf], 0));
                    
                    int r_start = r * B;
                    int cur_B = B;

                    size_t offset = (size_t)r_start * padded_n + (size_t)r_start;
                    CHECK_CUDA(cudaMemcpy2DAsync(
                        dev_ptrs[peer] + offset, padded_n * sizeof(unsigned short),
                        d_Dist + offset, padded_n * sizeof(unsigned short),
                        cur_B * sizeof(unsigned short), cur_B, // width (bytes), height
                        cudaMemcpyDeviceToDevice, stream_copy));
                        
                    // Record pivot_ready in stream_copy
                    CHECK_CUDA(cudaEventRecord(pivot_ready[tid][buf], stream_copy));
                }
                

                #pragma omp barrier // 因為後面 Phase 2 需要 peer 等 owner 的 pivot_ready

                // Phase 2
                // 這邊 owner 負責 Phase 2 Row 和 Phase 2 Col，peer 負責 Phase 2 Col
                if (tid == owner) {
                    // Phase 2 Row (in stream)
                    // mode=0, offset=0
                    phase2_kernel<<<grid_row, block, 0, stream>>>(d_Dist, padded_n, r, 0, 0);
                    
                    // Record Phase 2 Row done
                    CHECK_CUDA(cudaEventRecord(event_p2_row[buf], stream));
                    
                    // Copy updated pivot row strip to peer (in stream_copy)
                    CHECK_CUDA(cudaStreamWaitEvent(stream_copy, event_p2_row[buf], 0));
                    
                    int r_start = r * B;
                    int cur_B = B;
                    size_t row_offset = (size_t)r_start * padded_n;
                    
                    CHECK_CUDA(cudaMemcpyAsync(
                        dev_ptrs[peer] + row_offset,
                        d_Dist + row_offset,
                        sizeof(unsigned short) * (size_t)cur_B * padded_n,
                        cudaMemcpyDeviceToDevice,
                        stream_copy));
                        
                    // Record row_ready in stream_copy
                    CHECK_CUDA(cudaEventRecord(row_ready[tid][buf], stream_copy));
                } else {
                    // Peer
                    // Wait for Pivot Block (recorded in stream_copy by owner)
                    CHECK_CUDA(cudaStreamWaitEvent(stream, pivot_ready[owner][buf], 0));
                    
                    // Phase 2 Col
                    // mode=1, offset=start_block_y
                    // 注意 col 不用交換，因為每個 GPU 的 col block 本來就有自己的
                    phase2_kernel<<<grid_col, block, 0, stream>>>(d_Dist, padded_n, r, 1, start_block_y);
                }
                
                // Owner also needs to do Phase 2 Col (in stream)
                // This can now run in parallel with Copy Pivot Row (in stream_copy)
                if (tid == owner) {
                     // mode=1, offset=start_block_y
                     // 注意 col 不用交換，因為每個 GPU 的 col block 本來就有自己的
                     phase2_kernel<<<grid_col, block, 0, stream>>>(d_Dist, padded_n, r, 1, start_block_y);
                }
                
                #pragma omp barrier

                if (tid != owner) {
                    // Wait for Pivot Row (recorded in stream_copy by owner)
                    CHECK_CUDA(cudaStreamWaitEvent(stream, row_ready[owner][buf], 0));
                }

                // Phase 3
                phase3_kernel<<<grid_all, block, 0, stream>>>(d_Dist, padded_n, r, start_block_y);
                
                if ((r + 1) % 32 == 0) {
                    CHECK_CUDA(cudaStreamSynchronize(stream));
                }
                #pragma omp barrier
            }

            // copy back only this GPU's rows
            if (num_rows > 0) {
                CHECK_CUDA(cudaMemcpyAsync(
                    h_padded_Dist + (size_t)start_row * padded_n,
                    d_Dist + (size_t)start_row * padded_n,
                    sizeof(unsigned short) * (size_t)num_rows * padded_n,
                    cudaMemcpyDeviceToHost, stream));
                CHECK_CUDA(cudaStreamSynchronize(stream));
            }
        }

        if (d_Dist) cudaFree(d_Dist);
        cudaStreamDestroy(stream);
        cudaStreamDestroy(stream_copy);
        for (int i = 0; i < 32; ++i) {
            cudaEventDestroy(pivot_ready[tid][i]);
            cudaEventDestroy(row_ready[tid][i]);
            cudaEventDestroy(event_p1[i]);
            cudaEventDestroy(event_p2_row[i]);
        }
    } // parallel 結束

    if (error_flag) {
        fprintf(stderr, "Aborting due to CUDA errors.\n");
        cudaHostUnregister(h_padded_Dist);
        free(h_padded_Dist);
        // free(Dist); // Already freed
        return 1;
    }

    output(argv[2], h_padded_Dist, n, padded_n);

    cudaHostUnregister(h_padded_Dist);
    free(h_padded_Dist);
    // free(Dist); // Already freed
    return 0;
}

// ==================== I/O & helpers ====================

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    if (!file) {
        fprintf(stderr, "Error: Cannot open input file %s\n", infile);
        exit(1);
    }
    fread(&n, sizeof(int), 1, file); // number of vertices
    fread(&m, sizeof(int), 1, file); // number of edges

    Dist = (unsigned short*)malloc(sizeof(unsigned short) * n * n);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                Dist[i * (size_t)n + j] = 0; // distance to itself is 0
            } else {
                Dist[i * (size_t)n + j]= INF; // initialize other distances to INF
            }
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0] * (size_t)n + pair[1]] = (unsigned short)pair[2]; // set the distance for each edge (u, v), pair[0]=u, pair[1]=v, pair[2]=weight
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