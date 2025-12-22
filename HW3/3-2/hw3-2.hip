#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>

#ifndef TILE_SIZE
#define TILE_SIZE 8
#endif

#define WORK_PER_THREAD 4

#define BLOCK_DIM (TILE_SIZE / WORK_PER_THREAD)

const unsigned short INF = 65535;

int n, m;
unsigned short* Dist;

void input(char* inFileName);
void output(char* outFileName);

int ceil_int(int a, int b) { return (a + b - 1) / b; }

// Phase 1: Process the pivot block
// 每個 thread 負責計算 WORK_PER_THREAD x WORK_PER_THREAD 個元素
__global__ void phase1_kernel(unsigned short* __restrict__ d_Dist, int n, int Round) {
    __shared__ unsigned short tile[TILE_SIZE][TILE_SIZE + 1];

    int tx = threadIdx.x; 
    int ty = threadIdx.y; 

    // row_start, col_start 是這個 thread 在 tile 內負責的那塊 4×4 子區的左上角，就是 local index
    int row_start = ty * WORK_PER_THREAD;
    int col_start = tx * WORK_PER_THREAD;

    // global_row_start, global_col_start 是這個 thread 在整個矩陣內負責的那塊 4×4 子區的左上角，就是 global index
    int global_row_start = Round * TILE_SIZE + row_start;
    int global_col_start = Round * TILE_SIZE + col_start;

    // Load data from global memory to shared memory
    #pragma unroll
    for (int i = 0; i < WORK_PER_THREAD; ++i) {
        int r = global_row_start + i;
        #pragma unroll
        for (int j = 0; j < WORK_PER_THREAD; j += 4) { // 一次讀取 4 個 ushort，這樣每個 row 只需要 1 次 global memory 存取
            int c = global_col_start + j;
            if (r < n && c < n) {
                 unsigned long long val = *reinterpret_cast<unsigned long long*>(&d_Dist[r * n + c]);
                 tile[row_start + i][col_start + j + 0] = (unsigned short)(val & 0xFFFF);
                 tile[row_start + i][col_start + j + 1] = (unsigned short)((val >> 16) & 0xFFFF);
                 tile[row_start + i][col_start + j + 2] = (unsigned short)((val >> 32) & 0xFFFF);
                 tile[row_start + i][col_start + j + 3] = (unsigned short)((val >> 48) & 0xFFFF);
            }
        }
    }

    __syncthreads();

    // Compute shortest paths within the tile using registers
    #pragma unroll
    for (int k = 0; k < TILE_SIZE; ++k) {
        int a[WORK_PER_THREAD]; // store tile[row_start + i][k] in registers
        int b[WORK_PER_THREAD]; // store tile[k][col_start + j] in registers
        
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
    
    // Store results back to global memory
    #pragma unroll
    for (int i = 0; i < WORK_PER_THREAD; ++i) {
        int r = global_row_start + i;
        #pragma unroll
        for (int j = 0; j < WORK_PER_THREAD; j += 4) {
            int c = global_col_start + j;
            if (r < n && c < n) {
                unsigned long long val = 0;
                val |= (unsigned long long)tile[row_start + i][col_start + j + 0];
                val |= ((unsigned long long)tile[row_start + i][col_start + j + 1]) << 16;
                val |= ((unsigned long long)tile[row_start + i][col_start + j + 2]) << 32;
                val |= ((unsigned long long)tile[row_start + i][col_start + j + 3]) << 48;
                *reinterpret_cast<unsigned long long*>(&d_Dist[r * n + c]) = val;
            }
        }
    }
}

__global__ void phase2_kernel(unsigned short* __restrict__ d_Dist, int n, int Round) {
    __shared__ unsigned short pivot[TILE_SIZE][TILE_SIZE + 1]; // Shared memory for the pivot block
    __shared__ unsigned short target[TILE_SIZE][TILE_SIZE + 1]; // Shared memory for the target block (row or column pivot)
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_start = ty * WORK_PER_THREAD;
    int col_start = tx * WORK_PER_THREAD;
    
    // blk_idx 是這個 block 在 row/column blocks 中的索引, grid size 是 (round, 2)
    int blk_idx = blockIdx.x;
    if (blk_idx == Round) return; // Skip the pivot block itself
    
    // grid 一列給 row blocks, 一列給 column blocks
    int mode = blockIdx.y; // 0: row block, 1: column block
    
    int pivot_row_start = Round * TILE_SIZE + row_start;
    int pivot_col_start = Round * TILE_SIZE + col_start;
    
    int target_row_start, target_col_start;
    
    if (mode == 0) { 
        target_row_start = Round * TILE_SIZE + row_start;
        target_col_start = blk_idx * TILE_SIZE + col_start;
    } else { 
        target_row_start = blk_idx * TILE_SIZE + row_start;
        target_col_start = Round * TILE_SIZE + col_start;
    }
    
    // Load pivot block into shared memory
    #pragma unroll
    for (int i = 0; i < WORK_PER_THREAD; ++i) {
        int r = pivot_row_start + i;
        #pragma unroll
        for (int j = 0; j < WORK_PER_THREAD; j += 4) {
            int c = pivot_col_start + j;
            unsigned long long val = *reinterpret_cast<unsigned long long*>(&d_Dist[r * n + c]);
            pivot[row_start + i][col_start + j + 0] = (unsigned short)(val & 0xFFFF);
            pivot[row_start + i][col_start + j + 1] = (unsigned short)((val >> 16) & 0xFFFF);
            pivot[row_start + i][col_start + j + 2] = (unsigned short)((val >> 32) & 0xFFFF);
            pivot[row_start + i][col_start + j + 3] = (unsigned short)((val >> 48) & 0xFFFF);
        }
    }
    
    // Load target block into shared memory
    #pragma unroll
    for (int i = 0; i < WORK_PER_THREAD; ++i) {
        int r = target_row_start + i;
        #pragma unroll
        for (int j = 0; j < WORK_PER_THREAD; j += 4) {
            int c = target_col_start + j;
            unsigned long long val = *reinterpret_cast<unsigned long long*>(&d_Dist[r * n + c]);
            target[row_start + i][col_start + j + 0] = (unsigned short)(val & 0xFFFF);
            target[row_start + i][col_start + j + 1] = (unsigned short)((val >> 16) & 0xFFFF);
            target[row_start + i][col_start + j + 2] = (unsigned short)((val >> 32) & 0xFFFF);
            target[row_start + i][col_start + j + 3] = (unsigned short)((val >> 48) & 0xFFFF);
        }
    }
    
    __syncthreads(); // Synchronize to make sure the tiles are loaded
    
    // Load target block values into registers
    // Load target[row_start + i][col_start + j] into t_val[i][j] 因為後面會多次使用，所以先放到 register 裡
    // 一次 load 4*4 個元素 (per thread workload)
    int t_val[WORK_PER_THREAD][WORK_PER_THREAD];
    #pragma unroll
    for(int i=0; i<WORK_PER_THREAD; ++i) {
        #pragma unroll
        for(int j=0; j<WORK_PER_THREAD; ++j) {
            t_val[i][j] = target[row_start + i][col_start + j];
        }
    }
    
    // Compute shortest paths
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
    
    // Store results from registers back to global memory
    #pragma unroll
    for (int i = 0; i < WORK_PER_THREAD; ++i) {
        int r = target_row_start + i;
        #pragma unroll
        for (int j = 0; j < WORK_PER_THREAD; j += 4) {
            int c = target_col_start + j;
            unsigned long long val = 0;
            val |= (unsigned long long)t_val[i][j + 0];
            val |= ((unsigned long long)t_val[i][j + 1]) << 16;
            val |= ((unsigned long long)t_val[i][j + 2]) << 32;
            val |= ((unsigned long long)t_val[i][j + 3]) << 48;
            *reinterpret_cast<unsigned long long*>(&d_Dist[r * n + c]) = val;
        }
    }
}

__global__ void phase3_kernel(unsigned short* __restrict__ d_Dist, int n, int Round) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    if (bx == Round || by == Round) return; // Skip the row pivot and column pivot blocks
    
    __shared__ unsigned short row_tile[TILE_SIZE][TILE_SIZE + 1]; 
    __shared__ unsigned short col_tile[TILE_SIZE][TILE_SIZE + 1]; 
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_start = ty * WORK_PER_THREAD;
    int col_start = tx * WORK_PER_THREAD;
    
    // Load row pivot block into shared memory
    int row_row_start = by * TILE_SIZE + row_start;
    int row_col_start = Round * TILE_SIZE + col_start;
    
    #pragma unroll
    for (int i = 0; i < WORK_PER_THREAD; ++i) {
        int r = row_row_start + i;
        #pragma unroll
        for (int j = 0; j < WORK_PER_THREAD; j += 4) {
            int c = row_col_start + j;
            unsigned long long val = *reinterpret_cast<unsigned long long*>(&d_Dist[r * n + c]);
            row_tile[row_start + i][col_start + j + 0] = (unsigned short)(val & 0xFFFF);
            row_tile[row_start + i][col_start + j + 1] = (unsigned short)((val >> 16) & 0xFFFF);
            row_tile[row_start + i][col_start + j + 2] = (unsigned short)((val >> 32) & 0xFFFF);
            row_tile[row_start + i][col_start + j + 3] = (unsigned short)((val >> 48) & 0xFFFF);
        }
    }
    
    // Load column pivot block into shared memory
    int c_row_start = Round * TILE_SIZE + row_start;
    int c_col_start = bx * TILE_SIZE + col_start;
    
    #pragma unroll
    for (int i = 0; i < WORK_PER_THREAD; ++i) {
        int r = c_row_start + i;
        #pragma unroll
        for (int j = 0; j < WORK_PER_THREAD; j += 4) {
            int c = c_col_start + j;
            unsigned long long val = *reinterpret_cast<unsigned long long*>(&d_Dist[r * n + c]);
            col_tile[row_start + i][col_start + j + 0] = (unsigned short)(val & 0xFFFF);
            col_tile[row_start + i][col_start + j + 1] = (unsigned short)((val >> 16) & 0xFFFF);
            col_tile[row_start + i][col_start + j + 2] = (unsigned short)((val >> 32) & 0xFFFF);
            col_tile[row_start + i][col_start + j + 3] = (unsigned short)((val >> 48) & 0xFFFF);
        }
    }
    
    __syncthreads();
    
    int my_row_start = by * TILE_SIZE + row_start;
    int my_col_start = bx * TILE_SIZE + col_start;
    
    int c_val[WORK_PER_THREAD][WORK_PER_THREAD]; // Store current block values in registers
    #pragma unroll
    for (int i = 0; i < WORK_PER_THREAD; ++i) {
        int r = my_row_start + i;
        #pragma unroll
        for (int j = 0; j < WORK_PER_THREAD; j += 4) {
            int c = my_col_start + j;
            unsigned long long val = *reinterpret_cast<unsigned long long*>(&d_Dist[r * n + c]);
            c_val[i][j + 0] = (unsigned short)(val & 0xFFFF);
            c_val[i][j + 1] = (unsigned short)((val >> 16) & 0xFFFF);
            c_val[i][j + 2] = (unsigned short)((val >> 32) & 0xFFFF);
            c_val[i][j + 3] = (unsigned short)((val >> 48) & 0xFFFF);
        }
    }
    
    #pragma unroll
    for (int k = 0; k < TILE_SIZE; ++k) {
        int a[WORK_PER_THREAD];
        int b[WORK_PER_THREAD];
        
        // Load row and column values into registers
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
            unsigned long long val = 0;
            val |= (unsigned long long)c_val[i][j + 0];
            val |= ((unsigned long long)c_val[i][j + 1]) << 16;
            val |= ((unsigned long long)c_val[i][j + 2]) << 32;
            val |= ((unsigned long long)c_val[i][j + 3]) << 48;
            *reinterpret_cast<unsigned long long*>(&d_Dist[r * n + c]) = val;
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s input.bin output.bin\n", argv[0]);
        return 1;
    }

    input(argv[1]);

    int padded_n = ceil_int(n, TILE_SIZE) * TILE_SIZE;
    int round = padded_n / TILE_SIZE;

    unsigned short* d_Dist;
    hipMalloc(&d_Dist, sizeof(unsigned short) * padded_n * padded_n);

    unsigned short* h_padded_Dist = (unsigned short*)malloc(sizeof(unsigned short) * padded_n * padded_n);
    
    // Initialize padded distance matrix with INF
    for (int i = 0; i < padded_n * padded_n; ++i) h_padded_Dist[i] = INF;
    
    // Copy original Dist into padded Dist
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            h_padded_Dist[i * padded_n + j] = Dist[i * n + j];
        }
    }
    
    for (int i = n; i < padded_n; ++i) {
        h_padded_Dist[i * padded_n + i] = 0;
    }

    hipMemcpy(d_Dist, h_padded_Dist, sizeof(unsigned short) * padded_n * padded_n, hipMemcpyHostToDevice);

    dim3 block(BLOCK_DIM, BLOCK_DIM);
    
    for (int r = 0; r < round; ++r) {
        phase1_kernel<<<1, block>>>(d_Dist, padded_n, r);
        
        dim3 grid2(round, 2);
        phase2_kernel<<<grid2, block>>>(d_Dist, padded_n, r);
        
        dim3 grid3(round, round);
        phase3_kernel<<<grid3, block>>>(d_Dist, padded_n, r);
    }

    hipMemcpy(h_padded_Dist, d_Dist, sizeof(unsigned short) * padded_n * padded_n, hipMemcpyDeviceToHost);
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            Dist[i * n + j] = h_padded_Dist[i * padded_n + j];
        }
    }

    output(argv[2]);
    
    free(h_padded_Dist);
    free(Dist);
    hipFree(d_Dist);
    return 0;
}

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    Dist = (unsigned short*)malloc(sizeof(unsigned short) * n * n);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                Dist[i * n + j] = 0;
            } else {
                Dist[i * n + j]= INF;
            }
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0] * n + pair[1]] = (unsigned short)pair[2];
    }
    fclose(file);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "wb");
    int* outBuf = (int*)malloc(sizeof(int) * n * n);
    for (int i = 0; i < n * n; ++i) {
        outBuf[i] = (int)Dist[i];
        if (outBuf[i] >= INF) outBuf[i] = ((1 << 30) - 1); // Restore original INF for output
    }
    fwrite(outBuf, sizeof(int), n * n, outfile);
    free(outBuf);
    fclose(outfile);
}
