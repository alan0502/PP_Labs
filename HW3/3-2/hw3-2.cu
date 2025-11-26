#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#ifndef TILE_SIZE
#define TILE_SIZE 64
#endif

#define WORK_PER_THREAD 4

#define BLOCK_DIM (TILE_SIZE / WORK_PER_THREAD)

const unsigned short INF = 65535;

int n, m;
unsigned short* Dist;

void input(char* inFileName);
void output(char* outFileName);

int ceil_int(int a, int b) { return (a + b - 1) / b; }

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

__global__ void phase2_kernel(unsigned short* __restrict__ d_Dist, int n, int Round) {
    __shared__ unsigned short pivot[TILE_SIZE][TILE_SIZE + 1];
    __shared__ unsigned short target[TILE_SIZE][TILE_SIZE + 1];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_start = ty * WORK_PER_THREAD;
    int col_start = tx * WORK_PER_THREAD;
    
    int blk_idx = blockIdx.x;
    if (blk_idx == Round) return;
    
    int mode = blockIdx.y; 
    
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

__global__ void phase3_kernel(unsigned short* __restrict__ d_Dist, int n, int Round) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
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

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s input.bin output.bin\n", argv[0]);
        return 1;
    }

    input(argv[1]);

    int padded_n = ceil_int(n, TILE_SIZE) * TILE_SIZE;
    int round = padded_n / TILE_SIZE;

    unsigned short* d_Dist;
    cudaMalloc(&d_Dist, sizeof(unsigned short) * padded_n * padded_n);

    unsigned short* h_padded_Dist = (unsigned short*)malloc(sizeof(unsigned short) * padded_n * padded_n);
    
    for (int i = 0; i < padded_n * padded_n; ++i) h_padded_Dist[i] = INF;
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            h_padded_Dist[i * padded_n + j] = Dist[i * n + j];
        }
    }
    
    for (int i = n; i < padded_n; ++i) {
        h_padded_Dist[i * padded_n + i] = 0;
    }

    cudaMemcpy(d_Dist, h_padded_Dist, sizeof(unsigned short) * padded_n * padded_n, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_DIM, BLOCK_DIM);
    
    for (int r = 0; r < round; ++r) {
        phase1_kernel<<<1, block>>>(d_Dist, padded_n, r);
        
        dim3 grid2(round, 2);
        phase2_kernel<<<grid2, block>>>(d_Dist, padded_n, r);
        
        dim3 grid3(round, round);
        phase3_kernel<<<grid3, block>>>(d_Dist, padded_n, r);
    }

    cudaMemcpy(h_padded_Dist, d_Dist, sizeof(unsigned short) * padded_n * padded_n, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            Dist[i * n + j] = h_padded_Dist[i * padded_n + j];
        }
    }

    output(argv[2]);
    
    free(h_padded_Dist);
    free(Dist);
    cudaFree(d_Dist);
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
