#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>

#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// ============================================================
// Global config / host buffers (和 seq-flashattention 類似)
// ============================================================

int B, N, d;
float *h_Q = nullptr;
float *h_K = nullptr;
float *h_V = nullptr;
float *h_O = nullptr;

// 常數 tile 大小 (和 seq-flashattention 一樣)


constexpr int BR = 16;  // block rows // 32 代表每個 block 處理 32 個 query rows，因為一個 warp 有 32 個 thread
constexpr int BC = 32;  // block cols
constexpr int MAX_D = 64;  // 題目 d 只有 32 或 64


// ============================================================
// I/O
// ============================================================

void input(const char *input_filename) {
    FILE *file = fopen(input_filename, "rb");
    if (!file) {
        fprintf(stderr, "Cannot open input file %s\n", input_filename);
        exit(EXIT_FAILURE);
    }

    fread(&B, sizeof(int), 1, file);
    fread(&N, sizeof(int), 1, file);
    fread(&d, sizeof(int), 1, file);

    size_t total = (size_t)B * N * d;
    h_Q = (float *)malloc(total * sizeof(float));
    h_K = (float *)malloc(total * sizeof(float));
    h_V = (float *)malloc(total * sizeof(float));
    h_O = (float *)malloc(total * sizeof(float));

    if (!h_Q || !h_K || !h_V || !h_O) {
        fprintf(stderr, "Host malloc failed\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < B; i++) {
        fread(h_Q + (size_t)i * N * d, sizeof(float), (size_t)N * d, file);
        fread(h_K + (size_t)i * N * d, sizeof(float), (size_t)N * d, file);
        fread(h_V + (size_t)i * N * d, sizeof(float), (size_t)N * d, file);
    }

    memset(h_O, 0x00, total * sizeof(float));
    fclose(file);
}

void output(const char *output_filename) {
    FILE *file = fopen(output_filename, "wb");
    if (!file) {
        fprintf(stderr, "Cannot open output file %s\n", output_filename);
        exit(EXIT_FAILURE);
    }

    size_t total = (size_t)B * N * d;
    fwrite(h_O, sizeof(float), total, file);

    free(h_Q);
    free(h_K);
    free(h_V);
    free(h_O);

    fclose(file);
}

// ============================================================
// Device kernels
// ============================================================

// 初始化 m[i] = -inf, l[i] = 0, O = 0
__global__
void init_m_l_o_kernel(float *m, float *l, float *O, int N, int d) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        m[row] = -FLT_MAX;
        l[row] = 0.0f;
    }

    // 這邊順便把 O 清零
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * d;
    if (idx < total) {
        O[idx] = 0.0f;
    }
}

// 單一步驟：固定一個 key/value block j_block，
// 每個 block 處理一個 query block i_block (BR rows)。
__global__
void flash_step_kernel(const float *__restrict__ Q,
                       const float *__restrict__ K,
                       const float *__restrict__ V,
                       float       *__restrict__ O,
                       float       *__restrict__ m,
                       float       *__restrict__ l,
                       int N, int d,
                       int j_block,
                       float inv_sqrt_d)
{
    // 一個 CUDA block 處理一個 row-block (BR rows)
    // i_block 是從上到下的第幾個 block
    int i_block   = blockIdx.x;
    int row_start = i_block * BR;
    int col_start = j_block * BC;

    int lane = threadIdx.x;           // 0..31，一個 warp 裡的 lane
    int r    = threadIdx.y;           // 0..BR-1，表示 tile 裡的第幾個 row

    int global_row = row_start + r;

    // ---- shared memory: Q_i, K_j, V_j 的 tile ----
    __shared__ float sQ[BR][MAX_D];      // Q tile: BR x d
    __shared__ float sK[MAX_D][BC];      // K tile: d x BC (Transposed)
    __shared__ float sV[BC][MAX_D];      // V tile: BC x d

    // 取出這列目前的 m, l
    float old_m = -FLT_MAX;
    float old_l = 0.0f;
    if (global_row < N) {
        old_m = m[global_row];
        old_l = l[global_row];
    }

    // --------------------------------------------------------
    // 0. 將 Q_i, K_j, V_j tile 搬到 shared memory
    // --------------------------------------------------------
    
    // Load Q: Cooperative loading by warp
    if (global_row < N) {
        const float *q_ptr = Q + (size_t)global_row * d;
        for (int t = lane; t < d; t += 32) {
            sQ[r][t] = q_ptr[t];
        }
    } else {
        for (int t = lane; t < d; t += 32) {
            sQ[r][t] = 0.0f;
        }
    }

    // Load K (Transposed) and V: Cooperative loading by block
    int flat_tid = threadIdx.y * blockDim.x + threadIdx.x;
    int flat_stride = blockDim.x * blockDim.y;

    for (int idx = flat_tid; idx < BC * d; idx += flat_stride) {
        int c = idx / d;
        int t = idx % d;

        int gcol = col_start + c;
        if (gcol < N) {
            sK[t][c] = K[gcol * d + t]; // Transposed
            sV[c][t] = V[gcol * d + t];
        } else {
            sK[t][c] = 0.0f;
            sV[c][t] = 0.0f;
        }
    }

    __syncthreads();

    // --------------------------------------------------------
    // 1. 計算 S_row[c] = Q[row] · K[col]^T / sqrt(d)
    //    Each lane computes one element of S_row (since BC=32)
    // --------------------------------------------------------
    
    float my_sij = 0.0f;
    int c = lane; // This lane corresponds to column c of the tile
    
    float val = 0.0f;
    for (int t = 0; t < d; ++t) {
        val += sQ[r][t] * sK[t][c];
    }
    my_sij = val * inv_sqrt_d;
    
    int gcol = col_start + c;
    if (gcol >= N) {
        my_sij = -INFINITY;
    }

    // --------------------------------------------------------
    // 2. RowMax: tile_m = max_c sij[c]
    //    Warp All-Reduce
    // --------------------------------------------------------
    float tile_m = my_sij;
    for (int offset = 16; offset > 0; offset >>= 1) {
        tile_m = fmaxf(tile_m, __shfl_xor_sync(0xffffffff, tile_m, offset));
    }

    // --------------------------------------------------------
    // 3. MinusMaxAndExp + RowSum
    // --------------------------------------------------------
    float my_pij = 0.0f;
    if (gcol < N) {
        my_pij = expf(my_sij - tile_m);
    }
    
    float tile_l = my_pij;
    for (int offset = 16; offset > 0; offset >>= 1) {
        tile_l += __shfl_xor_sync(0xffffffff, tile_l, offset);
    }

    // --------------------------------------------------------
    // 4. online merge m_i, l_i
    // --------------------------------------------------------
    float mi_new = fmaxf(old_m, tile_m);
    float li_new = expf(old_m - mi_new) * old_l +
                   expf(tile_m - mi_new) * tile_l;

    // --------------------------------------------------------
    // 5. 更新 O 這一列
    // --------------------------------------------------------
    float scale_old = (old_l > 0.0f) ? old_l * expf(old_m - mi_new) : 0.0f;
    float scale_new = expf(tile_m - mi_new);

    if (li_new > 0.0f && global_row < N) {
        float *o_ptr = O + (size_t)global_row * d;

        // Parallelize over d (dim) using warp
        for (int dim = lane; dim < d; dim += 32) {
            float pv = 0.0f;
            // Sum over c (0..BC-1)
            for (int k = 0; k < BC; ++k) {
                // Broadcast pij[k] from lane k
                float val_pij = __shfl_sync(0xffffffff, my_pij, k);
                pv += val_pij * sV[k][dim];
            }

            float old_o = o_ptr[dim];
            float out = (scale_old * old_o + scale_new * pv) / li_new;
            o_ptr[dim] = out;
        }
    }

    // --------------------------------------------------------
    // 6. 寫回這一列的 m, l
    // --------------------------------------------------------
    if (lane == 0 && global_row < N) {
        m[global_row] = mi_new;
        l[global_row] = li_new;
    }
}




// ============================================================
// 在 GPU 上跑一個 batch 的 FlashAttention (對應 seq 的 flash_attention)
// ============================================================

void run_flashattention_batch(const float *d_Q,
                              const float *d_K,
                              const float *d_V,
                              float       *d_O,
                              int N, int d)
{
    // 為這個 batch 準備 m, l
    float *d_m = nullptr;
    float *d_l = nullptr;
    CHECK_CUDA(cudaMalloc(&d_m, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_l, N * sizeof(float)));

    // init m, l, O
    int threads = 256; // 只是為了初始化用的 thread 數量
    int blocks_for_rows = (N + threads - 1) / threads;
    int blocks_for_O    = (N * d + threads - 1) / threads;

    // 為了簡化，直接用較大的 grid，兩個 work 都會覆蓋到
    int blocks = max(blocks_for_rows, blocks_for_O);
    init_m_l_o_kernel<<<blocks, threads>>>(d_m, d_l, d_O, N, d);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // tile 切法
    int tr = (N + BR - 1) / BR;    // row tiles
    int tc = (N + BC - 1) / BC;   // col tiles

    float inv_sqrt_d = 1.0f / sqrtf((float)d);

    for (int j_block = 0; j_block < tc; ++j_block) {
        dim3 grid(tr);          // 每個 block 處理一個 row-block (BR rows)
        //dim3 block(BR);         // 32 threads, 每 thread 對應一個 row
        dim3 block(32, BR);          // x: 32 lanes, y: BR rows → 32*BR threads

        flash_step_kernel<<<grid, block>>>(
            d_Q, d_K, d_V,
            d_O,
            d_m, d_l,
            N, d,
            j_block,
            inv_sqrt_d
        );
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }



    CHECK_CUDA(cudaFree(d_m));
    CHECK_CUDA(cudaFree(d_l));
}

// ============================================================
// main
// ============================================================

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <input_filename> <output_filename>\n", argv[0]);
        return 1;
    }

    input(argv[1]);

    size_t total = (size_t)B * N * d;

    // 將整個 Q, K, V, O 丟到 GPU
    float *d_Q_all = nullptr;
    float *d_K_all = nullptr;
    float *d_V_all = nullptr;
    float *d_O_all = nullptr;

    CHECK_CUDA(cudaMalloc(&d_Q_all, total * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_K_all, total * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_V_all, total * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_O_all, total * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_Q_all, h_Q, total * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K_all, h_K, total * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V_all, h_V, total * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_O_all, 0, total * sizeof(float)));


    // 對每個 batch 跑一次 FlashAttention
    for (int b = 0; b < B; ++b) {
        size_t offset = (size_t)b * N * d;
        const float *d_Q = d_Q_all + offset;
        const float *d_K = d_K_all + offset;
        const float *d_V = d_V_all + offset;
        float       *d_O = d_O_all + offset;

        run_flashattention_batch(d_Q, d_K, d_V, d_O, N, d);
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    printf("(B, N, d): (%d, %d, %d)\n", B, N, d);
    

    // 把 O 拷回 host
    CHECK_CUDA(cudaMemcpy(h_O, d_O_all, total * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_Q_all));
    CHECK_CUDA(cudaFree(d_K_all));
    CHECK_CUDA(cudaFree(d_V_all));
    CHECK_CUDA(cudaFree(d_O_all));

    output(argv[2]);

    return 0;
}
