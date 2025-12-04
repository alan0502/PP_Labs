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
            exit(5);                                                         \
        }                                                                    \
    } while (0)

#define CHECK_CUDA_EXIT(call, code)                                          \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(code);                                                      \
        }                                                                    \
    } while (0)

// ============================================================
// Global config / host buffers
// ============================================================

int B, N, d;
float *h_Q = nullptr;
float *h_K = nullptr;
float *h_V = nullptr;
float *h_O = nullptr;

constexpr int BR = 16;   // block rows
constexpr int BC = 32;   // block cols
constexpr int MAX_D = 64; // d is 32 or 64

// ============================================================
// I/O: 一次讀完 / 一次寫完 （沒有 streaming）
// ============================================================

void input(const char *input_filename) {
    FILE *file = fopen(input_filename, "rb");
    if (!file) {
        fprintf(stderr, "Cannot open input file %s\n", input_filename);
        exit(2);
    }

    fread(&B, sizeof(int), 1, file);
    fread(&N, sizeof(int), 1, file);
    fread(&d, sizeof(int), 1, file);

    int padded_N = (N + 31) / 32 * 32;
    size_t batch_size_floats = (size_t)padded_N * d;
    size_t total_floats = (size_t)B * batch_size_floats;

    h_Q = (float *)malloc(total_floats * sizeof(float));
    h_K = (float *)malloc(total_floats * sizeof(float));
    h_V = (float *)malloc(total_floats * sizeof(float));
    h_O = (float *)malloc(total_floats * sizeof(float));

    if (!h_Q || !h_K || !h_V || !h_O) {
        fprintf(stderr, "Host malloc failed\n");
        exit(4);
    }

    // 先把 padding 區清 0
    memset(h_Q, 0, total_floats * sizeof(float));
    memset(h_K, 0, total_floats * sizeof(float));
    memset(h_V, 0, total_floats * sizeof(float));
    memset(h_O, 0, total_floats * sizeof(float));

    // 逐 batch 讀入真實的 N*d 資料到每個 padded slice 的前面
    for (int b = 0; b < B; ++b) {
        float *q_ptr = h_Q + (size_t)b * batch_size_floats;
        float *k_ptr = h_K + (size_t)b * batch_size_floats;
        float *v_ptr = h_V + (size_t)b * batch_size_floats;

        fread(q_ptr, sizeof(float), (size_t)N * d, file);
        fread(k_ptr, sizeof(float), (size_t)N * d, file);
        fread(v_ptr, sizeof(float), (size_t)N * d, file);
    }

    fclose(file);
}

void output(const char *output_filename) {
    FILE *file = fopen(output_filename, "wb");
    if (!file) {
        fprintf(stderr, "Cannot open output file %s\n", output_filename);
        exit(3);
    }

    int padded_N = (N + 31) / 32 * 32;
    size_t batch_size_floats = (size_t)padded_N * d;

    // 逐 batch 把前 N*d 的結果寫回檔案（padding 不寫）
    for (int b = 0; b < B; ++b) {
        float *o_ptr = h_O + (size_t)b * batch_size_floats;
        fwrite(o_ptr, sizeof(float), (size_t)N * d, file);
    }

    fclose(file);
}

// ============================================================
// Device kernels
// ============================================================

// 初始化 m[i] = -inf, l[i] = 0, O = 0 （grid-stride loop）
__global__
void init_m_l_o_kernel(float *m, float *l, float *O, int N, int d) {
    size_t total_rows = (size_t)N;
    size_t total_elements = (size_t)N * d;

    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;

    for (size_t i = idx; i < total_rows; i += stride) {
        m[i] = -FLT_MAX;
        l[i] = 0.0f;
    }

    for (size_t i = idx; i < total_elements; i += stride) {
        O[i] = 0.0f;
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
                       int j_block, // current key/value block index
                       float inv_sqrt_d)
{
    int i_block   = blockIdx.x; // 這是第幾個 query block
    int row_start = i_block * BR;
    int col_start = j_block * BC;

    int lane = threadIdx.x; // 0..31
    int r    = threadIdx.y; // 0..BR-1

    int global_row = row_start + r;

    __shared__ float sQ[BR][MAX_D + 1];
    __shared__ float sK[MAX_D][BC + 1];
    __shared__ float sV[BC][MAX_D + 1];

    // 取出這列目前的 m, l
    float old_m = m[global_row];
    float old_l = l[global_row];

    // Load Q row into shared
    const float *q_ptr = Q + (size_t)global_row * d; // q_ptr 指向 Q 的這一個 row
    for (int t = lane; t < d; t += 32) { // d 是 32 或 64，代表每個 row 的維度
        sQ[r][t] = q_ptr[t];
    }

    // Load K (transposed) & V into shared
    int flat_tid    = threadIdx.y * blockDim.x + threadIdx.x;
    int flat_stride = blockDim.x * blockDim.y;

    for (int idx = flat_tid; idx < BC * d; idx += flat_stride) {
        int c = idx / d;
        int t = idx % d;

        int gcol = col_start + c;
        sK[t][c] = K[(size_t)gcol * d + t];
        sV[c][t] = V[(size_t)gcol * d + t];
    }

    __syncthreads();

    // dot product -> score
    float my_sij = 0.0f;
    int   c      = lane;

    // Q 的 row 與 K 的 col 做 dot product 
    // 32 個 lane (threads) 平行計算同一 col
    float val = 0.0f;
    for (int t = 0; t < d; ++t) {
        val += sQ[r][t] * sK[t][c]; 
    }
    my_sij = val * inv_sqrt_d;

    int gcol = col_start + c;
    if (gcol >= N) {
        my_sij = -INFINITY;
    }

    // row max (tile_m)
    // 因為是同一個 warp，可以用 shuffle 來做 reduction (__shfl_xor_sync)
    // mask 0xffffffff 代表 warp 裡所有 lane 都參與
    // warp shuffle 不用 __syncthreads() 和 shared memory
    // 各自 lane 用 dot product 的結果做 max reduction
    float tile_m = my_sij;
    for (int offset = 16; offset > 0; offset >>= 1) {
        tile_m = fmaxf(tile_m, __shfl_xor_sync(0xffffffff, tile_m, offset));
    }

    // exp & row sum
    // softmax 的分子 (每個 lane 計算自己的 p_ij)
    float my_pij = 0.0f;
    if (gcol < N) {
        my_pij = __expf(my_sij - tile_m);
    }

    // row sum reduction，算 Σ exp(s_i - tile_m)
    // 也要用 shuffle 來做 reduction
    // 每個 lane 算自己的 exp(s_ij - tile_m) 然後加總
    float tile_l = my_pij;
    for (int offset = 16; offset > 0; offset >>= 1) {
        tile_l += __shfl_xor_sync(0xffffffff, tile_l, offset);
    }

    // online merge (m_i, l_i)，要用到 old_m, old_l, tile_m, tile_l
    // new_m = max(old_m, tile_m)
    // new_l = exp(old_m - new_m) * old_l + exp(tile_m - new_m) * tile_l
    float mi_new = fmaxf(old_m, tile_m);
    float li_new = expf(old_m - mi_new) * old_l +
                   __expf(tile_m - mi_new) * tile_l;

    // 5. 更新 O row
    float scale_old = (old_l > 0.0f) ? old_l * __expf(old_m - mi_new) : 0.0f;
    float scale_new = __expf(tile_m - mi_new);

    float *o_ptr = O + (size_t)global_row * d;

    for (int dim = lane; dim < d; dim += 32) {
        float pv = 0.0f;
        for (int k = 0; k < BC; ++k) {
            float val_pij = __shfl_sync(0xffffffff, my_pij, k);
            pv += val_pij * sV[k][dim];
        }

        float old_o = o_ptr[dim];
        float out   = (scale_old * old_o + scale_new * pv) / li_new;
        o_ptr[dim]  = out;
    }

    // 6. 寫回 m, l
    if (lane == 0) {
        m[global_row] = mi_new;
        l[global_row] = li_new;
    }
}

// ============================================================
// Run one batch (size N) on GPU, with padded_N for tiling
// ============================================================

void run_flashattention_batch(const float *d_Q,
                              const float *d_K,
                              const float *d_V,
                              float       *d_O,
                              float       *d_m,
                              float       *d_l,
                              int N, int d,
                              int padded_N)
{
    int threads = 256;
    int blocks  = 1024; // fixed blocks for grid-stride init

    init_m_l_o_kernel<<<blocks, threads>>>(d_m, d_l, d_O, padded_N, d);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    int tr = (padded_N + BR - 1) / BR;
    int tc = (padded_N + BC - 1) / BC;

    float inv_sqrt_d = 1.0f / sqrtf((float)d);

    dim3 block(32, BR);

    for (int j_block = 0; j_block < tc; ++j_block) {
        dim3 grid(tr);

        flash_step_kernel<<<grid, block>>>(
            d_Q, d_K, d_V,
            d_O,
            d_m, d_l,
            N, d,
            j_block,
            inv_sqrt_d
        );
        CHECK_CUDA(cudaGetLastError());
    }
}

// ============================================================
// main (無 streaming I/O 版本)
// ============================================================

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <input_filename> <output_filename>\n", argv[0]);
        return 1;
    }

    input(argv[1]);

    int padded_N = (N + 31) / 32 * 32;
    size_t batch_size_floats = (size_t)padded_N * d;
    size_t total_floats      = (size_t)B * batch_size_floats;

    // Allocate device memory for ALL batches at once
    float *d_Q_all = nullptr;
    float *d_K_all = nullptr;
    float *d_V_all = nullptr;
    float *d_O_all = nullptr;
    float *d_m     = nullptr;   // per-batch workspace: size padded_N
    float *d_l     = nullptr;

    CHECK_CUDA_EXIT(cudaMalloc(&d_Q_all, total_floats * sizeof(float)), 10);
    CHECK_CUDA_EXIT(cudaMalloc(&d_K_all, total_floats * sizeof(float)), 10);
    CHECK_CUDA_EXIT(cudaMalloc(&d_V_all, total_floats * sizeof(float)), 10);
    CHECK_CUDA_EXIT(cudaMalloc(&d_O_all, total_floats * sizeof(float)), 10);
    CHECK_CUDA_EXIT(cudaMalloc(&d_m,     (size_t)padded_N * sizeof(float)), 10);
    CHECK_CUDA_EXIT(cudaMalloc(&d_l,     (size_t)padded_N * sizeof(float)), 10);

    // Copy all batches Q/K/V to device in one shot
    CHECK_CUDA_EXIT(cudaMemcpy(d_Q_all, h_Q, total_floats * sizeof(float),
                               cudaMemcpyHostToDevice), 11);
    CHECK_CUDA_EXIT(cudaMemcpy(d_K_all, h_K, total_floats * sizeof(float),
                               cudaMemcpyHostToDevice), 11);
    CHECK_CUDA_EXIT(cudaMemcpy(d_V_all, h_V, total_floats * sizeof(float),
                               cudaMemcpyHostToDevice), 11);

    // For each batch, run flashattention on its padded slice
    for (int b = 0; b < B; ++b) {
        size_t offset = (size_t)b * batch_size_floats;

        const float *d_Q = d_Q_all + offset;
        const float *d_K = d_K_all + offset;
        const float *d_V = d_V_all + offset;
        float       *d_O = d_O_all + offset;

        run_flashattention_batch(d_Q, d_K, d_V,
                                 d_O, d_m, d_l,
                                 N, d, padded_N);
    }

    CHECK_CUDA_EXIT(cudaDeviceSynchronize(), 12);

    // Copy all outputs back
    CHECK_CUDA_EXIT(cudaMemcpy(h_O, d_O_all, total_floats * sizeof(float),
                               cudaMemcpyDeviceToHost), 11);

    printf("(B, N, d): (%d, %d, %d)\n", B, N, d);

    // Write outputs (only first N rows of each batch)
    output(argv[2]);

    // Free device & host
    CHECK_CUDA(cudaFree(d_Q_all));
    CHECK_CUDA(cudaFree(d_K_all));
    CHECK_CUDA(cudaFree(d_V_all));
    CHECK_CUDA(cudaFree(d_O_all));
    CHECK_CUDA(cudaFree(d_m));
    CHECK_CUDA(cudaFree(d_l));

    free(h_Q);
    free(h_K);
    free(h_V);
    free(h_O);

    return 0;
}
