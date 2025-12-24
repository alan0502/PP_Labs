#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>

#include <hip/hip_runtime.h>

#define CHECK_HIP(call)                                                      \
    do {                                                                     \
        hipError_t err = call;                                              \
        if (err != hipSuccess) {                                            \
            fprintf(stderr, "HIP error %s:%d: %s\n",                        \
                    __FILE__, __LINE__, hipGetErrorString(err));           \
            exit(5);                                                         \
        }                                                                    \
    } while (0)

#define CHECK_HIP_EXIT(call, code)                                           \
    do {                                                                     \
        hipError_t err = call;                                              \
        if (err != hipSuccess) {                                            \
            fprintf(stderr, "HIP error %s:%d: %s\n",                        \
                    __FILE__, __LINE__, hipGetErrorString(err));           \
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

constexpr int BR     = 16;   // block rows
constexpr int BC     = 32;   // block cols (per warp)
constexpr int MAX_D  = 64;   // d is 32 or 64

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
    size_t total_floats      = (size_t)B * batch_size_floats;

    h_Q = (float *)malloc(total_floats * sizeof(float));
    h_K = (float *)malloc(total_floats * sizeof(float));
    h_V = (float *)malloc(total_floats * sizeof(float));
    h_O = (float *)malloc(total_floats * sizeof(float));

    if (!h_Q || !h_K || !h_V || !h_O) {
        fprintf(stderr, "Host malloc failed\n");
        exit(4);
    }

    // padding 區先清 0
    memset(h_Q, 0, total_floats * sizeof(float));
    memset(h_K, 0, total_floats * sizeof(float));
    memset(h_V, 0, total_floats * sizeof(float));
    memset(h_O, 0, total_floats * sizeof(float));

    // 逐 batch 讀 N*d 的真實資料到各自 slice 前面
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
    size_t total_rows     = (size_t)N;
    size_t total_elements = (size_t)N * d;

    size_t idx    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
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
                       int j_block,
                       float inv_sqrt_d)
{
    int i_block   = blockIdx.x;
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
    const float *q_ptr = Q + (size_t)global_row * d;
    for (int t = lane; t < d; t += 32) {
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

    // 1. dot product -> score
    float my_sij = 0.0f;
    int   c      = lane;

    float val = 0.0f;
    for (int t = 0; t < d; ++t) {
        val += sQ[r][t] * sK[t][c];
    }
    my_sij = val * inv_sqrt_d;

    int gcol = col_start + c;
    if (gcol >= N) {
        my_sij = -INFINITY;
    }

    // 2. row max (tile_m) with warp(32) reduction
    float tile_m = my_sij;
    for (int offset = 16; offset > 0; offset >>= 1) {
        tile_m = fmaxf(tile_m,
                       __shfl_xor(tile_m, offset, 64));
    }

    // 3. exp & row sum
    float my_pij = 0.0f;
    if (gcol < N) {
        my_pij = __expf(my_sij - tile_m);
    }

    float tile_l = my_pij;
    for (int offset = 16; offset > 0; offset >>= 1) {
        tile_l += __shfl_xor(tile_l, offset, 32);
    }

    // 4. online merge (m_i, l_i)
    float mi_new = fmaxf(old_m, tile_m);
    float li_new = __expf(old_m - mi_new) * old_l +
                   __expf(tile_m - mi_new) * tile_l;

    // 5. 更新 O row
    float scale_old = (old_l > 0.0f) ? old_l * __expf(old_m - mi_new) : 0.0f;
    float scale_new = __expf(tile_m - mi_new);

    float *o_ptr = O + (size_t)global_row * d;

    for (int dim = lane; dim < d; dim += 32) {
        float pv = 0.0f;
        for (int k = 0; k < BC; ++k) {
            // broadcast pij from lane k, 只在 32-lane warp 內
            float val_pij = __shfl(my_pij, k, 32);
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

    hipLaunchKernelGGL(init_m_l_o_kernel,
                       dim3(blocks), dim3(threads), 0, 0,
                       d_m, d_l, d_O, padded_N, d);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());

    int tr = (padded_N + BR - 1) / BR;
    int tc = (padded_N + BC - 1) / BC;

    float inv_sqrt_d = 1.0f / sqrtf((float)d);

    dim3 block(32, BR);

    for (int j_block = 0; j_block < tc; ++j_block) {
        dim3 grid(tr);

        hipLaunchKernelGGL(flash_step_kernel,
                           grid, block, 0, 0,
                           d_Q, d_K, d_V,
                           d_O,
                           d_m, d_l,
                           N, d,
                           j_block,
                           inv_sqrt_d);
        CHECK_HIP(hipGetLastError());
        CHECK_HIP(hipDeviceSynchronize());
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

    CHECK_HIP_EXIT(hipMalloc(&d_Q_all, total_floats * sizeof(float)), 10);
    CHECK_HIP_EXIT(hipMalloc(&d_K_all, total_floats * sizeof(float)), 10);
    CHECK_HIP_EXIT(hipMalloc(&d_V_all, total_floats * sizeof(float)), 10);
    CHECK_HIP_EXIT(hipMalloc(&d_O_all, total_floats * sizeof(float)), 10);
    CHECK_HIP_EXIT(hipMalloc(&d_m,     (size_t)padded_N * sizeof(float)), 10);
    CHECK_HIP_EXIT(hipMalloc(&d_l,     (size_t)padded_N * sizeof(float)), 10);

    // Copy all batches Q/K/V to device in one shot
    CHECK_HIP_EXIT(hipMemcpy(d_Q_all, h_Q, total_floats * sizeof(float),
                             hipMemcpyHostToDevice), 11);
    CHECK_HIP_EXIT(hipMemcpy(d_K_all, h_K, total_floats * sizeof(float),
                             hipMemcpyHostToDevice), 11);
    CHECK_HIP_EXIT(hipMemcpy(d_V_all, h_V, total_floats * sizeof(float),
                             hipMemcpyHostToDevice), 11);

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

    CHECK_HIP_EXIT(hipDeviceSynchronize(), 12);

    // Copy all outputs back
    CHECK_HIP_EXIT(hipMemcpy(h_O, d_O_all, total_floats * sizeof(float),
                             hipMemcpyDeviceToHost), 11);

    printf("(B, N, d): (%d, %d, %d)\n", B, N, d);

    // Write outputs (only first N rows of each batch)
    output(argv[2]);

    // Free device & host
    CHECK_HIP(hipFree(d_Q_all));
    CHECK_HIP(hipFree(d_K_all));
    CHECK_HIP(hipFree(d_V_all));
    CHECK_HIP(hipFree(d_O_all));
    CHECK_HIP(hipFree(d_m));
    CHECK_HIP(hipFree(d_l));

    free(h_Q);
    free(h_K);
    free(h_V);
    free(h_O);

    return 0;
}
