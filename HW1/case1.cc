#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <algorithm>
#define BOOST_SPREADSORT_MAX_SPLITS 12
#include <boost/sort/spreadsort/float_sort.hpp>
#include <nvtx3/nvToolsExt.h>
#include <numa.h>
#include <sched.h>

using boost::sort::spreadsort::float_sort;

void merge_all_pair_partial(float* d, int n, int partner, int rank, int size, int N,
                                  float* recvbuf, int recvbuf_cap, float* keep) {
    int partner_n = N/size + (partner < N % size ? 1 : 0);

    // 直接交換全部資料
    MPI_Sendrecv(d, n, MPI_FLOAT, partner, 0,
                 recvbuf, partner_n, MPI_FLOAT, partner, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // two-way merge
    if (rank < partner) {
        int i = 0, j = 0, k = 0;
        while (k < n) {
            if (j >= partner_n || (i < n && d[i] <= recvbuf[j])) {
                keep[k++] = d[i++];
            } else {
                keep[k++] = recvbuf[j++];
            }
        }
    } else {
        int i = n - 1, j = partner_n - 1, k = n - 1;
        while (k >= 0) {
            if (j < 0 || (i >= 0 && d[i] >= recvbuf[j])) {
                keep[k--] = d[i--];
            } else {
                keep[k--] = recvbuf[j--];
            }
        }
    }
    std::copy(keep, keep + n, d);
}


// === even phase ===
void even_odd(float* d, int n, int rank, int size, int N, int phase,
              float* recvbuf, float* keep, int recvbuf_cap) {
    if (rank % 2 == 0 && rank + 1 < size) {
        merge_all_pair_partial(d, n, rank + 1, rank, size, N, recvbuf, recvbuf_cap, keep);
    } else if (rank % 2 == 1) {
        merge_all_pair_partial(d, n, rank - 1, rank, size, N, recvbuf, recvbuf_cap, keep);
    }
}

// === odd phase ===
void odd_even(float* d, int n, int rank, int size, int N, int phase,
              float* recvbuf, float* keep, int recvbuf_cap) {
    if (rank % 2 == 1 && rank + 1 < size) {
        merge_all_pair_partial(d, n, rank + 1, rank, size, N, recvbuf, recvbuf_cap, keep);
    } else if (rank % 2 == 0 && rank > 0) {
        merge_all_pair_partial(d, n, rank - 1, rank, size, N, recvbuf, recvbuf_cap, keep);
    }
}

int main(int argc, char **argv)
{
    nvtxRangePush("All");
    setenv("UCX_NET_DEVICES", "ibp3s0:1", 1);
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = atoi(argv[1]);
    const char *const input_filename = argv[2],
               *const output_filename = argv[3];

    int rank_size = N/size + (rank < N % size ? 1 : 0);
    int start     = rank * (N / size) + (rank < N % size ? rank : N % size);

    // === 配置 ===
    int max_chunk = N/size + 1;   // 任何 rank 的最大可能分片

    int cpu = sched_getcpu();
    int numa_node = numa_node_of_cpu(cpu);
    
    float *d       = (float*)numa_alloc_onnode(sizeof(float) * std::max(1, rank_size), numa_node);
    float *recvbuf = (float*)numa_alloc_onnode(sizeof(float) * std::max(1, max_chunk), numa_node);
    float *keep    = (float*)numa_alloc_onnode(sizeof(float) * std::max(1, rank_size), numa_node);

    // === 檔案讀取 ===
    MPI_File input_file, output_file;
    MPI_File_open(MPI_COMM_SELF, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    MPI_File_read_at(input_file,
        sizeof(float) * start,
        d,
        rank_size,
        MPI_FLOAT,
        MPI_STATUS_IGNORE);
    MPI_File_close(&input_file);

    // === 排序 ===
    nvtxRangePush("Sorting");
    float_sort(d, d + rank_size);
    nvtxRangePop();

    // === odd-even sort phases ===
    for (int phase = 0; phase < size/2 + 1; ++phase) {
        even_odd(d, rank_size, rank, size, N, phase, recvbuf, keep, max_chunk);
        odd_even(d, rank_size, rank, size, N, phase, recvbuf, keep, max_chunk);
    }

    // === 檔案輸出 ===
    MPI_File_open(MPI_COMM_SELF, output_filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    MPI_File_write_at(output_file,
        sizeof(float) * start,
        d,
        rank_size,
        MPI_FLOAT,
        MPI_STATUS_IGNORE);
    MPI_File_close(&output_file);

    numa_free(d, sizeof(float) * (rank_size));
    numa_free(recvbuf, sizeof(float) * std::max(1, max_chunk));
    numa_free(keep, sizeof(float) * std::max(1, rank_size));

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    nvtxRangePop();
    return 0;
}
