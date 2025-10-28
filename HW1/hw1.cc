#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <algorithm>
#include <boost/sort/spreadsort/float_sort.hpp>
#include <nvtx3/nvToolsExt.h>
#include <numa.h>
#include <sched.h>

using boost::sort::spreadsort::float_sort;

void merge_all_pair_partial(float* d, int n, int partner, int rank, int size, int N,
                            float* recvbuf, int recvbuf_cap, float* keep) {
    int partner_n = N/size + (partner < N % size ? 1 : 0);
    if (n == 0 || partner_n == 0) return;

    float my_min = d[0];
    float my_max = d[n-1];
    float partner_val;

    // Boundary check
    bool need_exchange = true;
    if (rank < partner) {
        MPI_Sendrecv(&my_max, 1, MPI_FLOAT, partner, 0,
                     &partner_val, 1, MPI_FLOAT, partner, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (my_max <= partner_val) need_exchange = false;
    } else {
        MPI_Sendrecv(&my_min, 1, MPI_FLOAT, partner, 0,
                     &partner_val, 1, MPI_FLOAT, partner, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (partner_val <= my_min) need_exchange = false;
    }
    if (!need_exchange) return;

    // Binary Search
    nvtxRangePush("Binary Search");
    int send_size = 0;
    float *sendbuf = nullptr;

    if (rank < partner) {
        float partner_min = partner_val;
        int cut_idx = (int)(std::upper_bound(d, d+n, partner_min) - d);
        send_size = n - cut_idx;
        if (send_size > 0) sendbuf = d + cut_idx;
    } else {
        float partner_max = partner_val;
        int cut_idx = (int)(std::upper_bound(d, d+n, partner_max) - d);
        send_size = cut_idx;
        if (send_size > 0) sendbuf = d;
    }
    nvtxRangePop();

    // Transfer sizes
    int recv_size = 0;
    MPI_Sendrecv(&send_size, 1, MPI_INT, partner, 1,
                 &recv_size, 1, MPI_INT, partner, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    if (recv_size > recvbuf_cap) {
        fprintf(stderr, "[rank %d] recv_size=%d > cap=%d (partner=%d)\n",
                rank, recv_size, recvbuf_cap, partner);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Transfer data
    float *send_ptr = (send_size > 0) ? sendbuf : d;
    float *recv_ptr = (recv_size > 0) ? recvbuf : d;

    MPI_Sendrecv(send_ptr, send_size, MPI_FLOAT, partner, 2,
                 recv_ptr, recv_size, MPI_FLOAT, partner, 2,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Merge
    nvtxRangePush("Merge");
    if (rank < partner) {
        // Left rank
        int i = 0, j = 0, k = 0;
        while (k < n) {
            if (j >= recv_size || (i < n && d[i] <= recvbuf[j])) {
                keep[k++] = d[i++];
            } else {
                keep[k++] = recvbuf[j++];
            }
        }
    } else {
        // Right rank
        int i = n - 1, j = recv_size - 1, k = n - 1;
        while (k >= 0) {
            if (j < 0 || (i >= 0 && d[i] >= recvbuf[j])) {
                keep[k--] = d[i--];
            } else {
                keep[k--] = recvbuf[j--];
            }
        }
    }
    nvtxRangePop();

    // Copy back
    std::copy(keep, keep + n, d);
}

// even phase
void even_odd(float* d, int n, int rank, int size, int N, int phase,
              float* recvbuf, float* keep, int recvbuf_cap) {
    if (rank % 2 == 0 && rank + 1 < size) {
        merge_all_pair_partial(d, n, rank + 1, rank, size, N, recvbuf, recvbuf_cap, keep);
    } else if (rank % 2 == 1) {
        merge_all_pair_partial(d, n, rank - 1, rank, size, N, recvbuf, recvbuf_cap, keep);
    }
}

// odd phase
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
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = atoi(argv[1]);
    const char *const input_filename = argv[2],
               *const output_filename = argv[3];

    int rank_size = N/size + (rank < N % size ? 1 : 0);
    int start     = rank * (N / size) + (rank < N % size ? rank : N % size);

    int max_chunk = N/size + 1; 

    int cpu = sched_getcpu();
    int numa_node = numa_node_of_cpu(cpu);
    
    // Data Arrays
    float *d       = (float*)numa_alloc_onnode(sizeof(float) * std::max(1, rank_size), numa_node);
    float *recvbuf = (float*)numa_alloc_onnode(sizeof(float) * std::max(1, max_chunk), numa_node);
    float *keep    = (float*)numa_alloc_onnode(sizeof(float) * std::max(1, rank_size), numa_node);

    MPI_File input_file, output_file;
    // File Input
    MPI_File_open(MPI_COMM_SELF, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    MPI_File_read_at(input_file,
        sizeof(float) * start,
        d,
        rank_size,
        MPI_FLOAT,
        MPI_STATUS_IGNORE);
    MPI_File_close(&input_file);

    nvtxRangePush("Sorting");
    float_sort(d, d + rank_size);
    nvtxRangePop();

    // Sort and Merge
    for (int phase = 0; phase < size/2 + 1; ++phase) {
        even_odd(d, rank_size, rank, size, N, phase, recvbuf, keep, max_chunk);
        odd_even(d, rank_size, rank, size, N, phase, recvbuf, keep, max_chunk);
    }

    // File Output
    MPI_File_open(MPI_COMM_SELF, output_filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    MPI_File_write_at(output_file,
        sizeof(float) * start,
        d,
        rank_size,
        MPI_FLOAT,
        MPI_STATUS_IGNORE);
    MPI_File_close(&output_file);

    // Free Memory
    numa_free(d, sizeof(float) * (rank_size));
    numa_free(recvbuf, sizeof(float) * std::max(1, max_chunk));
    numa_free(keep, sizeof(float) * std::max(1, rank_size));

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    nvtxRangePop();
    return 0;
}