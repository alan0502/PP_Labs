#include <cstdio>
#include <cstdlib>
#include <vector>
#include <mpi.h>
#include <algorithm>
#include <boost/sort/spreadsort/float_sort.hpp>
#include <boost/sort/spreadsort/spreadsort.hpp>
#include <nvtx3/nvToolsExt.h>

using boost::sort::spreadsort::float_sort;

// === partial merge function ===
void merge_all_pair_partial(std::vector<float>& d, int partner, int rank) {
    int n = d.size();

    // 交換大小
    int partner_n = 0;
    MPI_Sendrecv(&n, 1, MPI_INT, partner, 0,
                 &partner_n, 1, MPI_INT, partner, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // 交換資料
    std::vector<float> partner_buf(partner_n);
    MPI_Sendrecv(d.data(), n, MPI_FLOAT, partner, 1,
                 partner_buf.data(), partner_n, MPI_FLOAT, partner, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    nvtxRangePush("Merging");

    // === 部分 merge ===
    std::vector<float> keep;
    keep.reserve(n);

    if (rank < partner) {
        // 左邊 rank：保留最小的 n 個
        int i = 0, j = 0;
        while ((int)keep.size() < n) {
            if (i < (int)d.size() && (j >= (int)partner_buf.size() || d[i] <= partner_buf[j])) {
                keep.push_back(d[i++]);
            } else {
                keep.push_back(partner_buf[j++]);
            }
        }
        d.swap(keep);
    } else {
        // 右邊 rank：保留最大的 n 個
        int i = d.size() - 1, j = partner_buf.size() - 1;
        while ((int)keep.size() < n) {
            if (i >= 0 && (j < 0 || d[i] >= partner_buf[j])) {
                keep.push_back(d[i--]);
            } else {
                keep.push_back(partner_buf[j--]);
            }
        }
        std::reverse(keep.begin(), keep.end()); // 從大到小挑，要反轉
        d.swap(keep);
    }
    nvtxRangePop();
}

// === even phase ===
void even_odd(std::vector<float>& d, int rank, int size) {
    if (rank % 2 == 0 && rank + 1 < size) {
        merge_all_pair_partial(d, rank + 1, rank);
    } else if (rank % 2 == 1) {
        merge_all_pair_partial(d, rank - 1, rank);
    }
}

// === odd phase ===
void odd_even(std::vector<float>& d, int rank, int size) {
    if (rank % 2 == 1 && rank + 1 < size) {
        merge_all_pair_partial(d, rank + 1, rank);
    } else if (rank % 2 == 0 && rank > 0) {
        merge_all_pair_partial(d, rank - 1, rank);
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
    //printf("rank %d of %d processes, N=%d\n", rank, size, N);

    int rank_size = N/size + (rank < N % size ? 1 : 0);
    int start = rank * (N / size) + (rank < N % size ? rank : N % size);

   

    std::vector<float> d(rank_size);

    

    MPI_File input_file, output_file;
    MPI_File_open(MPI_COMM_SELF, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    MPI_File_read_at(input_file,
        sizeof(float) * start,
        d.data(),
        rank_size,
        MPI_FLOAT,
        MPI_STATUS_IGNORE);
    MPI_File_close(&input_file);

    //std::sort(d.begin(), d.end());
    nvtxRangePush("Sorting");
    float_sort(d.begin(), d.end());
    nvtxRangePop();
    //boost::sort::spreadsort::spreadsort(d.begin(), local_chunk.end());

    //for (int i = 0; i < rank_size; i++) {
    //    printf("rank %d will write data[%d] = %f\n", rank, start + i, d[i]);
    //}
    //printf("rank %d wrote %d floats starting at index %d\n", rank, rank_size, start);
    //printf("%d", N);
    if(N > 100'000'000){
        //printf("Too large N, skip sorting\n");
        for (int phase = 0; phase < size/2; ++phase) {
            even_odd(d, rank, size);
            odd_even(d, rank, size);
        }
    }
    else{
        for (int phase = 0; phase < size/2 + 1; ++phase) {
            even_odd(d, rank, size);
            odd_even(d, rank, size);
        }
    }
    MPI_File_open(MPI_COMM_SELF, output_filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    MPI_File_write_at(output_file,
        sizeof(float) * start,
        d.data(),
        rank_size,
        MPI_FLOAT,
        MPI_STATUS_IGNORE);
    MPI_File_close(&output_file);
    // === 收集結果到 rank 0 ===
    //int local_n = d.size();
    //std::vector<int> counts(size), displs(size);

    //MPI_Gather(&local_n, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    // rank 0 計算 displs
    //int total_n = 0;
    //if (rank == 0) {
    //    displs[0] = 0;
    //    for (int i = 0; i < size; i++) {
    //        if (i > 0) displs[i] = displs[i-1] + counts[i-1];
    //    }
    //    for (int i = 0; i < size; i++) total_n += counts[i];
    //}

    //std::vector<float> all_data;
    //if (rank == 0) all_data.resize(total_n);

    //MPI_Gatherv(d.data(), local_n, MPI_FLOAT,
    //            all_data.data(), counts.data(), displs.data(),
    //            MPI_FLOAT, 0, MPI_COMM_WORLD);

    //if (rank == 0) {
    //    printf("\n=== Final sorted result ===\n");
    //    for (int i = 0; i < 100; i++) {
    //        printf("%f ", all_data[i]);
    //        if ((i+1) % 8 == 0) printf("\n"); // 每8個換行，美觀一點
    //    }
    //    printf("\n");
    //}

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    nvtxRangePop();
    return 0;
}
