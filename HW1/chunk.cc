#include <cstdio>
#include <cstdlib>
#include <vector>
#include <mpi.h>
#include <algorithm>
#include <boost/sort/spreadsort/float_sort.hpp>
#include <boost/sort/spreadsort/spreadsort.hpp>
#include <nvtx3/nvToolsExt.h>

using boost::sort::spreadsort::float_sort;

void merge_all_pair_partial(std::vector<float>& d,
                            int partner, int rank, int size, int N, int tag, int phase) {
    int n = d.size();
    int partner_n = N/size + (partner < N % size ? 1 : 0);
    int num_blocks = 2;
    int block_sz = (n + num_blocks - 1) / num_blocks;
    int block_sz_partner = (partner_n + num_blocks - 1) / num_blocks;

    if (n == 0 || partner_n == 0) return;

    float my_min = d.front();
    float my_max = d.back();
    float partner_val;

    if (rank < partner) {
        // 左 rank：傳自己的最大值，收右邊的最小值
        MPI_Sendrecv(&my_max, 1, MPI_FLOAT, partner, 999,
                     &partner_val, 1, MPI_FLOAT, partner, 999,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (my_max <= partner_val) {
            // 已經有序 → 不需要交換
            return;
        }
    } else {
        // 右 rank：傳自己的最小值，收左邊的最大值
        MPI_Sendrecv(&my_min, 1, MPI_FLOAT, partner, 999,
                     &partner_val, 1, MPI_FLOAT, partner, 999,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (partner_val <= my_min) {
            // 已經有序 → 不需要交換
            return;
        }
    }

    std::vector<MPI_Request> send_reqs(num_blocks), recv_reqs(num_blocks);
    std::vector<std::vector<float>> my_blocks(num_blocks), partner_blocks(num_blocks);

    if (rank < partner) {
        // ===== 左 rank：由右往左傳，保留最小 n =====
        int end = n;
        for(int b = 0; b < num_blocks; b++) {
            // sz_my: 這一輪實際要切的 block 大小 ， start 就是這個 block 的起始 index
            int sz_my = std::min(block_sz, end);
            int start = end - sz_my;

            my_blocks[b] = std::vector<float>(d.begin() + start, d.begin() + end);
            end = start;

            int sz_partner = std::min(block_sz, partner_n - b*block_sz);
            partner_blocks[b].resize(sz_partner);

            if(rank == 0 && partner == 1) {
                printf("rank %d sending block %d of size %d to rank %d in phase %d in tag %d\n", rank, b, sz_my, partner, phase, tag);
                //printf("rank %d receiving block %d of size %d from rank %d in phase %d in tag %d\n", rank, b, sz_partner, partner, phase, tag);
                fflush(stdout);
            }

            MPI_Isend(my_blocks[b].data(), sz_my, MPI_FLOAT, partner, b, MPI_COMM_WORLD, &send_reqs[b]);
            MPI_Irecv(partner_blocks[b].data(), sz_partner, MPI_FLOAT, partner, b, MPI_COMM_WORLD, &recv_reqs[b]);
        }
        std::vector<float> keep;
        keep.reserve(n);

        for (int b = 0; b < num_blocks && (int)keep.size() < n; b++) {
            MPI_Wait(&recv_reqs[b], MPI_STATUS_IGNORE); // wait for partner's block 

            int recv_sz = partner_blocks[b].size();

            if (recv_sz > 0 && rank == 0 && partner == 1) {
                printf("[rank %d] phase=%d tag=%d received block %d size=%d [%f ... %f] from rank %d\n",
                    rank, phase, tag, b, recv_sz,
                    partner_blocks[b].front(), partner_blocks[b].back(),
                    partner);
                fflush(stdout);
}
            // merge keep[i] and d
            std::vector<float> cand;
            int i = 0, j = 0;
            while (i < (int)my_blocks[b].size() || j < (int)partner_blocks[b].size()) {
                if (i < (int)my_blocks[b].size() &&
                    (j >= (int)partner_blocks[b].size() || my_blocks[b][i] <= partner_blocks[b][j])) {
                    cand.push_back(my_blocks[b][i++]);
                } else {
                    cand.push_back(partner_blocks[b][j++]);
                }
            }

            // keep = merge_truncate(keep, cand, n)
            std::vector<float> new_keep;
            int a = 0, c = 0;
            while ((int)new_keep.size() < n &&
                   (a < (int)keep.size() || c < (int)cand.size())) {
                if (a < (int)keep.size() &&
                    (c >= (int)cand.size() || keep[a] <= cand[c]))
                    new_keep.push_back(keep[a++]);
                else
                    new_keep.push_back(cand[c++]);
            }
            keep.swap(new_keep);
        }

        MPI_Waitall(num_blocks, send_reqs.data(), MPI_STATUSES_IGNORE);
        d.swap(keep);
    }
    else {
        // ===== 右 rank：由左往右傳，保留最大 n =====
        int start = 0;
        for (int b = 0; b < num_blocks; b++) {
            int sz_my = std::min(block_sz, n - start);
            int end = start + sz_my;

            my_blocks[b] = std::vector<float>(d.begin() + start, d.begin() + end);
            start = end;

            int sz_partner = std::min(block_sz, partner_n - b * block_sz);
            partner_blocks[b].resize(sz_partner);

            if(rank == 1 && partner == 0) {
                printf("rank %d sending block %d of size %d to rank %d in phase %d in tag %d\n", rank, b, sz_my, partner, phase, tag);
                //printf("rank %d receiving block %d of size %d from rank %d in phase %d in tag %d\n", rank, b, sz_partner, partner, phase, tag);
                fflush(stdout);
            }

            MPI_Isend(my_blocks[b].data(), sz_my, MPI_FLOAT, partner, b, MPI_COMM_WORLD, &send_reqs[b]);
            MPI_Irecv(partner_blocks[b].data(), sz_partner, MPI_FLOAT, partner, b, MPI_COMM_WORLD, &recv_reqs[b]);
        }

        std::vector<float> keep;
        keep.reserve(n);

        for (int b = 0; b < num_blocks && (int)keep.size() < n; b++) {
            MPI_Wait(&recv_reqs[b], MPI_STATUS_IGNORE);
            int recv_sz = partner_blocks[b].size();
            if (recv_sz > 0 && rank == 1 && partner == 0) {
                printf("[rank %d] phase=%d tag=%d received block %d size=%d [%f ... %f] from rank %d\n",
                    rank, phase, tag, b, recv_sz,
                    partner_blocks[b].front(), partner_blocks[b].back(),
                    partner);
                fflush(stdout);
}

            // merge my_blocks[b] + partner_blocks[b] → cand (降序)
            std::vector<float> cand;
            int i = (int)my_blocks[b].size() - 1, j = (int)partner_blocks[b].size() - 1;
            while (i >= 0 || j >= 0) {
                if (i >= 0 && (j < 0 || my_blocks[b][i] >= partner_blocks[b][j])) {
                    cand.push_back(my_blocks[b][i--]);
                } else {
                    cand.push_back(partner_blocks[b][j--]);
                }
            }

            // keep = merge_truncate(keep, cand, n)（兩個都是大→小）
            std::vector<float> new_keep;
            int a = 0, c = 0;
            while ((int)new_keep.size() < n &&
                   (a < (int)keep.size() || c < (int)cand.size())) {
                if (a < (int)keep.size() &&
                    (c >= (int)cand.size() || keep[a] >= cand[c]))
                    new_keep.push_back(keep[a++]);
                else
                    new_keep.push_back(cand[c++]);
            }
            keep.swap(new_keep);
        }

        MPI_Waitall(num_blocks, send_reqs.data(), MPI_STATUSES_IGNORE);
        std::reverse(keep.begin(), keep.end());  // 回到升序
        d.swap(keep);
    }
}

// === even phase ===
void even_odd(std::vector<float>& d, int rank, int size, int N, int phase) {
    int tag = 1;
    if (rank % 2 == 0 && rank + 1 < size) {
        merge_all_pair_partial(d, rank + 1, rank, size, N, tag, phase);
    } else if (rank % 2 == 1) {
        merge_all_pair_partial(d, rank - 1, rank, size, N, tag, phase);
    }
}

// === odd phase ===
void odd_even(std::vector<float>& d, int rank, int size, int N, int phase) {
    int tag = 2;
    if (rank % 2 == 1 && rank + 1 < size) {
        merge_all_pair_partial(d, rank + 1, rank, size, N, tag, phase);
    } else if (rank % 2 == 0 && rank > 0) {
        merge_all_pair_partial(d, rank - 1, rank, size, N, tag, phase);
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
    
    for (int phase = 0; phase < size/2 + 1; ++phase) {
        even_odd(d, rank, size, N, phase);
        odd_even(d, rank, size, N, phase);
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
    //    for (int i = 0; i < 15; i++) {
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
