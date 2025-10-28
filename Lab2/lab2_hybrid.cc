#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s r k\n", argv[0]);
        return 1;
    }

    unsigned long long r = atoll(argv[1]);
    unsigned long long k = atoll(argv[2]);
    unsigned long long local_pixels = 0, total_pixels = 0;

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long double rr = (long double)r * r;

	// 把這個 for loop 的「每一次迭代」自動分給不同的 thread 執行。
	// reduction(+:local_pixels) to sum up contributions from threads，自動把每個 thread 的 local_pixels 加起來
	// reduction 是為了處理 不同 thread 都在做 local_pixels += y; 這件事
    #pragma omp parallel for reduction(+:local_pixels)
    for (unsigned long long x = rank; x < r; x += size) {
        unsigned long long diff = rr - x * x;
        if (diff < 0) diff = 0;
        unsigned long long y = sqrt(diff);
		if (diff > y * y) y++; // ceil
        local_pixels += y;
    }

	// 把所有 rank 的 local_pixels 加起來，存在 rank 0 的 total_pixels 裡
    MPI_Reduce(&local_pixels, &total_pixels, 1, MPI_UNSIGNED_LONG_LONG,
               MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        unsigned long long res = (4 * (total_pixels % k)) % k;
        printf("%llu\n", res);
    }

    MPI_Finalize();
    return 0;
}
