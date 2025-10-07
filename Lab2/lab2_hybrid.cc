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

    #pragma omp parallel for reduction(+:local_pixels)
    for (unsigned long long x = rank; x < r; x += size) {
        long double diff = rr - (long double)x * x;
        if (diff < 0) diff = 0;
        unsigned long long y = ceil(sqrtl(diff));
        local_pixels += y;
    }

    MPI_Reduce(&local_pixels, &total_pixels, 1, MPI_UNSIGNED_LONG_LONG,
               MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        unsigned long long res = (4 * (total_pixels % k)) % k;
        printf("%llu\n", res);
    }

    MPI_Finalize();
    return 0;
}
