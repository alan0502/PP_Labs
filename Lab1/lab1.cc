#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <immintrin.h>

int main(int argc, char** argv)
{
	int num_procs, rank;
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	unsigned long long r = atoll(argv[1]), 
		           k = atoll(argv[2]),
			   pixels = 0, total_pixels = 0;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	for(unsigned long long x = rank; x < r; x += num_procs){
		unsigned long long y = ceil(sqrtl(r * r - x * x));
		pixels += y;
		pixels %= k;
	}
	
	MPI_Reduce(&pixels, &total_pixels, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	if (rank == 0) {
        unsigned long long res = (4 * total_pixels) % k;
        printf("%llu\n", res);
    }
	MPI_Finalize();
	return 0;
}

