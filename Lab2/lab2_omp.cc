#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// 所有 x 都在單一機器 (rank) 上，用 OpenMP 分配迴圈給多 thread。
int main(int argc, char** argv)
{
    if (argc != 3) {
        fprintf(stderr, "must provide exactly 2 arguments!\n");
        return 1;
    }

    unsigned long long r = atoll(argv[1]);
    unsigned long long k = atoll(argv[2]);
    unsigned long long pixels = 0;  // will be used in reduction

    unsigned long long rr = (long double)r * r;

	// 把這個 for loop 的「每一次迭代」自動分給不同的 thread 執行。
	// reduction(+:pixels) to sum up contributions from threads，自動把每個 thread 的 pixels 加起來
	// reduction 是為了處理 不同 thread 都在做 pixels += y; 這件事
    #pragma omp parallel for reduction(+:pixels)
    for (unsigned long long x = 0; x < r; x++) {
        unsigned long long diff = rr - x * x;
        if (diff < 0) diff = 0;
        unsigned long long y = sqrt(diff);
		if (diff > y * y) y++; // ceil
        pixels += y;
    }

    unsigned long long res = (4 * (pixels % k)) % k;
    printf("%llu\n", res);
    return 0;
}
