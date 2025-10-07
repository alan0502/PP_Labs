#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

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

    #pragma omp parallel for reduction(+:pixels)
    for (unsigned long long x = 0; x < r; x++) {
        long double diff = rr - (long double)x * x;
        if (diff < 0) diff = 0;  // clamp to avoid sqrt domain error
        unsigned long long y = ceil(sqrtl(diff));
        pixels += y;
        // NOTE: don't use %k here — do final mod later for correctness
    }

    unsigned long long res = (4 * (pixels % k)) % k;
    printf("%llu\n", res);
    return 0;
}
