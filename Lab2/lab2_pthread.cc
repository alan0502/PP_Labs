#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <immintrin.h>

typedef unsigned long long ull;

typedef struct {
    ull r;
    ull k;
    int tid;
    int num_threads;
    ull partial;
} ThreadData;

void* worker(void* arg) {
    ThreadData* t = (ThreadData*)arg;
    ull r = t->r;
    ull k = t->k;
    int tid = t->tid;
    int num_threads = t->num_threads;

    ull local = 0;
    for (ull x = tid; x < r; x += num_threads) {
        ull y = ceil(sqrtl((long double)r * r - (long double)x * x));
        local += y;
        local %= k;  // mod every loop like MPI version
    }
    t->partial = local;
    return NULL;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "must provide exactly 2 arguments!\n");
        return 1;
    }

    ull r = atoll(argv[1]);
    ull k = atoll(argv[2]);

    int num_threads = 8;  // you can tune this
    pthread_t threads[num_threads];
    ThreadData td[num_threads];

    for (int i = 0; i < num_threads; i++) {
        td[i].r = r;
        td[i].k = k;
        td[i].tid = i;
        td[i].num_threads = num_threads;
        td[i].partial = 0;
        pthread_create(&threads[i], NULL, worker, &td[i]);
    }

    ull total_pixels = 0;
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
        total_pixels += td[i].partial;
        total_pixels %= k;  // keep consistent mod behavior
    }

    ull res = (4 * total_pixels) % k;
    printf("%llu\n", res);
    return 0;
}
