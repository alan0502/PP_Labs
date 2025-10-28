#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <immintrin.h>

typedef unsigned long long ull;

typedef struct {
    ull r; // radius
    ull k; // modulus
    int tid; // thread id
    int num_threads; // total number of threads
    ull partial; // partial sum
} ThreadData;

// Thread function
void* worker(void* arg) {
    ThreadData* t = (ThreadData*)arg;
    ull r = t->r;
    ull k = t->k;
    int tid = t->tid;
    int num_threads = t->num_threads;

    unsigned __int128 local = 0;
    for (ull x = tid; x < r; x += num_threads) {
		ull diff = r * r - x * x;
        ull y = sqrt(diff);
		if (diff > y * y) y++;
        local += y;
    }
	local %= k;
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

	// Get number of available CPUs, use that as number of threads
    cpu_set_t cpu_set;
	sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
	int num_threads = CPU_COUNT(&cpu_set);
	//printf("%d CPUs available, using %d threads\n", num_threads, num_threads);

	// Create threads
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

	// Join threads and accumulate results in total_pixels
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
