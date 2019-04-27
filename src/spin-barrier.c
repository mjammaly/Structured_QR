//
// Note that this implementation is not portable as it relies on non-standard
// GCC compiler intrinsics, specifically:
//
//     __sync_add_and_fetch
// 


#include <stdlib.h>
#include <stdbool.h>

#include "spin-barrier.h"


// Spin barrier object type.
struct pcp_barrier
{
    // The number of threads associated with this barrier.
    int count;

    // The number of threads that have arrived at the barrier.
    volatile int arrived;

    // The number of times this barrier has been used. 
    volatile int generation;
};


pcp_barrier_t *pcp_barrier_create(int nth)
{
    // Allocate memory and initialize the object.
    pcp_barrier_t *bar = (pcp_barrier_t*) malloc(sizeof(pcp_barrier_t));
    bar->count         = nth;
    bar->arrived       = 0;
    bar->generation    = 0;
    return bar;
}


void pcp_barrier_wait(pcp_barrier_t *bar)
{
    // Determine the active generation.
    const int generation = bar->generation;
    // Notify arrival. Am I the last one to arrive?
    if (__sync_add_and_fetch(&bar->arrived, 1) == bar->count) {
        // Yes, so reset and bump the generation counter to release all threads.
        bar->arrived = 0;
        __sync_add_and_fetch(&bar->generation, 1);
    } else {
        // No, so busy wait until the next generation. Note: the use of bar here
        // could lead to a data race with pcp_barrier_destroy.
        while (generation == bar->generation)
	    ;
    }
}


void pcp_barrier_destroy(pcp_barrier_t *bar)
{
    // Deallocate memory.
    free(bar);
}


void pcp_barrier_safe_destroy(pcp_barrier_t *bar)
{
    if (__sync_add_and_fetch(&bar->arrived, 1) == bar->count) {
        pcp_barrier_destroy(bar);
    }
}
