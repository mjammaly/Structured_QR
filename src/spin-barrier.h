#ifndef PCP_SPIN_BARRIER_H
#define PCP_SPIN_BARRIER_H

//
// PURPOSE
//
// Low-latency barrier based on read/write of shared variables and busy waiting.
//
//
// USAGE
//
// Create a barrier (allocated dynamically) for nth threads:
//
//     pcp_barrier_t *bar = pcp_barrier_create(nth);
//
// Use the barrier by letting all nth threads call:
//
//     pcp_barrier_wait(bar);
//
// To destroy (deallocate) the barrier, call on one thread:
//
//     pcp_barrier_destroy(bar);
//
// Note that you must ensure yourself (by other means) that no thread is using
// the barrier (e.g., is still inside the pcp_barrier_wait function) before
// destroying the barrier.
//
// An alternative means of destroying the barrier is by letting ALL threads call
//
//     pcp_barrier_safe_destroy(bar);
//
// The last thread will call pcp_barrier_destroy.
//

typedef struct pcp_barrier pcp_barrier_t;

pcp_barrier_t *pcp_barrier_create      (int nth);
void           pcp_barrier_destroy     (pcp_barrier_t *bar);
void           pcp_barrier_wait        (pcp_barrier_t *bar);
void           pcp_barrier_safe_destroy(pcp_barrier_t *bar);

#endif
