#include <stdio.h>
#include <stdlib.h>
#include <lapacke.h>
#include <cblas.h>
//#include <pthread.h>

#include "tasks.h"
#include "print_matrix.h"
#include "update_panel.h"
#include "utils.h"
#include "spin-barrier.h"

static int c_counter = 0;
static int p_counter = 0;

// Barrier used to synchronize the workers between iterations.
static pcp_barrier_t *barrier = NULL;


void factor_task_par_reconfigure(int nth)
{
    // Choose current size of the worker pool for barrier.
    if (barrier != NULL)
    {
	pcp_barrier_safe_destroy(barrier);	
    }
    barrier = pcp_barrier_create(nth);
}


void factor_task_par_finalize(void)
{
    if (barrier != NULL)
    {
	pcp_barrier_safe_destroy(barrier);
	barrier = NULL;
    }
}


// Factor a given matrix/panel to Q and R recursively and in parallel
void factor_panel_par(int m, int n, double *A, int ldA, double *Y, int ldY, double *Z, int ldZ, int nth, int rank)    
{
	
#define A(i,j) A[(i) + (j) * ldA]
#define Y(i,j) Y[(i) + (j) * ldY]
    
    int k = min (m-1,n);
    double tau = 0;
    
    if (n == 1)
    {
	// A is 1x1
	if(k == 0)
	    return;
	else
	{
	    if(rank == 0)		
	    {
		// construct reflector and apply it to column of A
		LAPACKE_dlarfg(m, A, &A(1,0),1, &tau);

		// construct column of Y = W * tau
		Y(0,0) = tau; // first element in W is 1 and it is not stored 
		for (int i = 1; i < m; i++)
		    Y(i,0) = A(i,0)*tau;
	    }
	    pcp_barrier_wait(barrier);
	}
    }
    else
    {	
	// slpit the matrix into two halves
	int n1 = ifloor(n, 2);
	int n2 = n-n1;
	
	// factor the left half
	factor_panel_par(m, n1, A, ldA, Y, ldY, Z, ldZ, nth, rank);
    
	pcp_barrier_wait(barrier);

	// update the right half
	// A <- (I - Y * W') * A
	update_panel_par(m, n2, n1, &A(0,n1), ldA, Y, ldY, A, ldA, Z, ldZ, nth, rank, barrier);
	
	pcp_barrier_wait(barrier);
	
	// factor the right part	    		
	// Locate block Y22
	double *Y22 = Y + n1 + n1*ldY;		
	// Locate block A22
	double *A22 = A + n1 + n1*ldA;	
		
	//factor_task_par(factor_arg, nth, rank);
	factor_panel_par(m-n1, n2, A22, ldA, Y22, ldY, Z, ldZ, nth, rank);
	
	pcp_barrier_wait(barrier);
	
	// augment the two WY parts
	// Y1 <- (I - Y2 * W2') * Y1
	update_panel_par(m-n1, n1, n2, &Y(n1,0), ldY, &Y(n1,n1), ldY, &A(n1,n1), ldA, Z, ldZ, nth, rank, barrier);
		
	pcp_barrier_wait(barrier);	
    }	
    
#undef A
#undef Y
}


// Wrapper to "factor_panel_par"
void factor_task_par(void *ptr, int nth, int rank)    
{
    struct factor_task_arg *factor_arg = (struct factor_task_arg*) ptr;

    int m = factor_arg->m;
    int n = factor_arg->n;
    double *A = factor_arg->A;
    int ldA = factor_arg->ldA;
    double *Y = factor_arg->Y;
    int ldY = factor_arg->ldY;
    double *Z = factor_arg->Z;
    int ldZ = factor_arg->ldZ;
        
    factor_panel_par(m, n, A, ldA, Y, ldY, Z, ldZ, nth, rank);
    
    pcp_barrier_wait(barrier);
}
