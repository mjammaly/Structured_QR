#include <stdio.h>
#include <stdlib.h>
#include <lapacke.h>
#include <cblas.h>

#include "tasks.h"
#include "print_matrix.h"
#include "update_panel.h"
#include "utils.h"

// Factor a given matrix to Q nad R recursively and sequentially.
void factor_task_seq(void *ptr)        
{
#define A(i,j) A[(i) + (j) * ldA]
#define Y(i,j) Y[(i) + (j) * ldY]

    struct factor_task_arg *arg = (struct factor_task_arg*) ptr;

    int m = arg->m;
    int n = arg->n;
    double *A = arg->A;
    int ldA = arg->ldA;
    double *Y = arg->Y;
    int ldY = arg->ldY;
    double *Z = arg->Z;
    int ldZ = arg->ldZ;    
    
    int k = min (m-1,n);
    double tau = 0;
    if (n == 1)
    {
	// A is 1x1
	if(k == 0)
	    return;
	else
	{
	    // construct reflector and apply it to column of A
	    LAPACKE_dlarfg(m, A, &A(1,0),1, &tau);

	    // construct column of Y = V * tau
	    Y(0,0) = tau; // first element in V is 1 and it is not stored 
	    for (int i = 1; i < m; i++)
		Y(i,0) = A(i,0)*tau;
	}
    }
    else
    {
	// slpit the matrix into two halves [A1 A2] , [Y1 Y2] , slpit A2 [A21 A22]' ,  Y2 [Y21 Y22]'
	int n1 = ifloor(n, 2);
	int n2 = n-n1;
	
	// factor the left half
	arg->n = n1;	
	factor_task_seq(arg);	
	
	// update the right half	
	// A <- (I - Y * W') * A
	update_panel_seq(m, n2, n1, &A(0,n1), ldA, Y, ldY, A, ldA, Z, ldZ);
	
	// factor the right part	
	// Locate block Y22
	double *Y22 = Y + n1 + n1*ldY;		
	// Locate block A22
	double *A22 = A + n1 + n1*ldA;	
	
	arg->m = m-n1;
	arg->n = n2;
	arg->A = A22;
	arg->Y = Y22;
	
	factor_task_seq(arg);		
	
	// augment the two WY parts
	update_panel_seq(m-n1, n1, n2, &Y(n1,0), ldY, &Y(n1,n1), ldY, &A(n1,n1), ldA, Z, ldZ);  			

	// Return changed arguments to its original values
	arg->m = m;
	arg->n = n;
	arg->A = A;
	arg->Y = Y;			
    }

#undef A
#undef Y    
}
