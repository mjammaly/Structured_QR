#include <stdlib.h>
#include <cblas.h>

#include "print_matrix.h"
#include "tasks.h"
#include "utils.h"

// This function will be called to (re)configure update_task_par kernel.
// Required by PCP runtime system.
void update_task_par_reconfigure(int nth)
{
    // empty
}

// This function will be called to finilize (destroy) configuration(s) of update_task_par kernel.
// Required by PCP runtime system.
void update_task_par_finalize(void)
{
    // empty
}

// compute : A <- (I - B * C') * A, where B and C are trapezoidal matrices.
// Z is temporary buffer
void update_task_par(void *ptr, int nth, int rank)
{
#define A(i,j) A[(i) + (j) * ldA]
#define B(i,j) B[(i) + (j) * ldB]
#define C(i,j) C[(i) + (j) * ldC]
#define Z(i,j) Z[(i) + (j) * ldZ]
    
    struct update_task_arg *arg = (struct update_task_arg*) ptr;

    int m = arg->m;
    int n = arg->n;
    int k = arg->k;    
    double *A = arg->A;
    int ldA = arg->ldA;
    double *B = arg->B;
    int ldB = arg->ldB;
    double *C = arg->C;
    int ldC = arg->ldC;           
    
    // Compute block size.
    int blksz = iceil(n, nth);
    
    // Determine my share of A.
    int my_first_col = blksz * rank;
    int my_blksz = min(blksz, n - my_first_col);


    if(my_blksz > 0)
    {
    	// Allocate temp buffer
    	int ldZ = k;
    	double *Z = (double*) malloc(my_blksz * ldZ * sizeof(double));
	
    	// Copy : Z <- A1
    	for(int j = 0 ; j < my_blksz; j++)
    	    for(int i = 0; i < k; i++)
    		Z(i,j) = A(i,my_first_col + j);
	
	
    	// TRMM : Z <- C1' * Z
    	cblas_dtrmm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, CblasUnit, k, my_blksz, 1.0, C, ldC, Z, ldZ);		

    	// GEMM : Z <- Z + C2' A2
    	cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, k, my_blksz, m-k, 1.0, &C(k,0), ldC, &A(k,my_first_col), ldA, 1.0, Z, ldZ);
	
    	// GEMM : A2 <- A2 - B2 * Z
    	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m-k, my_blksz, k, -1.0, &B(k,0), ldB, Z, ldZ, 1.0, &A(k,my_first_col), ldA);
	
    	// TRMM : Z <- B1 * Z
    	cblas_dtrmm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, k, my_blksz, 1.0, B, ldB, Z, ldZ);
	
    	// Subtract : A1 <- A1 - Z
    	for( int j = 0; j < my_blksz; j++ )
    	    for( int i = 0; i < k; i++ )
    		A(i,my_first_col + j) = A(i,my_first_col + j) - Z(i,j);

    	// Clean up
    	free(Z);	    
    }
        
#undef A
#undef B
#undef C
#undef Z
}
