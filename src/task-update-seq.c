#include <stdlib.h>
#include <cblas.h>

#include "print_matrix.h"
#include "tasks.h"
#include "utils.h"


// compute : A <- (I - B * C') * A, where B and C are trapezoidal matrices.
// Z is an intermediate matrix
void update_task_seq(void *ptr)    
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

    // Allocate temp buffer
    int ldZ = k;
    double *Z = (double*) malloc(n * ldZ * sizeof(double));    
    
    
    // Copy : Z <- A1
    for(int j = 0; j < n; j++)
	for(int i = 0; i < k; i++)
	    Z(i,j) = A(i,j);
	
    // TRMM : Z <- C1' * Z
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, CblasUnit, k, n, 1.0, C, ldC, Z, ldZ);    

    
    // GEMM : Z <- Z + C2' A2
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, k, n, m-k, 1.0, &C(k,0), ldC, &A(k,0), ldA, 1.0, Z, ldZ);    

    
    // GEMM : A2 <- A2 - B2 * Z
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m-k, n, k, -1.0, &B(k,0), ldB, Z, ldZ, 1.0, &A(k,0), ldA);
    

    // TRMM : Z <- B1 * Z
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, k, n, 1.0, B, ldB, Z, ldZ);


    // Subtract : A1 <- A1 - Z
    for( int j = 0; j < n; j++ )
	for( int i = 0; i < k; i++ )
	    A(i,j) = A(i,j) - Z(i,j);


    
#undef A
#undef B
#undef C
#undef Z

    free(Z);
}

