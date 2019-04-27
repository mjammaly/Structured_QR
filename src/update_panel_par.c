#include <stdlib.h>
#include <cblas.h>

#include "print_matrix.h"
#include "tasks.h"
#include "utils.h"
#include "spin-barrier.h"

// compute : A <- (I - B * C') * A, where B and C are trapezoidal matrices.
// Z is temporary buffer
void update_panel_par(int m, int n, int k, double *A, int ldA, double *B, int ldB, double *C, int ldC, double *Z, int ldZ, int nth, int rank, pcp_barrier_t *barrier)
{
#define A(i,j) A[(i) + (j) * ldA]
#define B(i,j) B[(i) + (j) * ldB]
#define C(i,j) C[(i) + (j) * ldC]
#define Z(i,j) Z[(i) + (j) * ldZ]
        
    if (rank == 0)
    {
	// Compute block size, other than the first block
	int blksz = iceil(m, nth);
	
	// Determine my share of A.
	int my_first_row = 0;
	int my_blksz = blksz;
	int my_first_row_A2 = 0;
	int my_blksz_A2 = 0;
	
	if(blksz < k)
	{
	    blksz = iceil(m-k, nth-1);
	    my_blksz = k;
	}
       	
	// Determine my share of Z.
	int my_first_col = 0;	

	// Copy : Z <- A1
	for(int j = 0 ; j < n; j++)
	    for(int i = 0; i < k; i++)
		Z(i,my_first_col + j) = A(my_first_row + i,j);
		
	// TRMM : Z <- C1' * Z
	cblas_dtrmm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, CblasUnit, k, n, 1.0, C, ldC, &Z(0,my_first_col), ldZ);    

	if(my_blksz > k)
	{
	    // Determine my share of A2.
	    my_first_row_A2 = k;
	    my_blksz_A2 = blksz - k;
		    
	    if(my_blksz_A2 > 0)
	    {
		// GEMM : Z <- Z + C2' A2
		cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, k, n, my_blksz_A2,  1.0, &C(my_first_row_A2,0), ldC, &A(my_first_row_A2,0), ldA, 1.0, &Z(0,my_first_col), ldZ);    	
	    }	    
	}
	
	// Sync	
	pcp_barrier_wait(barrier);
	
	// Sum Z from all threads
	// Z_0 += Z_i
	int thrd_first_col = 0;
	int thrd_first_row = 0;
	int thrd_blksz = 0;
	for(int thrd = 1; thrd < nth; thrd++)
	{
	    // Thread share of Z
	    thrd_first_col = thrd * n;

	    // Thread Share of A
	    if(my_blksz > k)
	    {
		thrd_first_row = blksz * thrd;
	    }
	    else
	    {
		thrd_first_row = k + (blksz * (thrd-1));
	    }		    
	    thrd_blksz = min(blksz, m - thrd_first_row);
	    
	    if(thrd_blksz > 0)
	    {
		for(int j = 0 ; j < n; j++)
		    for(int i = 0; i < k; i++)
		    {
			Z(i,my_first_col + j) += Z(i,thrd_first_col + j);
		    }
	    }
	}	
	pcp_barrier_wait(barrier);
	
	// Wait other threads to copy local bffer
	pcp_barrier_wait(barrier);

	if(my_blksz > k && my_blksz_A2 > 0)
	{
	    // GEMM : A2 <- A2 - B2 * Z
	    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, my_blksz_A2, n, k, -1.0, &B(my_first_row_A2,0), ldB, &Z(0,my_first_col), ldZ, 1.0, &A(my_first_row_A2,0), ldA);
	}
	
	// TRMM : Z <- B1 * Z
	cblas_dtrmm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, k, n, 1.0, B, ldB, &Z(0,my_first_col), ldZ);
		
	// Subtract : A1 <- A1 - Z
	for( int j = 0; j < n; j++ )
	    for( int i = 0; i < k; i++ )
		A(my_first_row + i,j) = A(my_first_row + i,j) - Z(i,my_first_col + j);
    }
    else
    {
	// Compute block size.
	int blksz = iceil(m, nth);

	// Determine my share of A.
	int my_first_row = blksz * rank;
	
	if(blksz < k) // first block must be >= k 
	{
	    // Update block size.
	    blksz = iceil(m-k, nth-1);

	    // Update my share of A.	    
	    my_first_row = k + (blksz * (rank-1));
	}

	int my_blksz = min(blksz, m - my_first_row);

	// Determine my share of Z.
	int my_first_col = rank*n;
	
	if(my_blksz > 0)
	{
	    // GEMM : Z <- Z + C2' A2
	    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, k, n, my_blksz,  1.0, &C(my_first_row,0), ldC, &A(my_first_row,0), ldA, 0.0, &Z(0,my_first_col), ldZ);    	
	}

	// Sync
	pcp_barrier_wait(barrier);
	
	// Wait master thread to sum Z
	pcp_barrier_wait(barrier);
	
	// Copy Z from master thread
	if(my_blksz > 0)
	{	
	    // Z_i = Z_0
	    for(int j = 0 ; j < n; j++)
		for(int i = 0; i < k; i++)
		{
		    Z(i,my_first_col + j) = Z(i,j);
		}
	}	
	pcp_barrier_wait(barrier);

	if(my_blksz > 0)
	{	
	    // GEMM : A2 <- A2 - B2 * Z
	    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, my_blksz, n, k, -1.0, &B(my_first_row,0), ldB, &Z(0,my_first_col), ldZ, 1.0, &A(my_first_row,0), ldA);
	}	
    }

#undef A
#undef B
#undef C
#undef Z
}

