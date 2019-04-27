#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>
#include <lapacke.h>

#include "print_matrix.h"
#include "utils.h"    

// Tests the result of factoring a given block triangular matrix with overlapped diagonal blocks to Q and R.
// computes:
// ||I-Q*Q'||_f
// ||A-Q*R|| / ||A||
void test_matrix_reduction(int n, int b, double *A, int ldA, double *R, int ldR, double *Y, int ldY, double *Z, int ldZ)
{
#define A(i,j) A[(i) + (j) * ldA]
#define R(i,j) R[(i) + (j) * ldR]        
#define Y(i,j) Y[(i) + (j) * ldY]
#define Z(i,j) Z[(i) + (j) * ldZ]
#define Q(i,j) Q[(i) + (j) * ldQ]
#define W(i,j) W[(i) + (j) * ldW]    
    
    // Allocate memory for Q and W
    int ldQ = n;
    double *Q = (double *) malloc(n * ldQ * sizeof(double));
    int ldW = 2*b;
    double * W= (double *) malloc(n * ldW * sizeof(double));


    // Set Q = I 
    for(int j = 0; j < n; j++)
    {
	for(int i = 0; i < n; i++)
	{
	    Q(i,j)=0.0;
	}
	    Q(j,j)=1.0;	
    }

    // Set W to zeros
    for(int j = 0; j < n; j++)
	for(int i = 0; i < ldW; i++)
	    W(i,j) = 0;	
    

    // Extract W from R and set R to upper triangular    
    int N = iceil(n,b); // number of blocks.
    int blksiz = n - (N-2) * b; // size of the last block.
    
    for(int k = 0; k < n - blksiz; k+=b)	
    {
	for(int j = 0; j < b; j++)
	{
	    W(j,k+j) = 1;	    
	    for(int i = j+1; i < 2*b; i++)
	    {
		W(i,k+j) = R(k+i,k+j);
		R(k+i,k+j) = 0;
	    }
	}
    }

    // last panel in R
    int k = n - blksiz; // start of the last block.

    for(int j = 0; j < blksiz; j++)
    {
	W(j,k+j) = 1;
	for(int i = j+1; i < blksiz; i++)
	{
	    W(i,k+j) = R(k+i,k+j);
	    R(k+i,k+j) = 0;
	}
    }

    
    // Construct Q = Q1*Q2*Q3 ... , where Qi = (I-Wi*Y'i).
    for(int i = 0; i < n - blksiz; i+=b)
    {
	{
	    // Z = Qi*Wi ,
	    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, i+(2*b), b, 2*b, 1.0, &Q(0,i), ldQ, &W(0,i), ldW, 0.0, Z, ldZ);	    
	    // Qi = Qi - Z*Y'i
	    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, i+(2*b), 2*b, b, -1.0, Z, ldZ, &Y(0,i), ldY, 1.0, &Q(0,i), ldQ);
	}
    }

    // last panel , W and Y are triangular
    // Z = Q
    for(int j = 0; j < blksiz; j++)
    {
	for (int i = 0; i < n; i++)
	    Z(i,j) = Q(i, j + (n - blksiz));
    }
    // Z = Z * W
    cblas_dtrmm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans, CblasNonUnit, n, blksiz, 1.0, &W(0,n-blksiz), ldW, Z, ldZ);    
    // Z = Z * Y'
    cblas_dtrmm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit, n, blksiz, 1.0, &Y(0,n-blksiz), ldY, Z, ldZ);        
    // Q = Q - Z
    for(int j = 0; j < blksiz; j++)
    {
	for (int i = 0; i < n; i++)
	    Q(i, j + (n - blksiz)) -=  Z(i,j) ;
    }        

    // verify that Q is orthogonal
    // Set Z = I
    for(int j = 0; j < n; j++)
    {
	for (int i = 0; i < n; i++)
	{
	    Z(i,j) = 0;
	}
	Z(j,j) = 1;
    }
    

    // ||I-Q*Q'||_f
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, n, n, n, -1.0, Q, ldQ, Q, ldQ, 1.0, Z, ldZ);
    double norm_Q = LAPACKE_dlange(LAPACK_COL_MAJOR, 'F', n, n, Z, ldZ);    
    printf("norm(I-Q'Q)=%e\n",norm_Q);
    
    // ||A||_f
    double norm_A = LAPACKE_dlange(LAPACK_COL_MAJOR, 'F', n, n, A, ldA);
    
    // A <- A - Q*R
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, -1.0, Q, ldQ, R, ldR, 1.0, A, ldA);    

    // ||A-Q*R|| / ||A||
    double norm_R = LAPACKE_dlange(LAPACK_COL_MAJOR, 'F', n, n, A, ldA);
    printf("norm(A-QR)=%e\n",norm_R);
    printf("norm(A-QR)/norm(A)=%e\n",norm_R/norm_A);

    if ( norm_Q < 1e-12 && (norm_R/norm_A) < 1e-14)
    {	
	printf("PASSED\n");
    }
    else
    {
	printf("FAILED\n");
    }
    
#undef A
#undef R
#undef Y
#undef Z
#undef Q
#undef W

    free(Q);
    free(W);    
}


// Tests the result of factoring a given rectangular dense matrix to Q and R.
// computes:
// ||I-Q*Q'||_f
// ||A-Q*R|| / ||A||
void test_panel_reduction(int m, int n, double *A, int ldA, double *R, int ldR, double *Y, int ldY, double *Z, int ldZ)
{
#define A(i,j) A[(i) + (j) * ldA]
#define R(i,j) R[(i) + (j) * ldR]        
#define Q(i,j) Q[(i) + (j) * ldQ]
#define Y(i,j) Y[(i) + (j) * ldY]
#define Z(i,j) Z[(i) + (j) * ldZ]

    // Verify the solution.
    printf("Verifying the solution... \n");   

    // Construct Q
    printf("Constructing Q ... \n");
    int ldQ = m;
    double *Q = (double *) malloc(m * ldQ * sizeof(double));


    // Q = [Q1 Q2]     
    // Copy : Q1 <- W
    // Set : W = 0 in  R
    for (int j = 0; j < n; j++)
    {
    	for (int i = 0; i < j; i++)	
    	{
    	    Q(i,j) = 0.0;
    	}

    	Q(j,j) = 1;

    	for (int i = j+1; i < m; i++)	
    	{
    	    Q(i,j) = R(i,j);
    	    R(i,j) = 0;
    	}	
    }
    
    // Create I , Z <- I
    for(int j = 0; j < m; j++)
    {
    	for(int i = 0; i < m; i++)
    	    Z(i,j) = 0;
    	Z(j,j) = 1;
    }

    // I = [I1 I2] 
    // Q2 = I2
    for(int j = n; j < m; j++)
    {
    	for(int i = 0; i < m; i++)
    	    Q(i,j) = Z(i,j);
    }    

    // Y' = [Y1 Y2]' where Y1 is triangular.
    // Q2 <-Q2 - (W * Y2') 
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, m, m-n, n, -1.0, Q, ldQ, &Y(n,0), ldY, 1.0, &Q(0,n), ldQ);        
    
    // Q1 <- W * Y1'
    cblas_dtrmm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit, m, n, 1.0, Y, ldY, Q, ldQ);

    // Q1 = I1 - Q1
    for(int j = 0; j < n; j++)
    {
	for(int i = 0; i < m; i++)	
	    Q(i,j) = Z(i,j) - Q(i,j);
    }

    // Verify that A = Q * R

    // ||A||_F
    double norm_A = LAPACKE_dlange(LAPACK_COL_MAJOR, 'F', m, n, A, ldA);

    // ||A - Q * R||_F
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, m, -1.0, Q, ldQ, R, ldR, 1.0, A, ldA);    
    double norm_R = LAPACKE_dlange(LAPACK_COL_MAJOR, 'F', m, n, A, ldA);
    printf("norm(A-QR)/norm(A)=%e\n",norm_R/norm_A);
    
    // Verify that Q is orthogonal

    // ||I - Q' * Q||_F    
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, m, m, -1.0, Q, ldQ, Q, ldQ, 1.0, Z, ldZ);
    double norm_Q = LAPACKE_dlange(LAPACK_COL_MAJOR, 'F', m, m, Z, ldZ);
    printf("norm(I-Q'Q)=%e\n",norm_Q);    

    // Clean up.
#undef A
#undef R
#undef Q    
#undef Y
#undef Z

    free(Q);    
}







