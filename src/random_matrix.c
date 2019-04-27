#include <stdio.h>
#include <stdlib.h>

#include "random_matrix.h"
#include "utils.h"

// Return a random double between max and min.
static double random_double(double min, double max)
{
    double range = max - min;
    return min + rand() / (RAND_MAX / range);
}


// Generate diagonal dominant dense rectangler matrix 
void generate_dense_rectangle_matrix(int m, int n, double *A, int ldA)
{
#define A(i,j) A[(i) + (j) * m]
    // Generate m-by-n matrix with random values in (0.0, 1.0).
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            A(i,j) = random_double(0.0, 1.0);
        }
    }
    
    // Make A diagonally dominant with A = A + n * I.
    for (int i = 0; i < n; i++) {
        A(i,i) += n;
    }
#undef A    
}



// Generate block upper triangular matrix with overlapping blocks on the diagonal.
void generate_dense_block_interleaved_upper_triangler_matrix(int n, int b, double *A, int ldA)
{
#define A(i,j) A[(i) + (j) * n]    
    // Generate n-by-n upper triangluar matrix with random values in (0.0, 1.0).
    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < j; i++)
	{
            A(i,j) = random_double(0.0, 1.0);
        }
        for (int i = j; i < n; i++)
	{
            A(i,j) = 0.0;
        }	
    }
    
    // Generate interleaved diagonal blockes of size 2b-by-2b

    int N = iceil(n,b); // number of blocks.
    
    for (int k = 0; k < N-2; k++)
    {
	for (int j = 0; j < b; j++)
	{
	    for (int i = j+1; i < 2*b; i++)
	    {
		A(i+(k*b),j+(k*b)) = random_double(0.0, 1.0);
	    }
	}
    }

    // Last block.
    int blksiz = n - (N-2) * b;

    int k = (N-2) * b; // First column (and row) of the last block.
    
    for (int j = 0; j < blksiz; j++)
    {
	for (int i = j+1; i < blksiz; i++)
	{
	    A(i+k,j+k) = random_double(0.0, 1.0);
	}
    }

    // Make A diagonally dominant with A = A + n * I.
    for (int i = 0; i < n; i++)
    {
        A(i,i) += n;
    }
    
#undef A        
}
