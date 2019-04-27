#include <stdio.h>
#include <stdlib.h>

// Print a matrix in a readable format on stdout
void print_matrix(int m, int n, double *A, int ldA) // m = rows , n = cols
{
#define A(i,j) A[(i) + (j) * ldA]        
    for(int i = 0; i < m; i++)	    
    {
	for(int j = 0; j < n; j++)
	    printf("%f ", A(i,j));	
	printf("\n");
    }
#undef A    
}
