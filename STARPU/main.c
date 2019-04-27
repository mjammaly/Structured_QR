#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cblas.h>
#include <lapacke.h>
#include <starpu.h>

#include "../src/str_qr.h"
#include "../src/random_matrix.h"
#include "../src/print_matrix.h"
#include "../src/test_reduction.h"
#include "../src/tasks.h"
#include "driver.h"

// To validate the results un-comment the following line
#define ENABLE_VALIDATION

static void usage(char *prog)
{
    printf("\n");
    printf("Usage: %s n b p itr\n", prog);    
    printf("n: The size of the matrix, n > 0 \n");
    printf("b: The tile size, n > b > 0 \n");
    printf("p: The number of threads, p >= 1\n");
    printf("q: The size of the reserved set (or 0 to use the regular mode)\n");
    printf("itr: The number of repetitions itr >= 1\n");    
    exit(EXIT_FAILURE);
}
    

int main(int argc, char** argv)
{
    // Verify the number of command line options.
    if (argc != 6)
    {
	usage(argv[0]);
    }

    // Parse command line.
    int n = atoi(argv[1]);
    int b = atoi(argv[2]);
    int nth = atoi(argv[3]);
    int reserved_set_size = atoi(argv[4]);
    int itr = atoi(argv[5]);   

    // Verify options.
    if ( n < 1 || b < 1 || nth < 1 || b > n || reserved_set_size >= nth || itr < 1)
    {
	usage(argv[0]);
    }

    // Allocate input matrix
    int ldA = n;
    double *A = (double *) malloc(n * ldA * sizeof(double));     

    
    // Initialise random number generator.
//    srand(time(NULL));
    
    // Generate block_interleaved_upper_triangler_matrix
    generate_dense_block_interleaved_upper_triangler_matrix(n, b, A, ldA);    
    
    // Allocate working buffers
    double *Ain = (double *) malloc(n * ldA * sizeof(double));
    int ldY = 2*b;
    double *Y = (double *) malloc(n * ldY * sizeof(double));     
    int ldZ = n;
    double *Z = (double *) malloc(n * ldZ * sizeof(double));
    int ldB = n;
    double *B = (double *) malloc(n * ldB * sizeof(double));
    
    // Copy A to Ain. 
    memcpy(Ain, A, n * ldA * sizeof(double));
    // Copy A to B. B used to verify the results
    memcpy(B, A, n * ldA * sizeof(double));	
    
    // Open file to collect measurments 
    FILE *f = fopen("time.txt","a");

    // Repeat the test "itr" times.
    for(int i = 0; i < itr; i++)
    {
	// Record the test inputs 
	fprintf(f,"%d %d %d %d\n",n,b,nth,reserved_set_size);	
		
	// Call the driver.
	if(reserved_set_size == 0) // Regular mode: sequential execution without PCP.
	{
	    printf("Calling the driver routine (sequential tasks)...\n");
	    sequential_str_qr(n, b, Ain, ldA, Y, ldY, nth, f);	    
	}
	
	else // Fixed mode: there is at least one thread reserved, apply PCP.
	{
	    printf("Calling the driver routine for (parallel critical tasks)...\n");
	    parallel_str_qr(n, b, Ain, ldA, Y, ldY, nth, reserved_set_size, f);	    
	}	


#ifdef ENABLE_VALIDATION
	// Validate results	
	if(i == 0) // only for the first test
	{
	    printf("Valedate the results ...\n");
	    test_matrix_reduction(n, b, B, ldB, Ain, ldA, Y, ldY, Z, ldZ);	
	}
#endif
	// Reset input and validation matrices.
	memcpy(Ain, A, n * ldA * sizeof(double));
	memcpy(B, A, n * ldA * sizeof(double));	

	// Print out some information about the completed test.
	printf("%d %d %d %d\n",n,b,nth,reserved_set_size);	
    }
    
    // Close the measurements file
    fclose(f);

    // Clean up
    free(A);
    free(Ain);    
    free(Y);
    free(Z);   
    
    return EXIT_SUCCESS;
}
