#include <stdlib.h>
#include <stdio.h>
#include <cblas.h>

#include "str_qr.h"
#include "print_matrix.h"
#include "tasks.h"
#include "utils.h"

// Factor a given block triangular matrix with overlapping blocks on the diagonal to Q and R using sequential kernels.
void str_qr_seq(int n, int b, double *A, int ldA, double *Y, int ldY)
{
    
#define Y(i,j) Y[(i) + (j) * ldY]
#define Z(i,j) Z[(i) + (j) * ldZ]

    // Allocate temp buffer
    int ldZ = 2*b;
    double *Z = (double*) malloc( 2*b * ldZ *sizeof(double));

    // File to store measurements
    FILE *f = fopen("results.txt","a");
    
    // timers 
    double timer;
    double flops = 0;
    
    // reset Y to 0s
    for(int j = 0; j < n; j++)
	for (int i = 0; i < ldY; i++)
	{
	    Y(i,j) = 0;
	}
    
    // reset Z to 0s
    for(int j = 0; j < 2*b; j++)
	for (int i = 0; i < ldZ; i++)	    
	{
	    Z(i,j) = 0;		
	}

    
    int N = iceil(n,b); // number of panels to reduce.

    fprintf(f,"%d %d\n",n,b);
    printf("%d %d\n",n,b);    
    
    struct factor_task_arg *factor_arg = (struct factor_task_arg*) malloc(sizeof(*factor_arg));
    struct update_task_arg *update_arg = (struct update_task_arg*) malloc(sizeof(*update_arg));    
    
    // Loop over the panels.
    for(int i = 0; i < N-2; i++)
    {
	// Number of columns in panel.
	int ncols = b;
	
	// Number of rows in panel.
	int nrows = 2*b;
	
	// Locate block A(i:i+1,i)
	double *Aii = A + i * b + i * b * ldA;

	// Locate block Y(i) , Y used to generate Q later for valedating, Y from all panels are stored in one row-array.
	double *Yi = Y + i * b * ldY;

	// Prepare factor_task arguments
	factor_arg->m = nrows;
	factor_arg->n = ncols;
	factor_arg->A = Aii;	
	factor_arg->ldA = ldA;
	factor_arg->Y = Yi;	
	factor_arg->ldY = ldY;
	factor_arg->Z = Z;	
	factor_arg->ldZ = ldZ;
	factor_arg->barrier = NULL;

	// Start timer
	timer = get_time();
	// Factor block
	factor_task_seq(factor_arg, 1, 0);
	// Stop timer
	timer = get_time() - timer;

	// Compute Gflops of the factor task
	flops = factor_flops(nrows,ncols); 

	// write perf to file 
	fprintf(f,"%f ",flops / timer);	

	// Prepare update_task arguments
	update_arg->m = nrows;
	update_arg->n = ncols;
	update_arg->k = b;	
	update_arg->ldA = ldA;
	update_arg->B = Yi;	
	update_arg->ldB = ldY;
	update_arg->C = Aii;	
	update_arg->ldC = ldA;	

	flops = 0;
	// Compute Gflops of the factor task
	flops = update_flops(nrows,ncols,b);
	
	// Loop over block columns to update them.
	for (int j = i+1; j < N-1; j++)
	{
	    // Locate block A(j:j+1,i)
	    double *Aij = A + i * b + j * b * ldA;

	    update_arg->A = Aij;

	    // Start timer
	    timer = get_time();
	    // Update block
	    update_task_seq(update_arg, 1, 0);
	    // Stop timer

	    timer = get_time() - timer;
	    // Write perf to file
	    fprintf(f,"%f ",flops / timer);	    
	}
	
	fprintf(f,"\n");
	fflush(f);
	fflush(stdout);
	
	// Last block
	double *Aij = A + i * b + (N-1) * b * ldA;
	
	update_arg->A = Aij;
	update_arg->n = n - (N-1)*b;	

	// Update block
	update_task_seq(update_arg, 1, 0);
    }
    
    // Last two panels are reduced togither as on big panel

    // Number of columns in panel.
    int ncols = n - (N-2) * b;
    
    // Number of rows in panel.
    int nrows = ncols;
    
    // Locate block A(i:i+1,i:i+1)
    double *Aii = A + (N-2) * b + (N-2) * b * ldA;
    
    // Locate block Y(i)
    double *Yi = Y + (N-2) * b * ldY;

    // Prepare factor_task arguments
    factor_arg->m = nrows;
    factor_arg->n = ncols;
    factor_arg->A = Aii;	
    factor_arg->ldA = ldA;
    factor_arg->Y = Yi;	
    factor_arg->ldY = ldY;
    factor_arg->Z = Z;	
    factor_arg->ldZ = ldZ;	    
    factor_arg->barrier = NULL;
	
    // Factor block
    factor_task_seq(factor_arg, 1, 0);
    
    // Clean up
    free(factor_arg);
    free(update_arg);    
    
    fclose(f);
    
#undef Y
#undef Z    
}
