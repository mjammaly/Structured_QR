#include <stdlib.h>
#include <stdio.h>
#include <cblas.h>
#include <pthread.h>
#include <hwloc.h>

#include "str_qr.h"
#include "print_matrix.h"
#include "tasks.h"
#include "utils.h"
#include "spin-barrier.h"

hwloc_topology_t topology;

// Initialize threads and invironments and calls "str_qr_par"  
void str_qr_par_init(int n, int b, double *A, int ldA, double *Y, int ldY, int nth)
{
    // initialize affinity
    hwloc_topology_init(&topology);
    hwloc_topology_load(topology);
    
#define Y(i,j) Y[(i) + (j) * ldY]
#define Z(i,j) Z[(i) + (j) * ldZ]

    // Allocate temp buffer
    int ldZ = 2*b;
    double *Z = (double *) malloc(nth * 2*b * ldZ * sizeof(double));

    // Allocate threads
    pthread_t threads[nth];
    
    // Allocate and initialise barrier
    pcp_barrier_t *barrier = pcp_barrier_create(nth);

    // Create local barrier for factor tasks
    factor_task_par_reconfigure(nth);
    
    // reset Y to 0s
    for(int j = 0; j < n; j++)
	for (int i = 0; i < ldY; i++)
	{
	    Y(i,j) = 0;
	}
    
    // reset Z to 0s   
    for(int j = 0; j < 2*b * nth; j++)
	for (int i = 0; i < ldZ; i++)
	{
	    Z(i,j) = 0;		
	}
    
    struct thrd_arg *t_arg = (struct thrd_arg*) malloc(nth*sizeof(*t_arg));

    // Prepare arguments and create threads for factor_task
    for(int t = 0; t < nth; t++)
    {
	// Thread arguemnts 
	t_arg[t].n = n;
	t_arg[t].b = b;
	t_arg[t].A = A;
	t_arg[t].ldA = ldA;
	t_arg[t].Y = Y;
	t_arg[t].ldY = ldY;    
	t_arg[t].Z = Z;
	t_arg[t].ldZ = ldZ;
	t_arg[t].nth = nth;
	t_arg[t].barrier = barrier;
	t_arg[t].rank = t;
	
	// Create thread
	pthread_create(&threads[t], NULL, str_qr_par, (void *)&t_arg[t]);
    }



    
    // Destroy threads
    for(int t = 0; t < nth; t++)
    {
	pthread_join(threads[t],NULL);	    
    }

    // Destroy barrier
    pcp_barrier_destroy(barrier);
    factor_task_par_finalize();
    
    // Clean up
    free(Z);
    free(t_arg);
    
#undef Y
#undef Z

    // destroy affinity 
    hwloc_topology_destroy(topology);    
}

// Factor a given block triangular matrix with overlapping blocks on the diagonal to Q and R using parallel kernels.
void *str_qr_par(void *ptr)
{
    
#define A(i,j) A[(i) + (j) * ldA]
#define Y(i,j) Y[(i) + (j) * ldY]
#define Z(i,j) Z[(i) + (j) * ldZ]
    
    struct thrd_arg *arg = (struct thrd_arg*) ptr;
    int n = arg->n;
    int b = arg->b;
    double *A = arg->A;
    int ldA = arg->ldA;
    double *Y = arg->Y;
    int ldY = arg->ldY;    
    double *Z = arg->Z;
    int ldZ = arg->ldZ;
    int nth = arg->nth;
    pcp_barrier_t *barrier = arg->barrier;
    int rank = arg->rank;
    
    // bind thread to core
    hwloc_obj_t obj;
    obj = hwloc_get_obj_by_type(topology, HWLOC_OBJ_CORE, rank);
    hwloc_set_cpubind(topology, obj->cpuset, HWLOC_CPUBIND_THREAD);
    if(rank == 0)
    {
	int cnt = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_CORE);
	printf("number of cores %d\n",cnt);
    }
    
    double timer = 0;
    double flops = 0;
    FILE *f;
	
    if(rank == 0)
    {
	// File to store measurements
	f = fopen("results.txt","a");
    }

    int N = iceil(n,b); // number of blocks to reduce.

    if(rank == 0)
    {
	fprintf(f,"%d %d\n",n,b);
	printf("%d %d\n",n,b);    
    }
    
    // Allocate tasks arguments
    struct factor_task_arg *factor_arg = (struct factor_task_arg*) malloc(sizeof(*factor_arg));
    struct update_task_arg *update_arg = (struct update_task_arg*) malloc(sizeof(*update_arg));


    
    // Loop over the blocks.	
    for(int i = 0; i < N-2; i++)
    {
	// Number of columns in block.
	int ncols = b;
	
	// Number of rows in block.
	int nrows = 2*b;
	
	// Locate block A(i:i+1,i)
	double *Aii = A + i * b + i * b * ldA;
	
	// Locate block Y(i), Y used to generate Q later for valedating.
	// Y from all blocks are stored in one row-array of size 2*b-by-n.
	double *Yi = Y + i * b * ldY;
		
	// Prepare arguments for factor_task
	factor_arg->m = nrows;
	factor_arg->n = ncols;
	factor_arg->A = Aii;
	factor_arg->ldA = ldA;
	factor_arg->Y = Yi;	
	factor_arg->ldY = ldY;
	factor_arg->Z = Z;	
	factor_arg->ldZ = ldZ;

	// sync
	pcp_barrier_wait(barrier);

	// start timer
	if(rank == 0)
	    timer = get_time();
		
	// factor panel
	factor_task_par(factor_arg, nth, rank);
	
	// sync
	pcp_barrier_wait(barrier);
	
	if(rank == 0)
	{
	    // stop timer
	    timer = get_time() - timer;

	    // Compute Gflops of the factor task
	    flops = factor_flops(nrows,ncols); 

	    // Write perf to file
	    fprintf(f,"%f ",flops / timer); 
	}

	// Compute Gflops of the factor task
	flops = update_flops(nrows,ncols,b);
	
	// Loop over block columns to update them.
	for (int j = i+1; j < N-1; j++)
	{
	    // Locate block A(j:j+1,i)
	    double *Aij = A + i * b + j * b * ldA;
	
	    // Prepare arguments for update_task	
	    update_arg->m = nrows;
	    update_arg->n = ncols;
	    update_arg->k = b;	
	    update_arg->A = Aij;
	    update_arg->ldA = ldA;
	    update_arg->B = Yi;	
	    update_arg->ldB = ldY;
	    update_arg->C = Aii;	
	    update_arg->ldC = ldA;	
		
	    // sync
	    pcp_barrier_wait(barrier);
	    
	    // start timer
	    if(rank == 0)
		timer = get_time();
	
	    // update panel
	    update_task_par(update_arg, nth, rank);

	    // sync
	    pcp_barrier_wait(barrier);
	
	    if(rank == 0)
	    {
		// stop timer
		timer = get_time() - timer;
				
		// Write perf to file
		fprintf(f,"%f ",flops / timer);		
	    }	    	    
	}

	
	if(rank == 0)
	{
	    fprintf(f,"\n");
	    fflush(f);
	    fflush(stdout);
	}

	// Last block, skip time measurement
	double *Aij = A + i * b + (N-1) * b * ldA;	    

	// Update_task arguments
	update_arg->A = Aij;
	update_arg->n = n - (N-1)*b;
		
	// sync
	pcp_barrier_wait(barrier);
	
	// update panel
	update_task_par(update_arg, nth, rank);
	
	// sync
	pcp_barrier_wait(barrier);	
	
    }
    
    // Last two blocks are reduced togither as one big block
    // Skip time measurment
    
    // Number of columns in block.
    int ncols = n - (N-2) * b;

    // Number of rows in block.
    int nrows = ncols;
    
    // Locate block A(i:i+1,i:i+1)
    double *Aii = A + (N-2) * b + (N-2) * b * ldA;
    
    // Locate block Y(i)
    double *Yi = Y + (N-2) * b * ldY;
    
    // Prepare arguments for factor_task
    factor_arg->m = nrows;
    factor_arg->n = ncols;
    factor_arg->A = Aii;	
    factor_arg->ldA = ldA;
    factor_arg->Y = Yi;	
    factor_arg->ldY = ldY;
    factor_arg->Z = Z;	
    factor_arg->ldZ = ldZ;	    
    
    // factor panel
    factor_task_par(factor_arg, nth, rank);	

    // clean up
    free(factor_arg);
    free(update_arg);
    if(rank == 0)
    {
	fclose(f);
    }
#undef A
#undef Y
#undef Z

}
