#include <stdio.h>
#include <stdlib.h>
#include <starpu.h>

#include "driver.h"
#include "../src/tasks.h"
#include "../src/utils.h"

#define STARPU_ENABLE_STATS 1

/////////////////////////////////////////////////////////////////
//                Setting the scheduling policy                //
/////////////////////////////////////////////////////////////////
//                                                             //
// To use eager scheduler: 1) Set Sched_Policy to "eager",     //
//                         2) Disable MANUAL_PRIORITY.         //
// To use prio scheduler: 1) Set Sched_Policy to "prio",       //
//                        2) Disable MANUAL_PRIORITY.          //
// To use manul prio scheduler: 1) Set Sched_Policy to "prio", //
//                              2) Enable MANUAL_PRIORITY.     //
//                                                             //
/////////////////////////////////////////////////////////////////

//static sched_policy = "eager";
static char Sched_Policy[] = "prio";

// Assign priority manually to tasks
#define ENABLE_MANUAL_PRIORITY


// a codelet that factorize a block sequentially
static struct starpu_codelet factor_seq_cl = {
    .name = "factor_seq",                // codelet name.
    .cpu_funcs = { factor_seq },         // pointers to the CPU implementations.
    .nbuffers = STARPU_VARIABLE_NBUFFERS // get buffer count from the task. 
};

// a codelet that factorize a block in parallel
static struct starpu_codelet factor_par_cl = {
    .name = "factor_par",                // codelet name
    .cpu_funcs = { factor_par },         // pointers to the CPU implementations
    .type = STARPU_SPMD,                 // parallelization type
    .max_parallelism = INT_MAX,          // max number of compined workers to use
    .nbuffers = STARPU_VARIABLE_NBUFFERS // get buffer count from the task. 
};

// a codelet that update a block sequentially
static struct starpu_codelet update_seq_cl = {
    .name = "update_seq",                // codelet name
    .cpu_funcs = { update_seq },         // pointers to the CPU implementations
    .nbuffers = STARPU_VARIABLE_NBUFFERS // get buffer count from the task. 
};

// a codelet that update a block in parallel
static struct starpu_codelet update_par_cl = {
    .name = "update_par",                // codelet name
    .cpu_funcs = { update_par },         // pointers to the CPU implementations
    .type = STARPU_SPMD,                 // parallelization type        
    .max_parallelism = INT_MAX,          // max number of compined workers to use
    .nbuffers = STARPU_VARIABLE_NBUFFERS // get buffer count from the task. 
};

// a wrapper that calles the sequential factor kernel    
static void factor_seq(void *buffers[], void *args)
{

    int nth = starpu_combined_worker_get_size(); // number of threads
    int rank = starpu_combined_worker_get_rank(); // my thread ID

    // allocate space for kernel's argument
    struct factor_task_arg *factor_arg = (struct factor_task_arg*) malloc(sizeof(*factor_arg));

    int nrows; // number of rows 
    int ncols; // number of columns
    // extract rows and columns from input arguments
    starpu_codelet_unpack_args(args, &nrows, &ncols);
    
    // prepare the kernel's inputs
    factor_arg->m = nrows;
    factor_arg->n = ncols;

    factor_arg->A = (double *)STARPU_MATRIX_GET_PTR(buffers[0]);
    factor_arg->ldA = STARPU_MATRIX_GET_LD(buffers[0]);
    factor_arg->Y = (double *)STARPU_MATRIX_GET_PTR(buffers[1]);
    factor_arg->ldY = STARPU_MATRIX_GET_LD(buffers[1]);
    factor_arg->Z = (double *)STARPU_MATRIX_GET_PTR(buffers[2]);
    factor_arg->ldZ = STARPU_MATRIX_GET_LD(buffers[2]);

    // call the kernel
    factor_task_seq(factor_arg);

    // clean up
    free(factor_arg);
}

// a wrapper that calles the parallel factor kernel
static void factor_par(void *buffers[], void *args)
{
    // a wrapper that calles the sequential factor kernel        
    struct factor_task_arg *factor_arg = (struct factor_task_arg*) malloc(sizeof(*factor_arg));

    int nth = starpu_combined_worker_get_size(); // number of threads
    int rank =  starpu_combined_worker_get_rank(); // my thread ID
    
    int nrows; // number of rows
    int ncols; // number of columns
    // extract rows and columns from input arguments    
    starpu_codelet_unpack_args(args, &nrows, &ncols);

    // prepare the kernel's inputs
    factor_arg->m = nrows;
    factor_arg->n = ncols;

    factor_arg->A = (double *)STARPU_MATRIX_GET_PTR(buffers[0]);
    factor_arg->ldA = STARPU_MATRIX_GET_LD(buffers[0]);
    factor_arg->Y = (double *)STARPU_MATRIX_GET_PTR(buffers[1]);
    factor_arg->ldY = STARPU_MATRIX_GET_LD(buffers[1]);
    factor_arg->Z = (double *)STARPU_MATRIX_GET_PTR(buffers[2]);
    factor_arg->ldZ = STARPU_MATRIX_GET_LD(buffers[2]);
    
    // call the kernel
    factor_task_par(factor_arg, nth, rank);

    // clean up
    free(factor_arg);
}

// a wrapper that calles the sequential update kernel
static void update_seq(void *buffers[], void *args)
{

    int nth = starpu_combined_worker_get_size(); // number of threads
    int rank = starpu_combined_worker_get_rank(); // my thread ID
    
    // allocate space for kernel's argument
    struct update_task_arg *update_arg = (struct update_task_arg*) malloc(sizeof(*update_arg));

    int nrows; // number of rows
    int ncols; // number of columns
    int b; // block size

    // extract rows, columns and b from input arguments
    starpu_codelet_unpack_args(args, &nrows, &ncols, &b);

    // prepare the kernel's inputs
    update_arg->m = nrows;
    update_arg->n = ncols;
    update_arg->k = b;
    
    update_arg->A = (double *)STARPU_MATRIX_GET_PTR(buffers[0]);
    update_arg->ldA = STARPU_MATRIX_GET_LD(buffers[0]);
    update_arg->B = (double *)STARPU_MATRIX_GET_PTR(buffers[1]);	
    update_arg->ldB = STARPU_MATRIX_GET_LD(buffers[1]);
    update_arg->C = (double *)STARPU_MATRIX_GET_PTR(buffers[2]);	
    update_arg->ldC = STARPU_MATRIX_GET_LD(buffers[2]);	

    // call the kernel
    update_task_seq(update_arg);	

    // clean up
    free(update_arg);
}
    
// a wrapper that calles the parallel update kernel
static void update_par(void *buffers[], void *args)
{
    
    struct update_task_arg *update_arg = (struct update_task_arg*) malloc(sizeof(*update_arg));

    int nth = starpu_combined_worker_get_size();
    int rank = starpu_combined_worker_get_rank();

    int nrows; // number of rows
    int ncols; // number of columns
    int b; // block size
    // extract rows, columns and b from input arguments
    starpu_codelet_unpack_args(args, &nrows, &ncols, &b); //, &nth, &rank);

    // prepare the kernel's inputs
    update_arg->m = nrows;
    update_arg->n = ncols;
    update_arg->k = b;
    
    update_arg->A = (double *)STARPU_MATRIX_GET_PTR(buffers[0]);		
    update_arg->ldA = STARPU_MATRIX_GET_LD(buffers[0]);
    update_arg->B = (double *)STARPU_MATRIX_GET_PTR(buffers[1]);	
    update_arg->ldB = STARPU_MATRIX_GET_LD(buffers[1]);
    update_arg->C = (double *)STARPU_MATRIX_GET_PTR(buffers[2]);	
    update_arg->ldC = STARPU_MATRIX_GET_LD(buffers[2]);	

    // call the kernel
    update_task_par(update_arg, nth, rank);

    // clean up
    free(update_arg);
}


// record task time to file
static void record_task_time(FILE *f)
{
    double length = 0;

    // get the task handle
    struct starpu_task *task = starpu_task_get_current();
    if(task == NULL)
	printf("Faild to get current task\n");
    
    int nth = starpu_combined_worker_get_size(); // number of threads
    int rank = starpu_combined_worker_get_rank(); // my thread ID
    
    // get task profiling info
    struct starpu_profiling_task_info *info = task->profiling_info;

    if(info == NULL)
	printf("Faild to get task's info\n");

    // compute the elapsed time
    length = starpu_timing_timespec_delay_us(&info->start_time, &info->end_time);

    // find the taks type and its ID
    // critical factor -> 0
    // critical update -> 1
    // non-critical update -> 2
    // record task type ID and elapsed time in the file 
    if(strstr(task->name,"critical_factor") != NULL)
    {
     	fprintf(f,"0 %f\n",length * 1e-6);	
    }
    else if(strstr(task->name,"critical_update") != NULL)
    {
	fprintf(f,"1 %f\n",length * 1e-6);	
    }
    else if(strstr(task->name,"normal_update") != NULL)
    {
	fprintf(f,"2 %f\n",length * 1e-6);
    }
    else
    {
	printf("WARNING: unrecognized task type !!!\n");	
    }
}



// Reduce square matrix A to upper triangular matrix U using QR decomposition.
// A is upper triangular with interleaved block diagonal.
// Parallel critical tasks are multi-threaded.
void parallel_str_qr(int n, int b, double *A, int ldA, double *Y, int ldY, int nth, int reserved, FILE *f)
{
#define A(i,j) A[(i) + (j) * ldA]
#define Y(i,j) Y[(i) + (j) * ldY]
#define Z(i,j) Z[(i) + (j) * ldZ]


    // Allocate temp buffer
    int ldZ = 2*b;
    double *Z = (double *) malloc(nth * 2*b * ldZ * sizeof(double));
    
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
    
    starpu_profiling_status_set(STARPU_PROFILING_ENABLE);
    
    // Initialize StarPU
    struct starpu_conf conf;    

    starpu_conf_init(&conf);
    conf.single_combined_worker = 1;
    conf.ncpus = nth;
    conf.sched_policy_name = Sched_Policy;

    int ret = starpu_init(&conf);
    if (ret != 0)
    {
	printf("Error initializing StarPU !!!\n");
        return;
    }
    
//////////////////////////////////////////////////////////////////////////////////////////////////////    

    /////////////////////
    // Create contexts //
    /////////////////////
    
    int max_workers = starpu_worker_get_count(); // max number of used workers in paralle context
    int *critical_workers = malloc(reserved * sizeof(int)); // allocate pointers to critical context workers
    int *normal_workers = malloc((max_workers - reserved) * sizeof(int)); // allocate pointers to normal context workers    
    printf("max workers = %d\n",max_workers);
    
    // select workers for critical context
     for (int i = 0; i < reserved; i++)
	 critical_workers[i] = i;     

     // select workers for normal context
     for (int i = 0; i < (max_workers-reserved); i++)
	 normal_workers[i] = i + reserved;     

     // create normal context
     int normal_ctx_id = starpu_sched_ctx_create(normal_workers, (max_workers-reserved), "normal_ctx",
						 STARPU_SCHED_CTX_POLICY_NAME, Sched_Policy,
#ifdef ENABLE_MANUAL_PRIORITY
						 STARPU_SCHED_CTX_POLICY_MIN_PRIO, -5000,
						 STARPU_SCHED_CTX_POLICY_MAX_PRIO, 5000,
#endif
						 0);

     // create critical context
     int critical_ctx_id = starpu_sched_ctx_create(critical_workers, reserved, "critical_ctx",
						   STARPU_SCHED_CTX_POLICY_NAME, "peager",
#ifdef ENABLE_MANUAL_PRIORITY						   
						   STARPU_SCHED_CTX_POLICY_MIN_PRIO, -5000,
						   STARPU_SCHED_CTX_POLICY_MAX_PRIO, 5000,
#endif						   
						   0);
     
     printf("Sched = %s\n",Sched_Policy);


     int normal_min =  starpu_sched_ctx_get_min_priority(normal_ctx_id);
     int normal_max =  starpu_sched_ctx_get_max_priority(normal_ctx_id);    
     
     int critical_min =  starpu_sched_ctx_get_min_priority(critical_ctx_id);
     int critical_max =  starpu_sched_ctx_get_max_priority(critical_ctx_id);    
     
//////////////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////
    // Allocate handles //
    //////////////////////
    
    // Compute the number of blocks.
    int N = iceil(n, b);

    // Allocate array of task handles for A
    starpu_data_handle_t **A_handles =  malloc(N * sizeof(starpu_data_handle_t *));
    for(int i = 0; i < N-2; i++)
    {
	A_handles[i] = malloc( N * sizeof(starpu_data_handle_t));

	for (int j = 0; j < N; j++)
	{
            starpu_matrix_data_register(
                &A_handles[i][j],                     // handle
                STARPU_MAIN_RAM,                      // memory node
                (uintptr_t)(A + i * b + j * b * ldA), // pointer
                ldA,                                  // leading dimension
                2*b,                                  // row count
                min(b, n - j*b),                      // column count
                sizeof(double));                      // element size
	}
    }

    // last two rows
    {	
	int i = N-2;
	A_handles[i] = malloc( N * sizeof(starpu_data_handle_t));
	for (int j = 0; j < N-2; j++)
	{
            starpu_matrix_data_register(
                &A_handles[i][j],                     // handle
                STARPU_MAIN_RAM,                      // memory node
                (uintptr_t)(A + i * b + j * b * ldA), // pointer
                ldA,                                  // leading dimension
                2*b,                                  // row count
                b,                                    // column count
                sizeof(double));                      // element size
	}

	// last two column
	int j = N-2;
	starpu_matrix_data_register(
	    &A_handles[i][j],                     // handle
	    STARPU_MAIN_RAM,                      // memory node
	    (uintptr_t)(A + i * b + j * b * ldA), // pointer
	    ldA,                                  // leading dimension
	    2*b,                                  // row count
	    (n - j*b),                            // column count
	    sizeof(double));                      // element size		
    }

    // Allocate array of task handles for Y    
    starpu_data_handle_t *Y_handles = malloc(N * sizeof(starpu_data_handle_t));
    for (int j = 0; j < N; j++)
    {
	starpu_matrix_data_register(
	    &Y_handles[j],                // handle
	    STARPU_MAIN_RAM,              // memory node
	    (uintptr_t)(Y + j * b * ldY), // pointer
	    ldY,                          // leading dimension
	    2*b,                          // row count
	    min(b, n-j*b),                // column count
	    sizeof(double));              // element size	
    }
    // last two rows
    starpu_data_handle_t *YN_handle = malloc(1 * sizeof(starpu_data_handle_t));
    {
	int j = N-2;
	starpu_matrix_data_register(
	    YN_handle,                    // handle
	    STARPU_MAIN_RAM,              // memory node
	    (uintptr_t)(Y + j * b * ldY), // pointer
	    ldY,                          // leading dimension
	    2*b,                          // row count
	    (n-j*b),                      // column count
	    sizeof(double));              // element size	
    }
    
    // Allocate task handles for Z 
    starpu_data_handle_t *Z_handle = malloc(1 * sizeof(starpu_data_handle_t));    
    starpu_matrix_data_register(
	Z_handle,        // handle
	STARPU_MAIN_RAM, // memory node
	(uintptr_t)(Z),  // pointer
	ldZ,             // leading dimension
	2*b,             // row count
	2*b*nth,         // column count
	sizeof(double)); // element size	
    
//////////////////////////////////////////////////////////////////////////////////////////////////////
    
    // create internal barrier for factor tasks
    factor_task_par_reconfigure(reserved);

    
    //////////////////
    // Insert tasks //
    //////////////////

    printf("start inserting tasks.......\n");

    // start the timer
    double exe_time = get_time();
    // reset profiling counteres
    starpu_profiling_init();


    // Loop through block rows.    
#ifdef ENABLE_MANUAL_PRIORITY
    int prio = 0; // task priority value
    int x = 0; // number of trailing block rows
#endif    
    for(int i = 0; i < N-2; i++)
    {
	// Number of columns in block.
	int ncols = b;
	
	// Number of rows in block.
	int nrows = 2*b;
	
#ifdef ENABLE_MANUAL_PRIORITY
	// find priority for current task
	x = N - i -1;
	prio = x*2-1;
#endif
        // Insert factor task for U(i,i).
	if(i == 0) // first row, less dependencies
	{
            starpu_task_insert(
		&factor_par_cl,
		STARPU_NAME, "critical_factor",				
		STARPU_SCHED_CTX, critical_ctx_id,
#ifdef ENABLE_MANUAL_PRIORITY		
		STARPU_PRIORITY,(critical_min + prio),
#endif
		STARPU_VALUE, &nrows, sizeof(int),
		STARPU_VALUE, &ncols, sizeof(int),
		STARPU_RW, A_handles[i][i],
		STARPU_RW, Y_handles[i],
		STARPU_RW, *Z_handle,
		STARPU_CALLBACK_WITH_ARG, &record_task_time, f,
		0);
	}
	else
	{
            starpu_task_insert(
		&factor_par_cl,
		STARPU_NAME, "critical_factor",
		STARPU_SCHED_CTX, critical_ctx_id,
#ifdef ENABLE_MANUAL_PRIORITY		
		STARPU_PRIORITY, (critical_min + prio),
#endif		
		STARPU_VALUE, &nrows, sizeof(int),
		STARPU_VALUE, &ncols, sizeof(int),
		STARPU_RW, A_handles[i][i],
		STARPU_RW, Y_handles[i],
		STARPU_RW, *Z_handle,
		STARPU_R, A_handles[i-1][i],
		STARPU_CALLBACK_WITH_ARG, &record_task_time, f,		
		0);	    
	}
	
	// Loop over block columns to update them.
	for (int j = i+1; j < N; j++)	    
	{
	    int ncols = min(b , n-(j*b));
	    // Insert update task for U(i,j).
#ifdef ENABLE_MANUAL_PRIORITY
	    prio -= 1;
	    prio = (prio > x)? prio : x;
#endif		
	    // first block afetr factor task is critical
	    if(j == i+1)
	    {
		if(i == 0) // first row, less dependencies
		{
		    starpu_task_insert(
			&update_par_cl,
			STARPU_NAME, "critical_update",					
			STARPU_SCHED_CTX, critical_ctx_id,
#ifdef ENABLE_MANUAL_PRIORITY			
			STARPU_PRIORITY, (critical_min + prio),
#endif			
			STARPU_VALUE, &nrows, sizeof(int),
			STARPU_VALUE, &ncols, sizeof(int),
			STARPU_VALUE, &b, sizeof(int),
			STARPU_RW, A_handles[i][j],
			STARPU_R, Y_handles[i],
			STARPU_R, A_handles[i][i],
			STARPU_CALLBACK_WITH_ARG, &record_task_time, f,
			0);		    		
		}
		else
		{
		    starpu_task_insert(
			&update_par_cl,
			STARPU_NAME, "critical_update",					
			STARPU_SCHED_CTX, critical_ctx_id,
#ifdef ENABLE_MANUAL_PRIORITY			
			STARPU_PRIORITY, (critical_min + prio),
#endif			
			STARPU_VALUE, &nrows, sizeof(int),
			STARPU_VALUE, &ncols, sizeof(int),
			STARPU_VALUE, &b, sizeof(int),
			STARPU_RW, A_handles[i][j],
			STARPU_R, Y_handles[i],
			STARPU_R, A_handles[i][i],
			STARPU_R, A_handles[i-1][j],
		        STARPU_CALLBACK_WITH_ARG, &record_task_time, f,
			0);						
		}
	    }
	    // the rest of update tasks are normal
	    else
	    {	    
		if(i == 0) // first row, less dependencies
		{
		    starpu_task_insert(
			&update_seq_cl,
			STARPU_NAME, "normal_update",				       
			STARPU_SCHED_CTX, normal_ctx_id,
#ifdef ENABLE_MANUAL_PRIORITY			
			STARPU_PRIORITY, (normal_min + prio),
#endif			
			STARPU_VALUE, &nrows, sizeof(int),
			STARPU_VALUE, &ncols, sizeof(int),
			STARPU_VALUE, &b, sizeof(int),
			STARPU_RW, A_handles[i][j],
			STARPU_R, Y_handles[i],
			STARPU_R, A_handles[i][i],
			STARPU_CALLBACK_WITH_ARG, &record_task_time, f,
			0);				
		}
		else
		{
		    starpu_task_insert(
			&update_seq_cl,
			STARPU_NAME, "normal_update",			
			STARPU_SCHED_CTX, normal_ctx_id,
#ifdef ENABLE_MANUAL_PRIORITY			
			STARPU_PRIORITY, (normal_min + prio),
#endif
			STARPU_VALUE, &nrows, sizeof(int),
			STARPU_VALUE, &ncols, sizeof(int),
			STARPU_VALUE, &b, sizeof(int),
			STARPU_RW, A_handles[i][j],
			STARPU_R, Y_handles[i],
			STARPU_R, A_handles[i][i],
			STARPU_R, A_handles[i-1][j],
			STARPU_CALLBACK_WITH_ARG, &record_task_time, f,
			0);						
		}
	    }
	}
    }

    // Last two block rows are reduced togither as one big block       
    {
	int i = N-2;
	    
	// Number of columns in block.
	int ncols = n - i * b;
	
	// Number of rows in block.
	int nrows = ncols;
	
	// Insert factor task for U(i,i).
	starpu_task_insert(
	    &factor_par_cl,
	    STARPU_NAME, "critical_factor",
	    STARPU_SCHED_CTX, critical_ctx_id,
#ifdef ENABLE_MANUAL_PRIORITY	    
	    STARPU_PRIORITY, (critical_min),
#endif
	    STARPU_VALUE, &nrows, sizeof(int),
	    STARPU_VALUE, &ncols, sizeof(int),
	    STARPU_RW, A_handles[i][i],
	    STARPU_RW, *YN_handle,
	    STARPU_RW, *Z_handle,
	    STARPU_R, A_handles[i-1][i],
	    STARPU_R, A_handles[i-1][i+1],
	    STARPU_CALLBACK_WITH_ARG, &record_task_time, f,	    
	    0);
    }
//////////////////////////////////////////////////////////////////////////////////////////////////////    

    
    // wait for all tasks to finish
    starpu_task_wait_for_all();
       
    // compute execution time
    exe_time = get_time() - exe_time;
    
    // distroy the local barrier
    factor_task_par_finalize();    

    printf("graph execution done.\ncleaning up.......\n");
    	
    // Clean up.    
    // A_handles
    for(int i = 0; i < N-2; i++)
    {
    	for (int j = 0; j < N; j++)
    	{
            starpu_data_unregister(A_handles[i][j]);
    	}
    }
    // last two rows
    {	
    	int i = N-2;	
    	for (int j = 0; j < N-1; j++)
    	{
            starpu_data_unregister(A_handles[i][j]);	    
    	}	
    }

    for (int i = 0; i < N-2; i++)	
	free(A_handles[i]);	

    free(A_handles);	

    // Y_handles
    for (int j = 0; j < N; j++)
    {
    	starpu_data_unregister(Y_handles[j]);		
    }
    free(Y_handles);
    
    starpu_data_unregister(*YN_handle);		
    free(YN_handle);	        

    // Z_handles
    starpu_data_unregister(*Z_handle);		
    free(Z_handle);	

    // critical context
    starpu_sched_ctx_delete(critical_ctx_id);
    starpu_sched_ctx_delete(normal_ctx_id);    
    
    // workers list 
    free(critical_workers);
    free(normal_workers);    


    // Stop the runtime system.
    starpu_shutdown();


    // record the total execution time
    fprintf(f,"%f\n",exe_time);
    
// #undef HANDLE
#undef A
#undef Y
#undef Z        
}


































// Reduce square matrix A to upper triangular matrix U using QR decomposition.
// A is upper triangular with interleaved block diagonal.
// Tasks are single threaded.
void sequential_str_qr(int n, int b, double *A, int ldA, double *Y, int ldY, int nth, FILE *f)
{
#define A(i,j) A[(i) + (j) * ldA]
#define Y(i,j) Y[(i) + (j) * ldY]
#define Z(i,j) Z[(i) + (j) * ldZ]

    // Allocate temp buffer
    int ldZ = 2*b;
    double *Z = (double *) malloc(nth * 2*b * ldZ * sizeof(double));
    
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
    
    starpu_profiling_status_set(STARPU_PROFILING_ENABLE);

    // Initialize StarPU
    struct starpu_conf conf;    
    starpu_conf_init(&conf);

    conf.single_combined_worker = 1;
    conf.ncpus = nth;
    conf.sched_policy_name = Sched_Policy;

#ifdef ENABLE_MANUAL_PRIORITY    
    conf.global_sched_ctx_min_priority = -5000;
    conf.global_sched_ctx_max_priority = 5000;
#endif

    int ret = starpu_init(&conf);
    if (ret != 0)
    {
	printf("Error initializing StarPU !!!\n");
        return;
    }

    int global_min =  starpu_sched_ctx_get_min_priority(0);
    int global_max =  starpu_sched_ctx_get_max_priority(0);

    
    printf("Sched = %s\n",Sched_Policy);


    
//////////////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////
    // Allocate handles //
    //////////////////////
         
    // Compute the number of blocks.
    int N = iceil(n, b);

    // Allocate array of task handles for A
    starpu_data_handle_t **A_handles =  malloc(N * sizeof(starpu_data_handle_t *));
    for(int i = 0; i < N-2; i++)
    {
	A_handles[i] = malloc( N * sizeof(starpu_data_handle_t));

	for (int j = 0; j < N; j++)
	{
            starpu_matrix_data_register(
                &A_handles[i][j],                     // handle
                STARPU_MAIN_RAM,                      // memory node
                (uintptr_t)(A + i * b + j * b * ldA), // pointer
                ldA,                                  // leading dimension
                2*b,                                  // row count
                min(b, n - j*b),                      // column count
                sizeof(double));                      // element size
	}
    }

    // last two rows
    {	
	int i = N-2;
	A_handles[i] = malloc( N * sizeof(starpu_data_handle_t));
	for (int j = 0; j < N-2; j++)
	{
            starpu_matrix_data_register(
                &A_handles[i][j],                     // handle
                STARPU_MAIN_RAM,                      // memory node
                (uintptr_t)(A + i * b + j * b * ldA), // pointer
                ldA,                                  // leading dimension
                2*b,                                  // row count
                b,                                    // column count
                sizeof(double));                      // element size
	}

	// last two column
	int j = N-2;
	starpu_matrix_data_register(
	    &A_handles[i][j],                     // handle
	    STARPU_MAIN_RAM,                      // memory node
	    (uintptr_t)(A + i * b + j * b * ldA), // pointer
	    ldA,                                  // leading dimension
	    2*b,                                  // row count
	    (n - j*b),                            // column count
	    sizeof(double));                      // element size		
    }

    // Allocate array of task handles for Y    
    starpu_data_handle_t *Y_handles = malloc(N * sizeof(starpu_data_handle_t));
    for (int j = 0; j < N; j++)
    {
	starpu_matrix_data_register(
	    &Y_handles[j],                 // handle
	    STARPU_MAIN_RAM,               // memory node
	    (uintptr_t)(Y + j * b * ldY),  // pointer
	    ldY,                           // leading dimension
	    2*b,                           // row count
	    min(b, n-j*b),                 // column count
	    sizeof(double));               // element size	
    }
    // last two rows
    starpu_data_handle_t *YN_handle = malloc(1 * sizeof(starpu_data_handle_t));
    {
	int j = N-2;
	starpu_matrix_data_register(
	    YN_handle,                 // handle
	    STARPU_MAIN_RAM,               // memory node
	    (uintptr_t)(Y + j * b * ldY), // pointer
	    ldY,                           // leading dimension
	    2*b,                           // row count
	    (n-j*b),                 // column count
	    sizeof(double));               // element size	
    }
    
    // Allocate task handles for Z 
    starpu_data_handle_t *Z_handle = malloc(1 * sizeof(starpu_data_handle_t));    
    starpu_matrix_data_register(
	Z_handle,                    // handle
	STARPU_MAIN_RAM,               // memory node
	(uintptr_t)(Z),                // pointer
	ldZ,                           // leading dimension
	2*b,                           // row count
	2*b*nth,                        // column count
	sizeof(double));               // element size	
    
//////////////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////
    // Insert tasks //
    //////////////////

    printf("start inserting tasks.......\n");

    // start the timer
    double exe_time = get_time();

    // reset profiling conters 
    starpu_profiling_init();

    // Insert tasks    
    // Loop through block rows.
#ifdef ENABLE_MANUAL_PRIORITY
    int prio = 0;
    int x = 0; // number of trailing block rows
#endif    
    for(int i = 0; i < N-2; i++)
    {
	// Number of columns in block.
	int ncols = b;
	
	// Number of rows in block.
	int nrows = 2*b;
#ifdef ENABLE_MANUAL_PRIORITY
	x = N - i -1;
	prio = x*2-1;
#endif       	
        // Insert factor task for U(i,i).
	if(i == 0) // first row, less dependencies
	{
            starpu_task_insert(
		&factor_seq_cl,
		STARPU_NAME, "critical_factor",
#ifdef ENABLE_MANUAL_PRIORITY		
		STARPU_PRIORITY,(global_min + prio),
#endif		
		STARPU_VALUE, &nrows, sizeof(int),
		STARPU_VALUE, &ncols, sizeof(int),
		STARPU_RW, A_handles[i][i],
		STARPU_RW, Y_handles[i],
		STARPU_RW, *Z_handle,
		STARPU_CALLBACK_WITH_ARG, &record_task_time, f,
		0);
	}
	else
	{
            starpu_task_insert(
		&factor_seq_cl,
		STARPU_NAME, "critical_factor",
#ifdef ENABLE_MANUAL_PRIORITY		
		STARPU_PRIORITY,(global_min + prio),
#endif		
		STARPU_VALUE, &nrows, sizeof(int),
		STARPU_VALUE, &ncols, sizeof(int),
		STARPU_RW, A_handles[i][i],
		STARPU_RW, Y_handles[i],
		STARPU_RW, *Z_handle,
		STARPU_R, A_handles[i-1][i],
		STARPU_CALLBACK_WITH_ARG, &record_task_time, f,		
		0);	    
	}
	// Loop over block columns to update them.
	for (int j = i+1; j < N; j++)	    
	{
	    int ncols = min(b , n-(j*b));
	    // Insert update task for U(i,j).
#ifdef ENABLE_MANUAL_PRIORITY
	    prio -= 1;
	    prio = (prio > x)? prio : x;
#endif
	    // first block afetr factor task is critical
	    if(j == i+1)
	    {
		if(i == 0) // first row, less dependencies
		{
		    starpu_task_insert(
			&update_seq_cl,
			STARPU_NAME, "critical_update",
#ifdef ENABLE_MANUAL_PRIORITY			
			STARPU_PRIORITY, (global_min + prio),
#endif
			STARPU_VALUE, &nrows, sizeof(int),
			STARPU_VALUE, &ncols, sizeof(int),
			STARPU_VALUE, &b, sizeof(int),
			STARPU_RW, A_handles[i][j],
			STARPU_R, Y_handles[i],
			STARPU_R, A_handles[i][i],
			STARPU_CALLBACK_WITH_ARG, &record_task_time, f,
			0);				
		}
		else
		{
		    starpu_task_insert(
			&update_seq_cl,
			STARPU_NAME, "critical_update",
#ifdef ENABLE_MANUAL_PRIORITY			
			STARPU_PRIORITY, (global_min + prio),					    
#endif
			STARPU_VALUE, &nrows, sizeof(int),
			STARPU_VALUE, &ncols, sizeof(int),
			STARPU_VALUE, &b, sizeof(int),
			STARPU_RW, A_handles[i][j],
			STARPU_R, Y_handles[i],
			STARPU_R, A_handles[i][i],
			STARPU_R, A_handles[i-1][j],
			STARPU_CALLBACK_WITH_ARG, &record_task_time, f,
			0);						
		}	    		
	    }
	    // the rest of update tasks are normal	    
	    else
	    {
		if(i == 0) // first row, less dependencies
		{
		    starpu_task_insert(
			&update_seq_cl,
			STARPU_NAME, "normal_update",			
#ifdef ENABLE_MANUAL_PRIORITY
			STARPU_PRIORITY, (global_min + prio),
#endif
			STARPU_VALUE, &nrows, sizeof(int),
			STARPU_VALUE, &ncols, sizeof(int),
			STARPU_VALUE, &b, sizeof(int),
			STARPU_RW, A_handles[i][j],
			STARPU_R, Y_handles[i],
			STARPU_R, A_handles[i][i],
			STARPU_CALLBACK_WITH_ARG, &record_task_time, f,
			0);				
		}
		else
		{
		    starpu_task_insert(
			&update_seq_cl,
			STARPU_NAME, "normal_update",			
#ifdef ENABLE_MANUAL_PRIORITY
			STARPU_PRIORITY, (global_min + prio),					    
#endif
			STARPU_VALUE, &nrows, sizeof(int),
			STARPU_VALUE, &ncols, sizeof(int),
			STARPU_VALUE, &b, sizeof(int),
			STARPU_RW, A_handles[i][j],
			STARPU_R, Y_handles[i],
			STARPU_R, A_handles[i][i],
			STARPU_R, A_handles[i-1][j],
			STARPU_CALLBACK_WITH_ARG, &record_task_time, f,
			0);						
		}	    		
	    }	    
	}
    }

    // Last two block rows are reduced togither as one big block       
    {
	int i = N-2;
	    
	// Number of columns in block.
	int ncols = n - i * b;
	
	// Number of rows in block.
	int nrows = ncols;
	
	// Insert factor task for U(i,i).
	starpu_task_insert(
	    &factor_seq_cl,
	    STARPU_NAME, "critical_factor",	    
#ifdef ENABLE_MANUAL_PRIORITY
	    STARPU_PRIORITY, (global_min),
#endif
	    STARPU_VALUE, &nrows, sizeof(int),
	    STARPU_VALUE, &ncols, sizeof(int),
	    STARPU_RW, A_handles[i][i],
	    STARPU_RW, *YN_handle,
	    STARPU_RW, *Z_handle,
	    STARPU_R, A_handles[i-1][i],
	    STARPU_R, A_handles[i-1][i+1],
	    STARPU_CALLBACK_WITH_ARG, &record_task_time, f,	    
	    0);
    }
//////////////////////////////////////////////////////////////////////////////////////////////////////
    
    // wait for all tasks to finish
    starpu_task_wait_for_all();
        
    // compute execution time
    exe_time = get_time() - exe_time;
    
    printf("graph execution done.\ncleaning up.......\n");
    	
    // Clean up.    
    // A_handles
    for(int i = 0; i < N-2; i++)
    {
    	for (int j = 0; j < N; j++)
    	{
            starpu_data_unregister(A_handles[i][j]);
    	}
    }
    // last two rows
    {	
    	int i = N-2;	
    	for (int j = 0; j < N-1; j++)
    	{
            starpu_data_unregister(A_handles[i][j]);	    
    	}	
    }

    for (int i = 0; i < N-2; i++)	
	free(A_handles[i]);	

    free(A_handles);	

    // Y_handles
    for (int j = 0; j < N; j++)
    {
    	starpu_data_unregister(Y_handles[j]);		
    }
    free(Y_handles);
    
    starpu_data_unregister(*YN_handle);		
    free(YN_handle);	        

    // Z_handles
    starpu_data_unregister(*Z_handle);		
    free(Z_handle);	
    
    // Stop the runtime system.
    starpu_shutdown();

    // record the total execution time
    fprintf(f,"%f\n",exe_time);
    
// #undef HANDLE
#undef A
#undef Y
#undef Z        
}

