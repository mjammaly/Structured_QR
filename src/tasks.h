#ifndef STR_QR_FACTORIZATION_TASKS_H
#define STR_QR_FACTORIZATION_TASKS_H

#include "spin-barrier.h"


/**
   @file tasks.h
   @brief Contains the kernels used to perform structured QR factorization sequentially and in parallel. 
*/

/**
   @struct factor_task_arg
   @brief Arguments of factor task.
*/
struct factor_task_arg
{
    int m; /**< Brief Numer of rows in the input matrix. */
    int n; /**< Brief Numer of columns  in the input matrix. */
    double *A; /**< Brief Pointer to the input matrix. */
    int ldA; /**< Brief Leading dimension of the input matrix. */
    double *Y; /**< Brief Pointer to the temporary matrix Y. */
    int ldY; /**< Brief Leading dimension of the temporary matrix Y. */
    double *Z; /**< Brief Pointer to the temporary matrix Z. */
    int ldZ; /**< Brief Leading dimension of the temporary matrix Z. */
};


/**
   @fn void factor_task_seq(void *ptr)
   @brief Facot a block to Q and R recursively and sequentially.

   @param[in,out] ptr Pointer to an object of type struct factor_task_arg.
 */
void factor_task_seq(void *ptr);

/**
   @fn void factor_task_par(void *ptr, int nth, int rank)
   @brief Facot a block to Q and R recursively in parallel.

   @param[in,out] ptr Pointer to an object of type struct factor_task_arg.
   @param[in] nth Number of available threads/cores.
   @param[in] rank The rank of the calling thread.
 */
void factor_task_par(void *ptr, int nth, int rank);

/**
   @fn void factor_task_par_reconfigure(int nth)
   @brief Configure and initilize the environments for factor parallel kernels.
   
   @param[in] nth Number 
 */
void factor_task_par_reconfigure(int nth);

/**
   @fn factor_task_par_finalize(void)
   @brief Finilize and destory the configured environment for factor parallel kernels.   
 */
void factor_task_par_finalize(void);



/**
   @struct update_task_arg
   @brief Arguments of update task.
*/
struct update_task_arg
{
    int m; /**< Brief Numer of rows in the input matrix. */
    int n; /**< Brief Numer of columns  in the input matrix. */
    int k; /**< Brief The size of the triangular part of matrices B and C. */
    double *A; /**< Brief Pointer to the input matrix. */
    int ldA; /**< Brief Leading dimension of the input matrix. */
    double *B; /**< Brief Pointer to a trapezoidal matrix B.*/
    int ldB; /**< Brief Leading dimension of the matrix B.*/
    double *C; /**< Brief Pointer to a trapezoidal matrix C.*/
    int ldC; /**< Brief Leading dimension of the matrix C.*/   
};

/**
   @fn void update_task_seq(void *ptr)
   @brief Update a block from left using Q sequentially.

   @param[in,out] ptr Pointer to an object of type struct update_task_arg.
 */
void update_task_seq(void *ptr);

/**
   @fn void update_task_par(void *ptr, int nth, int rank)
   @brief Update a block from left using Q sequentially.

   @param[in,out] ptr Pointer to an object of type struct update_task_arg.
   @param[in] nth Number of available threads/cores.
   @param[in] rank The rank of the calling thread.
 */
void update_task_par(void *ptr, int nth, int rank);

/**
   @fn void update_task_par_reconfigure(int nth)
   @brief Configure and initilize the environments for update parallel kernels.
   
   @param[in] nth Number 
 */
void update_task_par_reconfigure(int nth);

/**
   @fn update_task_par_finalize(void)
   @brief Finilize and destory the configured environment for update parallel kernels.   
 */
void update_task_par_finalize(void);

#endif
