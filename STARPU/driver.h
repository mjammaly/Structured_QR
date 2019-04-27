#ifndef EXAMPLES_STR_QR_DRIVER_H
#define EXAMPLES_STR_QR_DRIVER_H

/**
 \file driver.h Driver functions which implement structured QR factorization using StarPU.
*/

/**
   @fn void parallel_str_qr(int n, int b, double *A, int ldA, double *Y, int ldY, int nth, int reserved, FILE *f)
   @brief Parallel structured QR factorization.

   @param[in] n Matrix order.
   @param[in] b blocking size.
   @param[in, out] A Pointer to the input matrix. The matrix will be replaced by R and W at the output.
   @param[in] ldA Leading dimention for imput matrix A.
   @param[in,out] Y Pointer to the matrix Y.
   @param[in] ldY Leading dimentio for the matrix Y.
   @param[in] nth Number of available threads/cores.
   @param[in] reserved Number of resereved threads/cores for the critical path.
   @param[in] f Pointer to the file where measurements will be writen to.
*/
void parallel_str_qr(int n, int b, double *A, int ldA, double *Y, int ldY, int nth, int reserved, FILE *f);

/**
   @fn void sequential_str_qr(int n, int b, double *A, int ldA, double *Y, int ldY, int nth, FILE *f)
   @brief Sequential structured QR factorization.

   @param[in] n Matrix order.
   @param[in] b blocking size.
   @param[in,out] A Pointer to the input matrix. The matrix will be replaced by R and W at the output.
   @param[in] ldA Leading dimention for imput matrix A.
   @param[in,out] Y Pointer to the matrix Y.
   @param[in] ldY Leading dimentio for the matrix Y.
   @param[in] nth Number of available threads/cores.
   @param[in] f Pointer to the file where measurements will be writen to.
*/
void sequential_str_qr(int n, int b, double *A, int ldA, double *Y, int ldY, int nth, FILE *f);




/**
   @fn void factor_seq(void *buffers[], void *args)
   @brief Wrapper for calling the sequential factor kernel.

   @param[in] buffers Pointer to a list of bufferes used in the sequential factor kernel.
   @param[in] args Pointer to a list of arguments used in the sequential factor kernel.
*/
static void factor_seq(void *buffers[], void *args);   

/**
   @fn void factor_par(void *buffers[], void *args)
   @brief Wrapper for calling the parallel factor kernel.
   
   @param[in] buffers Pointer to a list of bufferes used in the parallel factor kernel.
   @param[in] args Pointer to a list of arguments used in the parallel factor kernel.
*/
static void factor_par(void *buffers[], void *args);

/**
   @fn void update_seq(void *buffers[], void *args)
   @brief Wrapper for calling the sequential update kernel.
   
   @param[in] buffers Pointer to a list of bufferes used in the sequential update kernel.
   @param[in] args Pointer to a list of arguments used in the sequential update kernel.
*/
static void update_seq(void *buffers[], void *args);

/**
   @fn void update_par(void *buffers[], void *args)
   @brief Wrapper for calling the parallel update kernel.
   
   @param[in] buffers Pointer to a list of bufferes used in the parallel update kernel.
   @param[in] args Pointer to a list of arguments used in the parallel update kernel.
*/
static void update_par(void *buffers[], void *args);

/**
   @fn void record_task_time(FILE *f)
   @brief Record the execution time of a task to a file.   

   @param[in] f Pointer to a file where task time will be writen.
*/
static void record_task_time(FILE *f);
    
#endif
