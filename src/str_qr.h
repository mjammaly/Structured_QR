#ifndef STR_QR_H
#define STR_QR_H

#include "spin-barrier.h"

/**
   @file str_qr.h
   @brief Contains functions to perform structured QR sequentially and in parallel.
*/

/**
   @struct thrd_arg
   @brief An object contains the arguments needed by the Pthread called function (str_qr_par) to perform structured QR factorization.
   
*/
struct thrd_arg
{
    int n; /**< Brief Matrix order.*/
    int b; /**< Brief Blocking size.*/
    double *A; /**< BriefPointer to the input matrix.*/
    int ldA; /**< Brief Leading dimension of the input matrixs.*/
    double *Y; /**< Brief Pointer to the temporary matrix Y.*/
    int ldY; /**< Brief Leading dimension of the matrix Y.*/   
    double *Z; /**< Brief Pointer to the temporary matrix Z.*/
    int ldZ; /**< Brief Leading dimension of the matrix Z.*/
    int nth; /**< Brief Number of available threads.*/
    pcp_barrier_t *barrier; /**< Brief Pointer to a local barier.*/
    int rank; /**< Brief Rank of the current thread.*/
};


/**
   @fn void str_qr_seq(int n, int b, double *A, int ldA, double *Y, int ldY)
   @brief Factor a structured matrix to R and Q sequentially
   
   @param[in] n Order of the input matrix.
   @param[in] b Size of overlapping blocks.
   @param[in,out] A Pointer to the input matrix. At the exit will be replaced by the output matrix R.
   @param[in] ldA Leading dimension of the input matrix.
   @param[in,out] Y Pointer to the temporary matrix Y.
   @param[in] ldY Leading dimension of the matrix Y.
*/
void str_qr_seq(int n, int b, double *A, int ldA, double *Y, int ldY);

/**
   @fn void str_qr_par_init(int n, int b, double *A, int ldA, double *Y, int ldY, int nth)
   @brief Initialize threads and calles (str_qr_par).
   
   @param[in] n Order of the input matrix.
   @param[in] b Size of overlapping blocks.
   @param[in,out] A Pointer to the input matrix. At exit will be replaced by the output matrix R.
   @param[in] ldA Leading dimension of the input matrix.
   @param[in,out] Y Pointer to the temporary matrix Y. At exit will contain the output matrix Y.
   @param[in] ldY Leading dimension of the matrix Y.
   @param[in] nth Number of available threads/cores.
 */
void str_qr_par_init(int n, int b, double *A, int ldA, double *Y, int ldY, int nth);

/**
   @fn void str_qr_par(void *ptr)
   @brief Factor a structured matrix to R and Q in parallel
   
   @param[in,out] ptr Pointer to an oject of type struct thrd_arg.
 */
void *str_qr_par(void *ptr);
    
#endif
