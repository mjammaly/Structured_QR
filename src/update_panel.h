#ifndef UPDATE_PANEL_H
#define UPDATE_PANEL_H


#include"spin-barrier.h"

/**
   @file update_panel.h
   @brief Contains functions to perform panel update from left.
*/


/**
   @fn void update_panel_seq(int m, int n, int k, double *A, int ldA, double *B, int ldB, double *C, int ldC, double *Z, int ldZ)
   @brief Update a panel from the left using Q sequentially.

   computes : A <- (I - B * C') * A, where B and C are trapezoidal matrices.
   Z is an intermediate matrix

   @param[in] m Numer of rows in the input matrix.
   @param[in] n Numer of columns  in the input matrix.
   @param[in] k The size of the triangular part of matrices B and C.
   @param[in,out] A Pointer to the input matrix. At the exit will be replaced by the updated matrix A.
   @param[in] ldA Leading dimension of the input matrix.
   @param[in] B  Pointer to a trapezoidal matrix B.
   @param[in] ldB Leading dimension of the matrix B.
   @param[in] C Pointer to a trapezoidal matrix C.
   @param[in] ldC Leading dimension of the matrix C.
   @param[in] Z Pointer to intermdiate matrix Z.
   @param[in] ldZ Leading dimension of the matrix Z.
*/
void update_panel_seq(int m, int n, int k, double *A, int ldA, double *B, int ldB, double *C, int ldC, double *Z, int ldZ);


/**
   @fn void update_panel_par(int m, int n, int k, double *A, int ldA, double *B, int ldB, double *C, int ldC, double *Z, int ldZ, int nth, int rank, pcp_barrier_t *bar)
   @brief Update a panel from the left using Q in parallel.

   computes : A <- (I - B * C') * A, where B and C are trapezoidal matrices.
   Z is an intermediate matrix

   @param[in] m Numer of rows in the input matrix.
   @param[in] n Numer of columns  in the input matrix.
   @param[in] k The size of the triangular part of matrices B and C.
   @param[in,out] A Pointer to the input matrix. At the exit will be replaced by the updated matrix A.
   @param[in] ldA Leading dimension of the input matrix.
   @param[in] B  Pointer to a trapezoidal matrix B.
   @param[in] ldB Leading dimension of the matrix B.
   @param[in] C Pointer to a trapezoidal matrix C.
   @param[in] ldC Leading dimension of the matrix C.
   @param[in] Z Pointer to intermdiate matrix Z.
   @param[in] ldZ Leading dimension of the matrix Z.
   @param[in] nth Number of available threads/cores.
   @param[in] rank The rank/ID of the calling thread.
   @param[in] bar Pointer to a local barrier.
*/
void update_panel_par(int m, int n, int k, double *A, int ldA, double *B, int ldB, double *C, int ldC, double *Z, int ldZ, int nth, int rank, pcp_barrier_t *bar);
    
#endif
