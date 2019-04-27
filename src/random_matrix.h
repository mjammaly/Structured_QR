#ifndef STR_QR_RANDOM_MATRIX_H
#define STR_QR_RANDOM_MATRIX_H


/**
   @file random_matrix.h
   @brief Contains functions that generates random matrices with specific shapes.
 */

/**
   @fn void generate_dense_rectangle_matrix(int m, int n, double *A, int ldA)
   @brief Generate random dense rectangular matrix which is diagonal domenant.
 
   @param[in] m Number of rows in the matrix.
   @param[in] n Number of columns in the matrix.
   @param[in,out] A Pointer to the matrix buffer. At exit will contain the output matrix.
   @param[in] ldA Leading dimension of the matrix.
*/
void generate_dense_rectangle_matrix(int m, int n, double *A, int ldA);

/**
   @fn void generate_dense_block_interleaved_upper_triangler_matrix(int n, int b, double *A, int ldA)
   @brief Generate random blocked uppaer triangular matrix with overlapping blocks on the diagonal.
   
   @param[in] n The matrix order.
   @param[in] b The size of the ovverlapped blocks.
   @param[in,out] A Pointer to the matrix buffer. At exit will contain the output matrix.
   @param[in] ldA Leading dimension of the matrix.
*/
void generate_dense_block_interleaved_upper_triangler_matrix(int n, int b, double *A, int ldA);

#endif 
