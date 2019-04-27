#ifndef STR_QR_TEST_REDUCTION_H
#define STR_QR_TEST_REDUCTION_H

/**
   @file test_reduction.h
   @brief Contains functions to testing and validating results.
*/


/**
   @fn void test_panel_reduction(int m, int n, double *A, int ldA, double *R, int ldR, double *Y, int ldY, double *Z, int ldZ)
   @brief Validate the factorization of a dense rectangluer matrix to Q and R.

   @param[in] m Number of rows in the input matrix.
   @param[in] n Number of columns in the input matrix.
   @param[in] A Pointer to the input matrix.
   @param[in] ldA Leading dimension of the input matrix.
   @param[in] R Pointer to the output of factoring the input matrix, its upper part contains R.
   @param[in] ldR Leading dimension of the output matrix
   @param[in] Y Pointer to the temporary matrix Y.
   @param[in] ldY Leading dimension of matrix Y.
   @param[in] Z Pointer to the temporary matrix Z.
   @param[in] ldZ Leading dimension of the matrix Z.
*/
void test_panel_reduction(int m, int n, double *A, int ldA, double *R, int ldR, double *Y, int ldY, double *Z, int ldZ);


/**
   @fn void test_matrix_reduction(int n, int b, double *A, int ldA, double *R, int ldR, double *Y, int ldY, double *Z, int ldZ)
   @brief Validate the factorization of a block triangular matrix with overlapping blocks on the diagonal to Q and R.

   @param[in] n The order of the input matrix.
   @param[in] b The size of the overlapped blocks.
   @param[in] A Pointer to the input matrix.
   @param[in] ldA Leading dimension of the input matrix.
   @param[in] R Pointer to the output of factoring the input matrix, its upper part contains R.
   @param[in] ldR Leading dimension of the output matrix
   @param[in] Y Pointer to the temporary matrix Y.
   @param[in] ldY Leading dimension of matrix Y.
   @param[in] Z Pointer to the temporary matrix Z.
   @param[in] ldZ Leading dimension of the matrix Z.
*/
void test_matrix_reduction(int n, int b, double *A, int ldA, double *R, int ldR, double *Y, int ldY, double *Z, int ldZ);

#endif
