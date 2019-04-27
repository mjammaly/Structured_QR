#ifndef PRINT_MATRIX_H
#define PRINT_MATRIX_H

/**
   @file print_matrix.h
   @brief Contains matrix printing functions.
*/

/**
   @fn void print_matrix(int m, int n, double *A, int ldA)
   @brief Print a matrix on stdout in a readable way.

   @param[in] m Number of rows in the input matrix.
   @param[in] n Number of columns in the input matrix.
   @param[in] A Pointer to the input matrix.
   @param[in] ldA Leading dimension of the input matrix.
 */
void print_matrix(int m, int n, double *A, int ldA);

#endif
