#ifndef UTILS_H
#define UTILS_H


/**
@file utils.h
@brief Contains some utility functions.
 */

/**
   @fn int min(int a, int b)
   @brief Minimum of two integer numbers.

   @param[in] a First integer number.
   @param[in] b Second integer number.
   @return The minimum of the two integer numbers.
*/
static inline int min(int a, int b)
{
    return a < b ? a : b;
}

/**
   @fn int max(int a, int b)
   @brief Maximum of two integer numbers.

   @param[in] a First integer number.
   @param[in] b Second integer number.
   @return The maximum of the two integer numbers.
*/
static inline int max(int a, int b)
{
    return a > b ? a : b;
}

/**
   @fn int iceil(int a, int b)
   @brief Integer ceil.

   @param[in] a Dividend.
   @param[in] b Divisor.
   @return The nearest higher integer for dividing a by b.
*/
static inline int iceil(int a, int b)
{
    return (a + b - 1) / b;
}

/**
   @fn int ifloor(int a, int b)
   @brief Integer floor.

   @param[in] a Dividend.
   @param[in] b Divisor.
   @return The nearest lower integer for dividing a by b.
*/
static inline int ifloor(int a, int b)
{
    return a / b;
}

/**
   @fn double get_time(void)
   @brief Get time in micro seconds.

   @return The wall time in micro seconds.
*/
double get_time(void);

/**
   @fn double factor_flops(int m, int n)
   @brief Compute number of flops in factor kernel.

   @param[in] m Number of rows in the matrix.
   @param[in] n Number of columns in the matrix.
   @return Number of floating point operations performed by the factor kernel for the givin matrix dimensions.
*/
double factor_flops(int m, int n);

/**
   @fn double update_flops(int m, int n, int k)
   @brief Compute number of flops in update kernel.

   @param[in] m Number of rows in the matrix.
   @param[in] n Number of columns in the matrix.
   @param[in] k blocking size.
   @return Number of floating point operations performed by the update kernel for the givin matrix dimensions.
*/
double update_flops(int m, int n, int k);
    
#endif
