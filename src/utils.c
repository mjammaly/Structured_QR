#include<stdlib.h>
#include<sys/time.h>

#include "utils.h"

// Returns the current wall clock time in seconds.
double get_time(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + 1e-6*tv.tv_usec;
}

// Returns the total flops executed in an update-task.
double update_flops(int m, int n, int k)
{
    return ((1e-9 * 2 * n * (k * k)) + (1e-9 * 4 * k * n * (m - k)) + (1e-9* n * k));
}

// Returns the total flops executed in a factor-task.
double factor_flops(int m, int n)
{
    double sum = 0; // counter to keep the total flops
    
    int k = min(m-1,n);
    if (n == 1)
    {
	// A is 1x1
	if(k == 0)
	    return sum;
	else
	{
	    // reflector flops
	    sum += (3*m*1e-9);

	    // Construct Y
	    sum += (m*1e-9);
	}
    }
    else
    {
	// split block into two halvs
	int n1 = ifloor(n, 2);
	int n2 = n-n1;

	// factor left half
	sum += factor_flops(m, n1);

	// update right half
	sum += update_flops(m, n2, n1);

	// factor right half

	sum += factor_flops(m-n1, n2);

	// augment the two WY parts
	sum += update_flops(m-n1, n1, n2);	
    }

    return sum;
}
