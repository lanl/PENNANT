#include <sys/time.h>
#include <iostream>

double get_time(){
	struct timeval sbegin;
	double tbegin;
        gettimeofday(&sbegin, NULL);
        tbegin = sbegin.tv_sec + sbegin.tv_usec * 1.e-6;
	return tbegin;
}
