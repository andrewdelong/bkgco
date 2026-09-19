// Non-template parts of the library. All templated code lives in
// GCoptimization.inl, which GCoptimization.h includes.
#include "GCoptimization.h"
#include <stdio.h>
#include <stdlib.h>

// Choose reasonably high-precision timer (sub-millisec resolution if possible).
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define VC_EXTRALEAN
#define NOMINMAX
#include <windows.h>

extern "C" gcoclock_t GCO_CLOCKS_PER_SEC = 0;

extern "C" gcoclock_t gcoclock() // TODO: not thread safe; separate begin/end so that end doesn't have to check for query frequency
{
	gcoclock_t result = 0;
	if (GCO_CLOCKS_PER_SEC == 0)
		QueryPerformanceFrequency((LARGE_INTEGER*)&GCO_CLOCKS_PER_SEC);
	QueryPerformanceCounter((LARGE_INTEGER*)&result);
	return result;
}

#else
extern "C" {
gcoclock_t GCO_CLOCKS_PER_SEC = CLOCKS_PER_SEC;
}
extern "C" gcoclock_t gcoclock() { return clock(); }
#endif

void GCException::Report() 
{
	printf("\n%s\n",message);
	exit(0);
}
