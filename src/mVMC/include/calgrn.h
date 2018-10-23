#ifndef _CALGRN
#define _CALGRN
#include <complex.h>

void CalculateGreenFunc(const double w, const double complex ip, int *eleIdx, int *eleCfg,
                         int *eleNum, int *eleProjCnt, double *eleGPWKern);

void CalculateGreenFuncBF(const double w, const double ip, int *eleIdx, int *eleCfg,
                          int *eleNum, int *eleProjCnt, const int *eleProjBFCnt);
#endif
