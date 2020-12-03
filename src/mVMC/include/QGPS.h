// TODO include license and description

#ifndef _QGPS_INCLUDE_FILES
#define _QGPS_INCLUDE_FILES

#include "complex.h"

double complex LogQGPSVal(const double complex *workspace);
 
double complex QGPSVal(const double complex *eleGPWInSum);
 
double complex LogQGPSRatio(const double complex *eleGPWInSumNew, const double complex *eleGPWInSumOld);
 
double complex QGPSRatio(const double complex *eleGPWInSumNew, const double complex *eleGPWInSumOld);

void CalculateQGPSInsum(double complex *eleGPWInSum, const int *eleNum);

void UpdateQGPSInSum(const int ri, const int rj, const int *cfgOldReduced,
                     double complex *eleGPWInSumNew, const double complex *eleGPWInSumOld,
                     const int *eleNum);

void calculateQGPSderivative(double complex *derivative, double complex *eleGPWInSum, int *eleNum);
#endif // QGPS_INCLUDE_FILES