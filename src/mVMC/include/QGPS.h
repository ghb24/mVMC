// TODO include license and description

#ifndef _QGPS_INCLUDE_FILES
#define _QGPS_INCLUDE_FILES

#include "complex.h"

double complex LogQGPSVal(const double complex *QGPSAmplitude);
 
double complex QGPSVal(const double complex *QGPSAmplitude);
 
double complex LogQGPSRatio(const double complex *QGPSAmplitudeNew, const double complex *QGPSAmplitudeOld);
 
double complex QGPSRatio(const double complex *QGPSAmplitudeNew, const double complex *QGPSAmplitudeOld);

void ComputeQGPSAmplitude(double complex *QGPSAmplitude, const double complex *eleGPWInSum);

void CalculateQGPSInsum(double complex *eleGPWInSum, const int *eleNum);

void UpdateQGPSInSum(const int ri, const int rj, const int *cfgOldReduced,
                     double complex *eleGPWInSumNew, const double complex *eleGPWInSumOld,
                     const int *eleNum);

void calculateQGPSderivative(double complex *derivative, const double complex *QGPSAmplitude,
                             const double complex *eleGPWInSum, const int *eleNum);
#endif // QGPS_INCLUDE_FILES