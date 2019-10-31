#ifndef _GPW_PROJECTION_
#define _GPW_PROJECTION_

extern inline double complex LogGPWVal(const double *eleGPWKern);
extern inline double complex GPWVal(const double *eleGPWKern);
extern inline double complex LogGPWRatio(const double *eleGPWKernNew, const double *eleGPWKernOld);
extern inline double complex GPWRatio(const double *eleGPWKernNew, const double *eleGPWKernOld);
void CalculateGPWKern(double *eleGPWKern, int *eleGPWDelta, double *eleGPWInSum, const int *eleNum);

void UpdateGPWKern(const int ri, const int rj, double *eleGPWKernNew,
                   int *eleGPWDeltaNew, double *eleGPWInSumNew,
                   const double *eleGPWKernOld, const int *eleGPWDeltaOld,
                   const double *eleGPWInSumOld, const int *eleNum);
#endif
