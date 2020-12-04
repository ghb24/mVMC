#ifndef _GPW_PROJECTION_
#define _GPW_PROJECTION_

double complex LogGPWVal(const double complex *eleGPWKern, const double complex *eleGPWInSum);
double complex GPWExpansionargument(const double complex *eleGPWKern);
double complex GPWVal(const double complex *eleGPWKern, const double complex *eleGPWInSum);
double complex LogGPWRatio(const double complex *eleGPWKernNew, const double complex *eleGPWKernOld,
                           const double complex *eleGPWInSumNew, const double complex *eleGPWInSumOld);
double complex GPWRatio(const double complex *eleGPWKernNew, const double complex *eleGPWKernOld,
                        const double complex *eleGPWInSumNew, const double complex *eleGPWInSumOld);
void CalculateGPWKern(double complex *eleGPWKern, double complex *eleGPWInSum, const int *eleNum);

void UpdateGPWKern(const int ri, const int rj, const int *cfgOldReduced,
                   double complex *eleGPWKernNew, double complex *eleGPWInSumNew,
                   const double complex *eleGPWKernOld, const double complex *eleGPWInSumOld,
                   const int *eleNum);
#endif
