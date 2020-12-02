// TODO include license and description

#ifndef _QGPS_INCLUDE_FILES
#define _QGPS_INCLUDE_FILES

#include "complex.h"

// Computes the inner sum for the exponential kernel with optimised basis
void ComputeInSumExpBasisOpt(double complex *inSum, const int *plaquetteAIdx, const int sizeA,
                             const int plaquetteSize, const double complex *distWeights,
                             const int *eleNum, const int tRSym, const int shift);

// Updates the inner sum for the exponential kernel with optimised basis
void UpdateInSumExpBasisOpt(double complex *inSumNew, const int *cfgOldReduced, const int *eleNum,
                            const int *plaquetteAIdx, const int sizeA, const int plaquetteSize,
                            const double complex *distWeights, int **plaqHash, const int ri,
                            const int rj, const int tRSym, const int shift);

// Computes the exponential kernel with optimised basis
double complex ComputeExpKernelBasisOpt(const int size, const int tRSym,
                                        const double complex *inSum, const int shift);

extern inline double complex LogQGPSVal(const double complex *workspace);
 
extern inline double complex QGPSExpansionargument(const double complex *eleGPWInSum);
 
extern inline double complex QGPSVal(const double complex *eleGPWInSum);
 
extern inline double complex LogQGPSRatio(const double complex *eleGPWInSumNew, const double complex *eleGPWInSumOld);
 
extern inline double complex QGPSRatio(const double complex *eleGPWInSumNew, const double complex *eleGPWInSumOld);

void CalculateQGPSInsum(double complex *eleGPWInSum, const int *eleNum);

void UpdateQGPSInSum(const int ri, const int rj, const int *cfgOldReduced,
                     double complex *eleGPWInSumNew, const double complex *eleGPWInSumOld,
                     const int *eleNum);

void calculateQGPSderivative(double complex *derivative, double complex *eleGPWInSum, int *eleNum);
#endif // QGPS_INCLUDE_FILES