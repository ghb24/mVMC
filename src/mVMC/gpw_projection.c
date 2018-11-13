/*
TODO: Add License + Description*/
#include "global.h"
#include "gpw_projection.h"
#include "gpw_kernel.h"


inline double complex LogGPWVal(const double *eleGPWKern) {
  int idx;
  double complex z=0.0+0.0*I;
  for(idx=0;idx<NGPWIdx;idx++) {
    z += GPWVar[idx] * eleGPWKern[idx];
  }
  return z;
}

inline double complex LogGPWRatio(const double *eleGPWKernNew, const double *eleGPWKernOld) {
  int idx;
  double complex z=0.0+0.0*I;
  for(idx=0;idx<NGPWIdx;idx++) {
    z += GPWVar[idx] * (eleGPWKernNew[idx]-eleGPWKernOld[idx]);
  }
  return z;
}

inline double complex GPWRatio(const double *eleGPWKernNew, const double *eleGPWKernOld) {
  int idx;
  double complex z=0.0+0.0*I;
  for(idx=0;idx<NGPWIdx;idx++) {
    z += GPWVar[idx] * (eleGPWKernNew[idx]-eleGPWKernOld[idx]);
  }
  return cexp(z);
}

void CalculateGPWKern(double *eleGPWKern, const int *eleNum) {
  const int nGPWIdx=NGPWIdx;
  int i;

  #pragma omp parallel for default(shared) private(i)
  for(i=0; i<nGPWIdx;i++) {
    eleGPWKern[i] = (double)GPWKernel1(eleNum, Nsite, GPWTrnCfg[i], GPWTrnSize[i]);
  }

  return;
}


// TODO more efficient way?
void UpdateGPWKern(const int ri, const int rj, const int s,
									 double *eleGPWKernNew, const double *eleGPWKernOld,
									 const int *eleNum) {
  CalculateGPWKern(eleGPWKernNew, eleNum);
  return;
}

// TODO more efficient way?
void UpdateGPWKern_fsz(const int ri, const int rj, const int s, const int t,
											 double *eleGPWKernNew, const double *eleGPWKernOld,
											 const int *eleNum) {
  CalculateGPWKern(eleGPWKernNew, eleNum);
  return;
}
