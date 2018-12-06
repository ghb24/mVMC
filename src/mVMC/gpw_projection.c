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
  for(i=0;i<nGPWIdx;i++) {
    if (KernelFunc == 0) {
      eleGPWKern[i] = GPWKernelInPlace(eleNum, Nsite, GPWTrnCfg[i], GPWTrnSize[i], CutRad, Theta0, ThetaC);
    }
    else {
      eleGPWKern[i] = GPWKernelN(eleNum, Nsite, GPWTrnCfg[i], GPWTrnSize[i], KernelFunc);
    }
  }

  return;
}

void UpdateGPWKern(const int ri, const int rj, double *eleGPWKernNew, const double *eleGPWKernOld, const int *eleNum) {
  CalculateGPWKern(eleGPWKernNew, eleNum);
}
