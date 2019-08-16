/*
TODO: Add License + Description*/
#include "global.h"
#include "gpw_projection.h"
#include "gpw_kernel.h"
#include <omp.h>


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
  int outerThreadNum = omp_get_thread_num();

  #pragma omp parallel default(shared) private(i)
  {
    int totalThreadNum = outerThreadNum*omp_get_num_threads() + omp_get_thread_num();
    int *workspace = GPWKernWorkspace[totalThreadNum];
    #pragma omp for
    for(i=0;i<nGPWIdx;i++) {
      if (GPWKernelFunc[GPWTrnLat[i]] == 0) {
        eleGPWKern[i] = GPWKernel(eleNum, GPWSysPlaquetteIdx[GPWTrnLat[i]],
                                  Nsite, GPWTrnCfg[i],
                                  GPWTrnPlaquetteIdx[GPWTrnLat[i]],
                                  GPWTrnSize[GPWTrnLat[i]], Dim,
                                  GPWPower[GPWTrnLat[i]],
                                  GPWTheta0[GPWTrnLat[i]],
                                  GPWThetaC[GPWTrnLat[i]],
                                  GPWTRSym[GPWTrnLat[i]],
                                  GPWShift[GPWTrnLat[i]],
                                  GPWPlaquetteSizes[GPWTrnLat[i]],
                                  GPWDistList[GPWTrnLat[i]], workspace);
      }
      else {
        eleGPWKern[i] = GPWKernelN(eleNum, GPWSysPlaquetteIdx[GPWTrnLat[i]],
                                   Nsite, GPWTrnCfg[i],
                                   GPWTrnPlaquetteIdx[GPWTrnLat[i]],
                                   GPWTrnSize[GPWTrnLat[i]], Dim,
                                   GPWKernelFunc[GPWTrnLat[i]],
                                   GPWTRSym[GPWTrnLat[i]],
                                   GPWShift[GPWTrnLat[i]], workspace);
      }
    }
  }
  return;
}

void UpdateGPWKern(const int ri, const int rj, double *eleGPWKernNew, const double *eleGPWKernOld, const int *eleNum) {
  CalculateGPWKern(eleGPWKernNew, eleNum);
}
