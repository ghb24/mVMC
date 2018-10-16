/*
TODO: Add License + Description*/
#include "global.h"
#include "gpw_projection.h"


//TODO: parallelise?
inline double complex LogGPWVal(const double *eleGPWKern) {
  int idx;
  double complex z=0.0+0.0*I;
  for(idx=0;idx<NGPWIdx;idx++) {
    z += GPWVar[idx] * (double)(eleGPWKern[idx]);
  }
  return z;
}

inline double complex LogGPWRatio(const double *eleGPWKernNew, const double *eleGPWKernOld) {
  int idx;
  double complex z=0.0+0.0*I;
  for(idx=0;idx<NGPWIdx;idx++) {
    z += GPWVar[idx] * (double)(eleGPWKernNew[idx]-eleGPWKernOld[idx]); 
  }
  return z;
}

inline double complex GPWRatio(const double *eleGPWKernNew, const double *eleGPWKernOld) {
  int idx;
  double complex z=0.0+0.0*I;
  for(idx=0;idx<NGPWIdx;idx++) {
    z += GPWVar[idx] * (double)(eleGPWKernNew[idx]-eleGPWKernOld[idx]); 
  }
  return cexp(z);
}

void CalculateGPWKern(double *eleGPWKern, const int *eleNum) {
  const int nGPWIdx=NGPWIdx;
  
  const int *n0=eleNum; //up-spin
  const int *n1=eleNum+Nsite; //down-spin
  int idx;
  
  // TODO: implement functionality
  
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
