/*
TODO: Add License + Description*/
#include "global.h"
#include "gpw_projection.h"


//TODO: Implement functionality
inline double complex LogGPWVal(const double *eleGPWKern) {
  return 0.0;
}

inline double complex LogGPWRatio(const double *eleGPWKernNew, const double *eleGPWKernOld) {
  return 0.0;
}

inline double complex GPWRatio(const int *projCntNew, const int *projCntOld) {
  return 0.0;
}

void CalculateGPWKern(double *eleGPWKern, const int *eleNum) {
  return;
}

void UpdateGPWKern(const int ri, const int rj, const int s,
									 const double *eleGPWKernNew, const double *eleGPWKernOld,
									 const int *eleNum) {
  return;
}

void UpdateGPWKern_fsz(const int ri, const int rj, const int s, const int t,
											 const double *eleGPWKernNew, const double *eleGPWKernOld,
											 const int *eleNum) {
  return;
}
