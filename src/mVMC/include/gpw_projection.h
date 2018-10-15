#ifndef _GPW_PROJECTION_
#define _GPW_PROJECTION_

extern inline double complex LogGPWVal(const double *eleGPWKern);
extern inline double complex LogGPWRatio(const double *eleGPWKernNew, const double *eleGPWKernOld);
extern inline double complex GPWRatio(const int *projCntNew, const int *projCntOld);
void CalculateGPWKern(double *eleGPWKern, const int *eleNum);
void UpdateGPWKern(const int ri, const int rj, const int s,
									 const double *eleGPWKernNew, const double *eleGPWKernOld,
									 const int *eleNum);
void UpdateGPWKern_fsz(const int ri, const int rj, const int s, const int t,
											 const double *eleGPWKernNew, const double *eleGPWKernOld,
											 const int *eleNum);

#endif
