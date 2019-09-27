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

void CalculateGPWKern(double *eleGPWKern, int *eleGPWDelta, double *eleGPWInSum, const int *eleNum) {
  const int nGPWIdx=NGPWIdx;
  int i;
  int outerThreadNum = omp_get_thread_num();

  #pragma omp parallel for default(shared) private(i)
  for(i=0;i<nGPWIdx;i++) {
    int latId = GPWTrnLat[i];
    int offset = 0;
    int j;

    for (j = 0; j < i; j++) {
      offset += Nsite*GPWTrnSize[GPWTrnLat[j]];
    }



    if (GPWKernelFunc[latId] == 1) {
      eleGPWKern[i] = GPWKernel1(eleNum, Nsite, GPWTrnCfg[i], GPWTrnSize[latId],
                                 GPWTRSym[latId], GPWShift[latId], 0, 0);
    }

    else {
      CalculatePairDelta(eleGPWDelta+offset, eleNum, Nsite, GPWTrnCfg[i],
                         GPWTrnSize[latId]);

      if (GPWTRSym[latId]) {
        CalculatePairDeltaFlipped(eleGPWDelta+(GPWTrnCfgSz/2)*Nsite+offset, eleNum,
                                  Nsite, GPWTrnCfg[i], GPWTrnSize[latId]);
      }


      if (GPWKernelFunc[latId] == 0) {
        ComputeInSum(eleGPWInSum+offset, eleGPWDelta+offset, GPWSysPlaquetteIdx[latId],
                     Nsite, GPWTrnPlaquetteIdx[latId],
                     GPWTrnSize[latId], GPWPlaquetteSizes[latId],
                     GPWDistList[latId]);
        if (GPWTRSym[latId]) {
          ComputeInSum(eleGPWInSum+(GPWTrnCfgSz/2)*Nsite+offset,
                       eleGPWDelta+(GPWTrnCfgSz/2)*Nsite+offset,
                       GPWSysPlaquetteIdx[latId], Nsite,
                       GPWTrnPlaquetteIdx[latId], GPWTrnSize[latId],
                       GPWPlaquetteSizes[latId], GPWDistList[latId]);
        }
        eleGPWKern[i] = ComputeKernel(Nsite, GPWTrnSize[latId],
                                      GPWPower[latId], GPWTheta0[latId],
                                      GPWNorm[latId], GPWTRSym[latId],
                                      GPWShift[latId], 0, 0, eleGPWDelta+offset,
                                      eleGPWDelta+(GPWTrnCfgSz/2)*Nsite+offset,
                                      eleGPWInSum+offset,
                                      eleGPWInSum+(GPWTrnCfgSz/2)*Nsite+offset);
      }

      else {
        eleGPWKern[i] = ComputeKernelN(eleNum, GPWSysPlaquetteIdx[latId], Nsite,
                                       GPWTrnCfg[i], GPWTrnPlaquetteIdx[latId], GPWTrnSize[latId],
                                       GPWKernelFunc[latId], GPWTRSym[latId],
                                       GPWShift[latId], 0, 0, eleGPWDelta+offset,
                                       eleGPWDelta+(GPWTrnCfgSz/2)*Nsite+offset);
      }
    }
  }
  return;
}

void UpdateGPWKern(const int ri, const int rj, double *eleGPWKernNew,
                   int *eleGPWDeltaNew, double *eleGPWInSumNew,
                   const double *eleGPWKernOld, const int *eleGPWDeltaOld,
                   const double *eleGPWInSumOld, const int *eleNum) {
  const int nGPWIdx=NGPWIdx;
  int i;
  int outerThreadNum = omp_get_thread_num();

  #pragma omp parallel for default(shared) private(i)
  for(i=0;i<nGPWIdx;i++) {
    int latId = GPWTrnLat[i];
    int offset = 0;
    int j;

    for (j = 0; j < i; j++) {
      offset += Nsite*GPWTrnSize[GPWTrnLat[j]];
    }

    if (GPWKernelFunc[latId] == 1) {
      eleGPWKernNew[i] = GPWKernel1(eleNum, Nsite, GPWTrnCfg[i], GPWTrnSize[latId],
                                    GPWTRSym[latId], GPWShift[latId], 0, 0);
    }

    else {
      UpdateDelta(eleGPWDeltaNew+offset, eleGPWDeltaOld+offset, eleNum, Nsite, GPWTrnCfg[i],
                  GPWTrnSize[latId], ri, rj);

      if (GPWTRSym[latId]) {
        UpdateDeltaFlipped(eleGPWDeltaNew+(GPWTrnCfgSz/2)*Nsite+offset,
                           eleGPWDeltaOld+(GPWTrnCfgSz/2)*Nsite+offset, eleNum,
                           Nsite, GPWTrnCfg[i], GPWTrnSize[latId], ri, rj);
      }


      if (GPWKernelFunc[latId] == 0) {
        UpdateInSum(eleGPWInSumNew+offset, eleGPWInSumOld+offset, eleGPWDeltaNew+offset,
                    eleGPWDeltaOld+offset, GPWSysPlaquetteIdx[latId],
                    Nsite, GPWTrnPlaquetteIdx[latId],
                    GPWTrnSize[latId], GPWPlaquetteSizes[latId],
                    GPWDistList[latId], GPWSysPlaqHash[latId],
                    GPWSysPlaqHashSz[latId], ri, rj);


        if (GPWTRSym[latId]) {
          UpdateInSum(eleGPWInSumNew+(GPWTrnCfgSz/2)*Nsite+offset,
                      eleGPWInSumOld+(GPWTrnCfgSz/2)*Nsite+offset,
                      eleGPWDeltaNew+(GPWTrnCfgSz/2)*Nsite+offset,
                      eleGPWDeltaOld+(GPWTrnCfgSz/2)*Nsite+offset,
                      GPWSysPlaquetteIdx[latId], Nsite,
                      GPWTrnPlaquetteIdx[latId], GPWTrnSize[latId],
                      GPWPlaquetteSizes[latId], GPWDistList[latId],
                      GPWSysPlaqHash[latId], GPWSysPlaqHashSz[latId],
                      ri, rj);
        }
        eleGPWKernNew[i] = ComputeKernel(Nsite, GPWTrnSize[latId],
                                         GPWPower[latId], GPWTheta0[latId],
                                         GPWNorm[latId], GPWTRSym[latId],
                                         GPWShift[latId], 0, 0, eleGPWDeltaNew+offset,
                                         eleGPWDeltaNew+(GPWTrnCfgSz/2)*Nsite+offset,
                                         eleGPWInSumNew+offset,
                                         eleGPWInSumNew+(GPWTrnCfgSz/2)*Nsite+offset);
      }

      else {
        eleGPWKernNew[i] = ComputeKernelN(eleNum, GPWSysPlaquetteIdx[latId], Nsite,
                                          GPWTrnCfg[i], GPWTrnPlaquetteIdx[latId], GPWTrnSize[latId],
                                          GPWKernelFunc[latId], GPWTRSym[latId],
                                          GPWShift[latId], 0, 0, eleGPWDeltaNew+offset,
                                          eleGPWDeltaNew+(GPWTrnCfgSz/2)*Nsite+offset);
      }
    }
  }
  return;
}
