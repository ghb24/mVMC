/*
TODO: Add License + Description*/
#include "global.h"
#include "gpw_projection.h"
#include "gpw_kernel.h"
#include "gpw_exp_kernel.h"
#include <omp.h>


inline double complex LogGPWVal(const double complex *eleGPWKern, const double complex *eleGPWInSum) {
  int idx;
  double complex z=0.0+0.0*I;
  if (QGPS) {
    return LogQGPSVal(eleGPWInSum);
  }
  else {
    if (GPWExpansionOrder == -1) {
      #pragma omp parallel for default(shared) private(idx) reduction(+:z)
      for(idx=0;idx<NGPWIdx;idx++) {
        z += GPWVar[idx] * eleGPWKern[idx];
      }

      return z;
    }
    else {
      return clog(GPWVal(eleGPWKern, eleGPWInSum));
    }
  }
}

inline double complex GPWExpansionargument(const double complex *eleGPWKern) {
  int idx;
  double complex expansionargument=0.0+0.0*I;

  #pragma omp parallel for default(shared) private(idx) reduction(+:expansionargument)
  for(idx=0;idx<NGPWIdx;idx++) {
    expansionargument += GPWVar[idx] * eleGPWKern[idx];
  }

  return expansionargument;
}

inline double complex GPWVal(const double complex *eleGPWKern, const double complex *eleGPWInSum) {
  int i;
  double complex expansionargument;
  double complex z=0.0+0.0*I;
  int factorial = 1;

  if (QGPS) {
    return QGPSVal(eleGPWInSum);
  }
  else {
    expansionargument = GPWExpansionargument(eleGPWKern);

    if (GPWExpansionOrder == -1) {
      return cexp(expansionargument);
    }

    else {
      for(i = 0; i <= GPWExpansionOrder; i++) {
        z += cpow(expansionargument, i)/factorial;
        factorial *= (i+1);
      }
      return z;
    }
  }
}

inline double complex LogGPWRatio(const double complex *eleGPWKernNew, const double complex *eleGPWKernOld,
                                  const double complex *eleGPWInSumNew, const double complex *eleGPWInSumOld) {
  int idx;
  double complex z=0.0+0.0*I;

  if (QGPS) {
    return LogQGPSRatio(eleGPWInSumNew, eleGPWInSumOld);
  }
  else {
    if (GPWExpansionOrder == -1) {
      #pragma omp parallel for default(shared) private(idx) reduction(+:z)
      for(idx=0;idx<NGPWIdx;idx++) {
        z += GPWVar[idx] * (eleGPWKernNew[idx]-eleGPWKernOld[idx]);
      }
      return z;
    }

    else {
      return (LogGPWVal(eleGPWKernNew, eleGPWInSumNew)-LogGPWVal(eleGPWKernOld, eleGPWInSumOld));
    }
  }
}

inline double complex GPWRatio(const double complex *eleGPWKernNew, const double complex *eleGPWKernOld,
                               const double complex *eleGPWInSumNew, const double complex *eleGPWInSumOld) {
  if (QGPS) {
    return QGPSRatio(eleGPWInSumNew, eleGPWInSumOld);
  }
  else {
    if (GPWExpansionOrder == -1) {
      return cexp(LogGPWRatio(eleGPWKernNew, eleGPWKernOld, eleGPWInSumNew, eleGPWInSumOld));
    }

    else {
      return (GPWVal(eleGPWKernNew, eleGPWInSumNew)/GPWVal(eleGPWKernOld, eleGPWInSumOld));
    }
  }
}

void CalculateGPWKern(double complex *eleGPWKern, double complex *eleGPWInSum, const int *eleNum) {
  const int nGPWIdx=NGPWIdx;
  int i;


  if (QGPS) {
    return CalculateQGPSInsum(eleGPWInSum, eleNum);
  }
  else {
    #pragma omp parallel for default(shared) private(i)
    for(i=0;i<nGPWIdx;i++) {
      int latId = GPWTrnLat[i];
      int offset = 0;
      int offsetDistWeights = 0;
      int j, inSumSize;

      for (j = 0; j < i; j++) {
        inSumSize = 1;
        if (abs(GPWShift[GPWTrnLat[j]]) & 1) {
          inSumSize *= Nsite;
        }
        if ((abs(GPWShift[GPWTrnLat[j]]) & 2) >> 1) {
          inSumSize *= GPWTrnSize[GPWTrnLat[j]];
        }
        if (GPWTRSym[GPWTrnLat[j]]) {
          inSumSize *= 2;
        }

        offset += inSumSize;
        offsetDistWeights += 4*GPWPlaquetteSizes[GPWTrnLat[j]];
      }

      if (GPWKernelFunc[latId] == 1) {
        eleGPWKern[i] = GPWKernel1(eleNum, Nsite, GPWTrnCfg[i], GPWTrnSize[latId],
                                  GPWTRSym[latId], GPWShift[latId], 0, 0);
      }
      else if (GPWKernelFunc[latId] < 0) {
        ComputeInSumExp(eleGPWInSum+offset, GPWSysPlaquetteIdx[latId],
                        eleNum, Nsite, GPWTrnPlaquetteIdx[latId],
                        GPWTrnCfg[i], GPWTrnSize[latId], GPWPlaquetteSizes[latId],
                        GPWDistWeights, GPWDistWeightIdx[i],
                        GPWShift[latId], 0, 0, GPWTRSym[latId]);
        if (GPWKernelFunc[latId] == -1) {
          eleGPWKern[i] = ComputeExpKernel(eleNum, Nsite, GPWTrnCfg[i], GPWTrnSize[latId],
                                          GPWTRSym[latId], GPWShift[latId], 0, 0,
                                          eleGPWInSum+offset, 0);
        }
        else if (GPWKernelFunc[latId] == -2) {
          eleGPWKern[i] = ComputeExpKernel(eleNum, Nsite, GPWTrnCfg[i], GPWTrnSize[latId],
                                          GPWTRSym[latId], GPWShift[latId], 0, 0,
                                          eleGPWInSum+offset, 1);
        }
        else {
          printf("Error, bad kernel type\n");
        }
      }

      else if (GPWKernelFunc[latId] == 0) {
        ComputeInSum(eleGPWInSum+offset, GPWSysPlaquetteIdx[latId],
                    eleNum, Nsite, GPWTrnPlaquetteIdx[latId],
                    GPWTrnCfg[i], GPWTrnSize[latId], GPWPlaquetteSizes[latId],
                    GPWDistList[latId], GPWShift[latId], 0, 0,
                    GPWDistWeightPower[latId], GPWTRSym[latId]);
        eleGPWKern[i] = ComputeKernel(eleNum, Nsite, GPWTrnCfg[i], GPWTrnSize[latId],
                                      GPWPower[latId], GPWThetaVar[latId],
                                      GPWNorm[latId], GPWTRSym[latId],
                                      GPWShift[latId], 0, 0, eleGPWInSum+offset);
      }

      else {
        eleGPWKern[i] = ComputeKernelN(eleNum, GPWSysPlaquetteIdx[latId], Nsite,
                                      GPWTrnCfg[i], GPWTrnPlaquetteIdx[latId], GPWTrnSize[latId],
                                      GPWKernelFunc[latId], GPWTRSym[latId],
                                      GPWShift[latId], 0, 0, eleGPWInSum+offset);
      }
    }
  }
  return;
}

void UpdateGPWKern(const int ri, const int rj, const int *cfgOldReduced,
                   double complex *eleGPWKernNew, double complex *eleGPWInSumNew,
                   const double complex *eleGPWKernOld, const double complex *eleGPWInSumOld,
                   const int *eleNum) {
  const int nGPWIdx=NGPWIdx;
  int i;
  if (QGPS) {
    return UpdateQGPSInSum(ri, rj, cfgOldReduced, eleGPWInSumNew, eleGPWInSumOld, eleNum);
  }
  else {
    memcpy(eleGPWInSumNew, eleGPWInSumOld, sizeof(double complex)*GPWInSumSize);

  // #pragma omp parallel for default(shared) private(i)
    for(i=0;i<nGPWIdx;i++) {
      int j, inSumSize;
      int latId = GPWTrnLat[i];
      int offset = 0;
      int offsetDistWeights = 0;

      for (j = 0; j < i; j++) {
        inSumSize = 1;
        if (abs(GPWShift[GPWTrnLat[j]]) & 1) {
          inSumSize *= Nsite;
        }
        if ((abs(GPWShift[GPWTrnLat[j]]) & 2) >> 1) {
          inSumSize *= GPWTrnSize[GPWTrnLat[j]];
        }
        if (GPWTRSym[GPWTrnLat[j]]) {
          inSumSize *= 2;
        }

        offset += inSumSize;
        offsetDistWeights += 4*GPWPlaquetteSizes[GPWTrnLat[j]];
      }

      if (GPWKernelFunc[latId] == 1) {
        eleGPWKernNew[i] = GPWKernel1(eleNum, Nsite, GPWTrnCfg[i], GPWTrnSize[latId],
                                      GPWTRSym[latId], GPWShift[latId], 0, 0);
      }

      else if (GPWKernelFunc[latId] < 0) {
        UpdateInSumExp(eleGPWInSumNew+offset, cfgOldReduced, eleNum,
                      GPWSysPlaquetteIdx[latId], Nsite, GPWTrnPlaquetteIdx[latId],
                      GPWTrnCfg[i], GPWTrnSize[latId], GPWPlaquetteSizes[latId],
                      GPWDistWeights, GPWDistWeightIdx[i], GPWShift[latId],
                      0, 0, GPWSysPlaqHash[latId], GPWSysPlaqHashSz[latId], ri, rj,
                      GPWTRSym[latId]);
        if (GPWKernelFunc[latId] == -1) {
          eleGPWKernNew[i] = ComputeExpKernel(eleNum, Nsite, GPWTrnCfg[i], GPWTrnSize[latId],
                                              GPWTRSym[latId], GPWShift[latId],
                                              0, 0, eleGPWInSumNew+offset, 0);
        }
        else if (GPWKernelFunc[latId] == -2) {
          eleGPWKernNew[i] = ComputeExpKernel(eleNum, Nsite, GPWTrnCfg[i], GPWTrnSize[latId],
                                              GPWTRSym[latId], GPWShift[latId],
                                              0, 0, eleGPWInSumNew+offset, 1);
        }
        else {
          printf("Error, bad kernel type\n");
        }
      }
      else if (GPWKernelFunc[latId] == 0) {
        UpdateInSum(eleGPWInSumNew+offset, cfgOldReduced, eleNum,
                    GPWSysPlaquetteIdx[latId], Nsite, GPWTrnPlaquetteIdx[latId],
                    GPWTrnCfg[i], GPWTrnSize[latId], GPWPlaquetteSizes[latId],
                    GPWDistList[latId], GPWShift[latId], 0, 0, GPWDistWeightPower[latId],
                    GPWSysPlaqHash[latId], GPWSysPlaqHashSz[latId], ri, rj, GPWTRSym[latId]);
        eleGPWKernNew[i] = ComputeKernel(eleNum, Nsite, GPWTrnCfg[i], GPWTrnSize[latId],
                                        GPWPower[latId], GPWThetaVar[latId],
                                        GPWNorm[latId], GPWTRSym[latId],
                                        GPWShift[latId], 0, 0, eleGPWInSumNew+offset);
      }
      else {
        eleGPWKernNew[i] = ComputeKernelN(eleNum, GPWSysPlaquetteIdx[latId], Nsite,
                                          GPWTrnCfg[i], GPWTrnPlaquetteIdx[latId], GPWTrnSize[latId],
                                          GPWKernelFunc[latId], GPWTRSym[latId],
                                          GPWShift[latId], 0, 0, eleGPWInSumNew+offset);
      }
    }
  }
  return;
}
