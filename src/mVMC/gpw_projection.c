/*
TODO: Add License + Description*/
#include "global.h"
#include "gpw_projection.h"
#include "gpw_kernel.h"
#include "gpw_exp_kernel.h"
#include <omp.h>


inline double complex LogGPWVal(const double *eleGPWKern) {
  int idx;
  double complex z=0.0+0.0*I;

  #pragma omp parallel for default(shared) private(idx) reduction(+:z)
  for(idx=0;idx<NGPWIdx;idx++) {
    z += GPWVar[idx] * eleGPWKern[idx];
  }

  if (!GPWLinModFlag) {
    return z;
  }
  else {
    return clog(z + 1.0);
  }
}

inline double complex GPWVal(const double *eleGPWKern) {
  int idx;
  double complex z=0.0+0.0*I;

  #pragma omp parallel for default(shared) private(idx) reduction(+:z)
  for(idx=0;idx<NGPWIdx;idx++) {
    z += GPWVar[idx] * eleGPWKern[idx];
  }


  if (!GPWLinModFlag) {
    return cexp(z);
  }
  else {
    return (z + 1.0);
  }
}

inline double complex LogGPWRatio(const double *eleGPWKernNew, const double *eleGPWKernOld) {
  int idx;
  double complex z=0.0+0.0*I;

  if (!GPWLinModFlag) {
    #pragma omp parallel for default(shared) private(idx) reduction(+:z)
    for(idx=0;idx<NGPWIdx;idx++) {
      z += GPWVar[idx] * (eleGPWKernNew[idx]-eleGPWKernOld[idx]);
    }
    return z;
  }

  else {
    return (LogGPWVal(eleGPWKernNew)-LogGPWVal(eleGPWKernOld));
  }

}

inline double complex GPWRatio(const double *eleGPWKernNew, const double *eleGPWKernOld) {
  if (!GPWLinModFlag) {
    return cexp(LogGPWRatio(eleGPWKernNew, eleGPWKernOld));
  }

  else {
    return (GPWVal(eleGPWKernNew)/GPWVal(eleGPWKernOld));
  }
}

void CalculateGPWKern(double *eleGPWKern, double *eleGPWInSum, const int *eleNum) {
  const int nGPWIdx=NGPWIdx;
  int i;

  #pragma omp parallel for default(shared) private(i)
  for(i=0;i<nGPWIdx;i++) {
    int latId = GPWTrnLat[i];
    int offset = 0;
    int j, distWeightFlag;

    if (GPWThetaC[latId] > 0.0) {
      distWeightFlag = 1;
    }
    else {
      distWeightFlag = 0;
    }

    for (j = 0; j < i; j++) {
      offset += Nsite*GPWTrnSize[GPWTrnLat[j]];
    }

    if (GPWKernelFunc[latId] == 1) {
      eleGPWKern[i] = GPWKernel1(eleNum, Nsite, GPWTrnCfg[i], GPWTrnSize[latId],
                                 GPWTRSym[latId], GPWShift[latId], 0, 0);
    }

    else {
      CalculatePairDelta(eleGPWInSum+offset, eleNum, Nsite, GPWTrnCfg[i],
                         GPWTrnSize[latId]);

      if (GPWTRSym[latId]) {
        CalculatePairDeltaFlipped(eleGPWInSum+(GPWTrnCfgSz/2)*Nsite+offset, eleNum,
                                  Nsite, GPWTrnCfg[i], GPWTrnSize[latId]);
      }

      if (GPWKernelFunc[latId] < 0) {
        ComputeInSumExp(eleGPWInSum+offset, GPWSysPlaquetteIdx[latId],
                        Nsite, GPWTrnPlaquetteIdx[latId],
                        GPWTrnSize[latId], GPWPlaquetteSizes[latId],
                        GPWDistWeights, GPWDistWeightIdx[i]);
        if (GPWTRSym[latId]) {
          ComputeInSumExp(eleGPWInSum+(GPWTrnCfgSz/2)*Nsite+offset,
                          GPWSysPlaquetteIdx[latId], Nsite,
                          GPWTrnPlaquetteIdx[latId], GPWTrnSize[latId],
                          GPWPlaquetteSizes[latId], GPWDistWeights,
                          GPWDistWeightIdx[i]);
        }
        if (GPWKernelFunc[latId] == -1) {
          eleGPWKern[i] = ComputeExpKernel(Nsite, GPWTrnSize[latId],
                                           GPWTRSym[latId], GPWShift[latId], 0, 0,
                                           eleGPWInSum+offset,
                                           eleGPWInSum+(GPWTrnCfgSz/2)*Nsite+offset,
                                           0);
        }
        else if (GPWKernelFunc[latId] == -2) {
          eleGPWKern[i] = ComputeExpKernel(Nsite, GPWTrnSize[latId],
                                           GPWTRSym[latId], GPWShift[latId], 0, 0,
                                           eleGPWInSum+offset,
                                           eleGPWInSum+(GPWTrnCfgSz/2)*Nsite+offset,
                                           1);
        }
        else {
          printf("Error, bad kernel type\n");
        }
      }

      else if (GPWKernelFunc[latId] == 0) {
        ComputeInSum(eleGPWInSum+offset, GPWSysPlaquetteIdx[latId],
                     Nsite, GPWTrnPlaquetteIdx[latId],
                     GPWTrnSize[latId], GPWPlaquetteSizes[latId],
                     GPWDistList[latId], distWeightFlag);
        if (GPWTRSym[latId]) {
          ComputeInSum(eleGPWInSum+(GPWTrnCfgSz/2)*Nsite+offset,
                       GPWSysPlaquetteIdx[latId], Nsite,
                       GPWTrnPlaquetteIdx[latId], GPWTrnSize[latId],
                       GPWPlaquetteSizes[latId], GPWDistList[latId],
                       distWeightFlag);
        }
        eleGPWKern[i] = ComputeKernel(Nsite, GPWTrnSize[latId],
                                      GPWPower[latId], GPWThetaVar[latId],
                                      GPWNorm[latId], GPWTRSym[latId],
                                      GPWShift[latId], 0, 0,
                                      eleGPWInSum+offset,
                                      eleGPWInSum+(GPWTrnCfgSz/2)*Nsite+offset);
      }

      else {
        eleGPWKern[i] = ComputeKernelN(eleNum, GPWSysPlaquetteIdx[latId], Nsite,
                                       GPWTrnCfg[i], GPWTrnPlaquetteIdx[latId], GPWTrnSize[latId],
                                       GPWKernelFunc[latId], GPWTRSym[latId],
                                       GPWShift[latId], 0, 0, eleGPWInSum+offset,
                                       eleGPWInSum+(GPWTrnCfgSz/2)*Nsite+offset);
      }
    }
  }
  return;
}

void UpdateGPWKern(const int ri, const int rj, double *eleGPWKernNew,
                   double *eleGPWInSumNew, const double *eleGPWKernOld,
                   const double *eleGPWInSumOld, const int *eleNum) {
  const int nGPWIdx=NGPWIdx;
  int i;
  memcpy(eleGPWInSumNew, eleGPWInSumOld, sizeof(double)*GPWTrnCfgSz*Nsite);

  #pragma omp parallel for default(shared) private(i)
  for(i=0;i<nGPWIdx;i++) {
    int latId = GPWTrnLat[i];
    int offset = 0;
    int j, distWeightFlag;

    if (GPWThetaC[latId] > 0.0) {
      distWeightFlag = 1;
    }
    else {
      distWeightFlag = 0;
    }

    for (j = 0; j < i; j++) {
      offset += Nsite*GPWTrnSize[GPWTrnLat[j]];
    }

    if (GPWKernelFunc[latId] == 1) {
      eleGPWKernNew[i] = GPWKernel1(eleNum, Nsite, GPWTrnCfg[i], GPWTrnSize[latId],
                                    GPWTRSym[latId], GPWShift[latId], 0, 0);
    }

    else {
      UpdateDelta(eleGPWInSumNew+offset, eleNum, Nsite, GPWTrnCfg[i],
                  GPWTrnSize[latId], ri, rj);

      if (GPWTRSym[latId]) {
        UpdateDeltaFlipped(eleGPWInSumNew+(GPWTrnCfgSz/2)*Nsite+offset, eleNum,
                           Nsite, GPWTrnCfg[i], GPWTrnSize[latId], ri, rj);
      }

      if (GPWKernelFunc[latId] < 0) {
        UpdateInSumExp(eleGPWInSumNew+offset, eleGPWInSumOld+offset,
                       GPWSysPlaquetteIdx[latId], Nsite, GPWTrnPlaquetteIdx[latId],
                       GPWTrnSize[latId], GPWPlaquetteSizes[latId],
                       GPWDistWeights, GPWDistWeightIdx[i], GPWSysPlaqHash[latId],
                       GPWSysPlaqHashSz[latId], ri, rj);

        if (GPWTRSym[latId]) {
          UpdateInSumExp(eleGPWInSumNew+(GPWTrnCfgSz/2)*Nsite+offset,
                         eleGPWInSumOld+(GPWTrnCfgSz/2)*Nsite+offset,
                         GPWSysPlaquetteIdx[latId], Nsite,
                         GPWTrnPlaquetteIdx[latId], GPWTrnSize[latId],
                         GPWPlaquetteSizes[latId], GPWDistWeights,
                         GPWDistWeightIdx[i], GPWSysPlaqHash[latId],
                         GPWSysPlaqHashSz[latId], ri, rj);
        }
        if (GPWKernelFunc[latId] == -1) {
          eleGPWKernNew[i] = ComputeExpKernel(Nsite, GPWTrnSize[latId],
                                            GPWTRSym[latId], GPWShift[latId],
                                            0, 0, eleGPWInSumNew+offset,
                                            eleGPWInSumNew+(GPWTrnCfgSz/2)*Nsite+offset,
                                            0);
        }
        else if (GPWKernelFunc[latId] == -2) {
          eleGPWKernNew[i] = ComputeExpKernel(Nsite, GPWTrnSize[latId],
                                            GPWTRSym[latId], GPWShift[latId],
                                            0, 0, eleGPWInSumNew+offset,
                                            eleGPWInSumNew+(GPWTrnCfgSz/2)*Nsite+offset,
                                            1);
        }
        else {
          printf("Error, bad kernel type\n");
        }
      }
      else if (GPWKernelFunc[latId] == 0) {
        UpdateInSum(eleGPWInSumNew+offset, eleGPWInSumOld+offset,
                    GPWSysPlaquetteIdx[latId], Nsite, GPWTrnPlaquetteIdx[latId],
                    GPWTrnSize[latId], GPWPlaquetteSizes[latId],
                    GPWDistList[latId], distWeightFlag, GPWSysPlaqHash[latId],
                    GPWSysPlaqHashSz[latId], ri, rj);

        if (GPWTRSym[latId]) {
          UpdateInSum(eleGPWInSumNew+(GPWTrnCfgSz/2)*Nsite+offset,
                      eleGPWInSumOld+(GPWTrnCfgSz/2)*Nsite+offset,
                      GPWSysPlaquetteIdx[latId], Nsite,
                      GPWTrnPlaquetteIdx[latId], GPWTrnSize[latId],
                      GPWPlaquetteSizes[latId], GPWDistList[latId],
                      distWeightFlag, GPWSysPlaqHash[latId],
                      GPWSysPlaqHashSz[latId], ri, rj);
        }
        eleGPWKernNew[i] = ComputeKernel(Nsite, GPWTrnSize[latId],
                                         GPWPower[latId], GPWThetaVar[latId],
                                         GPWNorm[latId], GPWTRSym[latId],
                                         GPWShift[latId], 0, 0, eleGPWInSumNew+offset,
                                         eleGPWInSumNew+(GPWTrnCfgSz/2)*Nsite+offset);
      }

      else {
        eleGPWKernNew[i] = ComputeKernelN(eleNum, GPWSysPlaquetteIdx[latId], Nsite,
                                          GPWTrnCfg[i], GPWTrnPlaquetteIdx[latId], GPWTrnSize[latId],
                                          GPWKernelFunc[latId], GPWTRSym[latId],
                                          GPWShift[latId], 0, 0, eleGPWInSumNew+offset,
                                          eleGPWInSumNew+(GPWTrnCfgSz/2)*Nsite+offset);
      }
    }
  }
  return;
}
