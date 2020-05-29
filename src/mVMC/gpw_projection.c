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
    else if (GPWKernelFunc[latId] == -3) {
      ComputeInSumExpBasisOpt(eleGPWInSum+offset, GPWSysPlaquetteIdx[latId],
                              Nsite, GPWPlaquetteSizes[latId],
                              GPWDistWeights + offsetDistWeights, eleNum, 0);
      eleGPWKern[i] = ComputeExpKernelBasisOpt(Nsite, GPWTRSym[latId], eleGPWInSum+offset);
    }
    else if (GPWKernelFunc[latId] < 0) {
      ComputeInSumExp(eleGPWInSum+offset, GPWSysPlaquetteIdx[latId],
                      eleNum, Nsite, GPWTrnPlaquetteIdx[latId],
                      GPWTrnCfg[i], GPWTrnSize[latId], GPWPlaquetteSizes[latId],
                      GPWDistWeights, GPWDistWeightIdx[i], GPWTRSym[latId],
                      GPWShift[latId], 0, 0);
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
                   GPWTrnCfg[latId], GPWTrnSize[latId], GPWPlaquetteSizes[latId],
                   GPWDistList[latId], GPWShift[latId], 0, 0,
                   GPWDistWeightPower[latId], GPWTRSym[latId]);
      eleGPWKern[i] = ComputeKernel(eleNum, Nsite, GPWTrnCfg[latId], GPWTrnSize[latId],
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
  return;
}

void UpdateGPWKern(const int ri, const int rj, const int *cfgOldReduced,
                   double *eleGPWKernNew, double *eleGPWInSumNew,
                   const double *eleGPWKernOld, const double *eleGPWInSumOld,
                   const int *eleNum) {
  const int nGPWIdx=NGPWIdx;
  int i;
  memcpy(eleGPWInSumNew, eleGPWInSumOld, sizeof(double)*GPWInSumSize);

  #pragma omp parallel for default(shared) private(i)
  for(i=0;i<nGPWIdx;i++) {
    int j, distWeightFlag, inSumSize;
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
    else if (GPWKernelFunc[latId] == -3) {
      UpdateInSumExpBasisOpt(eleGPWInSumNew+offset, cfgOldReduced, eleNum,
                             GPWSysPlaquetteIdx[latId], Nsite, GPWPlaquetteSizes[latId],
                             GPWDistWeights + offsetDistWeights,
                             GPWSysPlaqHash[latId], ri, rj, GPWTRSym[latId]);
      eleGPWKernNew[i] = ComputeExpKernelBasisOpt(Nsite, GPWTRSym[latId],
                                                  eleGPWInSumNew+offset);
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
  return;
}
