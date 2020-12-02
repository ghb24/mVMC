/*
TODO: Add License + Description*/
#include "global.h"
#include "QGPS.h"
#include "include/global.h"
#include <omp.h>

void ComputeInSumExpBasisOpt(double complex *inSum, const int *plaquetteAIdx, const int sizeA,
                             const int plaquetteSize, const double complex *distWeights,
                             const int *eleNum, const int tRSym, const int shift) {
  int i, k, id, tSym,count;
  double complex innerSum;
  double complex element;
  int shiftSys = 1;

  if (abs(shift) & 1) {
    shiftSys = sizeA;
  }

  count = 0;
  for (tSym = 0; tSym <= tRSym; tSym++) {
    for (i = 0; i < shiftSys; i++) {
      innerSum = 1.0;
      for (k = 0; k < plaquetteSize; k++) {
        if (tSym) {
          id = (1-eleNum[plaquetteAIdx[i*plaquetteSize+k]]) + 2 * (1-eleNum[plaquetteAIdx[i*plaquetteSize+k]+sizeA]);
        }
        else {
          id = eleNum[plaquetteAIdx[i*plaquetteSize+k]] + 2 * eleNum[plaquetteAIdx[i*plaquetteSize+k]+sizeA];
        }
        element = distWeights[k*4+id];
        innerSum *= element;
      }
      inSum[count] = innerSum;
      count++;
    }
  }
}

void UpdateInSumExpBasisOpt(double complex *inSumNew, const int *cfgOldReduced, const int *eleNum,
                            const int *plaquetteAIdx, const int sizeA, const int plaquetteSize,
                            const double complex *distWeights, int **plaqHash, const int ri,
                            const int rj, const int tRSym, const int shift) {
  int i, k, occupationIdOld, occupationIdNew, tSym;
  double complex elementOld;
  double complex elementNew;
  int shiftSys = 1;

  if (abs(shift) & 1) {
    shiftSys = sizeA;
  }

  for (tSym = 0; tSym <= tRSym; tSym++) {
    if (tSym) {
      occupationIdOld = (1-cfgOldReduced[0]) + 2 * (1-cfgOldReduced[2]);
      occupationIdNew = (1-eleNum[ri]) + 2 * (1-eleNum[ri+sizeA]);
    }
    else {
      occupationIdOld = cfgOldReduced[0] + 2 * cfgOldReduced[2];
      occupationIdNew = eleNum[ri] + 2 * eleNum[ri+sizeA];
    }
    for (i = 0; i < shiftSys; i++) {
      k = plaqHash[ri + i *sizeA][0];
      elementOld = distWeights[k*4+occupationIdOld];
      elementNew = distWeights[k*4+occupationIdNew];

      inSumNew[tSym * shiftSys + i] /= elementOld;
      inSumNew[tSym * shiftSys + i] *= elementNew;
    }
    if (ri != rj) {
      if (tSym) {
        occupationIdOld = (1-cfgOldReduced[1]) + 2 * (1-cfgOldReduced[3]);
        occupationIdNew = (1-eleNum[rj]) + 2 * (1-eleNum[rj+sizeA]);
      }
      else {
        occupationIdOld = cfgOldReduced[1] + 2 * cfgOldReduced[3];
        occupationIdNew = eleNum[rj] + 2 * eleNum[rj+sizeA];
      }
      for (i = 0; i < shiftSys; i++) {
        k = plaqHash[rj + i *sizeA][0];
        elementOld = distWeights[k*4+occupationIdOld];
        elementNew = distWeights[k*4+occupationIdNew];
        inSumNew[tSym * shiftSys + i] /= elementOld;
        inSumNew[tSym * shiftSys + i] *= elementNew;
      }
    }
  }
}

double complex ComputeExpKernelBasisOpt(const int size, const int tRSym,
                                        const double complex *inSum, const int shift) {
  int i, tSym, count;
  double complex kernel = 0.0;
  int shiftSys = 1;

  if (abs(shift) & 1) {
    shiftSys = size;
  }

  count = 0;
  for (tSym = 0; tSym <= tRSym; tSym++) {
    for (i = 0; i < shiftSys; i++) {
      kernel += inSum[count];
      count++;
    }
  }
  return kernel;
}


inline double complex LogQGPSVal(const double complex *workspace) {
  return clog(QGPSVal(workspace));
}

inline double complex QGPSExpansionargument(const double complex *eleGPWInSum) {
  int idx;
  double complex expansionargument=0.0+0.0*I;

  int offset = 0;
  int inSumSize;

  for(idx=0;idx<NGPWIdx;idx++) {
    expansionargument += ComputeExpKernelBasisOpt(Nsite, GPWTRSym[0], eleGPWInSum+offset, GPWShift[0]);
    inSumSize = 1;
    if (abs(GPWShift[0]) & 1) {
      inSumSize *= Nsite;
    }
    if ((abs(GPWShift[0]) & 2) >> 1) {
      inSumSize *= GPWTrnSize[0];
    }
    if (GPWTRSym[0]) {
      inSumSize *= 2;
    }

    offset += inSumSize;
  }

  return expansionargument;
}

inline double complex QGPSVal(const double complex *eleGPWInSum) {
  int i;
  double complex expansionargument;
  double complex z=0.0+0.0*I;
  int factorial = 1;

  expansionargument = QGPSExpansionargument(eleGPWInSum);

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

inline double complex LogQGPSRatio(const double complex *eleGPWInSumNew, const double complex *eleGPWInSumOld) {
  return (LogQGPSVal(eleGPWInSumNew)-LogQGPSVal(eleGPWInSumOld));
}

inline double complex QGPSRatio(const double complex *eleGPWInSumNew, const double complex *eleGPWInSumOld) {
  return (QGPSVal(eleGPWInSumNew)/QGPSVal(eleGPWInSumOld));
}

void CalculateQGPSInsum(double complex *eleGPWInSum, const int *eleNum) {
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


    ComputeInSumExpBasisOpt(eleGPWInSum+offset, GPWSysPlaquetteIdx[latId],
                            Nsite, GPWPlaquetteSizes[latId],
                            GPWDistWeights + offsetDistWeights, eleNum, GPWTRSym[latId], GPWShift[latId]);
  }
  return;
}

void UpdateQGPSInSum(const int ri, const int rj, const int *cfgOldReduced,
                     double complex *eleGPWInSumNew, const double complex *eleGPWInSumOld,
                     const int *eleNum) {
  const int nGPWIdx=NGPWIdx;
  int i;

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

    UpdateInSumExpBasisOpt(eleGPWInSumNew+offset, cfgOldReduced, eleNum,
                           GPWSysPlaquetteIdx[latId], Nsite, GPWPlaquetteSizes[latId],
                           GPWDistWeights + offsetDistWeights,
                           GPWSysPlaqHash[latId], ri, rj, GPWTRSym[latId], GPWShift[latId]);
  }
  return;
}


void calculateQGPSderivative(double complex *derivative, double complex *eleGPWInSum, int *eleNum) {
  int i, j, k, l;
  double complex expansionargument;
  double complex prefactor;
  int factorial; 
  int offset = NGPWIdx + NGPWTrnLat;
  
  for (i = 0; i < NGPWDistWeights; i++) {
    derivative[(offset+i)*2] = 0.0;
    derivative[(offset+i)*2+1] = 0.0*I;
  }

  #pragma omp parallel default(shared) private(i, j, k, l)
  {
    int matOffset, latId, trnSize, plaqSize, kernFunc, plaqId, basisOptOffset;
    int shiftSys, shiftTrn, translationSys, translationTrn, occId, inSumSize, tSym;
    double complex *inSum;
    double sumTargetLat;
    int *sysPlaquetteIdx, *trnPlaquetteIdx;
    double complex *distWeightDeriv = (double complex*)calloc(NGPWDistWeights, sizeof(double complex));

    #pragma omp for
    for (i = 0; i < NGPWIdx; i++) {
      latId = GPWTrnLat[i];
      trnSize = GPWTrnSize[latId];
      kernFunc = GPWKernelFunc[latId];
      plaqSize = GPWPlaquetteSizes[latId];
      sysPlaquetteIdx = GPWSysPlaquetteIdx[latId];
      trnPlaquetteIdx = GPWTrnPlaquetteIdx[latId];

      shiftSys = 1;
      shiftTrn = 1;
      translationSys = 1;
      translationTrn = 1;

      if (abs(GPWShift[latId]) & 1) {
        shiftSys = Nsite;
        if (GPWShift[latId] < 0) {
          translationSys = trnSize;
        }
      }
      if ((abs(GPWShift[latId]) & 2) >> 1) {
        shiftTrn = trnSize;

        if (GPWShift[latId] < 0) {
          translationTrn = Nsite;
        }
      }
      matOffset = 0;
      basisOptOffset = 0;
      for (k = 0; k < i; k++) {
        inSumSize = 1;
        if (abs(GPWShift[GPWTrnLat[k]]) & 1) {
          inSumSize *= Nsite;
        }
        if ((abs(GPWShift[GPWTrnLat[k]]) & 2) >> 1) {
          inSumSize *= GPWTrnSize[GPWTrnLat[k]];
        }
        if (GPWTRSym[GPWTrnLat[k]]) {
          inSumSize *= 2;
        }
        matOffset += inSumSize;
        basisOptOffset += 4*GPWPlaquetteSizes[GPWTrnLat[k]];
      }

      inSum = eleGPWInSum+matOffset;

      for (tSym = 0; tSym <= GPWTRSym[latId]; tSym++) {
        for (l = 0; l < shiftSys; l++) {
          for (k = 0; k < plaqSize; k++) {
            if (tSym) {
              occId = (1-eleNum[sysPlaquetteIdx[l*plaqSize+k]]) + 2 * (1-eleNum[sysPlaquetteIdx[l*plaqSize+k]+Nsite]);
            }
            else {
              occId = eleNum[sysPlaquetteIdx[l*plaqSize+k]] + 2 * eleNum[sysPlaquetteIdx[l*plaqSize+k]+Nsite];
            }
            distWeightDeriv[basisOptOffset + 4*k + occId] += GPWVar[i] * inSum[tSym * Nsite + l] / GPWDistWeights[basisOptOffset + 4*k + occId];
          }
        }
      }
    }

    #pragma omp critical
    for (i = 0; i < NGPWDistWeights; i++) {
      derivative[(offset+i)*2] += distWeightDeriv[i];
      derivative[(offset+i)*2 + 1] += I * distWeightDeriv[i];
    }
    free(distWeightDeriv);
  }


  offset += NGPWDistWeights;

  if (GPWExpansionOrder >= 0) {
    expansionargument = QGPSExpansionargument(eleGPWInSum);
    prefactor = 0.0;
    factorial = 1;
    for (j = 0; j < GPWExpansionOrder; j++) {
      prefactor += cpow(expansionargument, j)/factorial;
      factorial *= (j+1);
    }

    prefactor /= QGPSVal(eleGPWInSum);

    #pragma omp parallel for default(shared) private(i)
    for (i = 0; i < (NGPWDistWeights); i++) {
      derivative[i*2] *= prefactor;
      derivative[i*2 + 1] *= prefactor;
    }
  }
  return;
}