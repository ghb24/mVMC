// TODO include license and description

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include "gpw_exp_kernel.h"
#include "gpw_kernel.h"

void ComputeInSumExp(double complex *inSum, const int *plaquetteAIdx, const int *cfgA,
                     const int sizeA, const int *plaquetteBIdx, const int *cfgB,
                     const int sizeB, const int plaquetteSize, const double complex *distWeights,
                     const int *distWeightIdx, const int shift, const int startIdA, const int startIdB,
                     const int tRSym) {
  int i, a, k, iPlaqSize, aPlaqSize, tSym, count;
  double complex innerSum;
  double complex element;

  int shiftSys = 1 + startIdA;
  int shiftTrn = 1 + startIdB;
  int translationSys = 1;
  int translationTrn = 1;

  if (abs(shift) & 1) {
    shiftSys = sizeA;
    if (shift < 0) {
      translationSys = sizeB;
    }
  }
  if ((abs(shift) & 2) >> 1) {
    shiftTrn = sizeB;

    if (shift < 0) {
      translationTrn = sizeA;
    }
  }


  count = 0;
  for (tSym = 0; tSym <= tRSym; tSym++) {
    iPlaqSize = 0;
    for (i = startIdA; i < shiftSys; i+=translationSys) {
      aPlaqSize = 0;
      for (a = startIdB; a < shiftTrn; a+=translationTrn) {
        innerSum = 0.0;
        for (k = 0; k < plaquetteSize; k++) {
          if (!delta(cfgA, sizeA, cfgB, sizeB, plaquetteAIdx[i*plaquetteSize+k], plaquetteBIdx[a*plaquetteSize+k], tSym)) {
            element = distWeights[distWeightIdx[sizeB*k + a]];
            innerSum += element;
          }
        }
        inSum[count] = innerSum;
        aPlaqSize += plaquetteSize;
        count++;
      }
      iPlaqSize += plaquetteSize;
    }
  }
}

void UpdateInSumExp(double complex *inSumNew, const int *cfgAOldReduced, const int *cfgANew,
                    const int *plaquetteAIdx, const int sizeA,
                    const int *plaquetteBIdx, const int *cfgB, const int sizeB,
                    const int plaquetteSize,
                    const double complex *distWeights,
                    const int *distWeightIdx, const int shift, const int startIdA,
                    const int startIdB,int **plaqHash, int *plaqHashSz, const int siteA,
                    const int siteB, const int tRSym) {
  int i, a, k, countA, countB, id, tSym, count;
  const int *hashListA, *hashListB;
  double complex element;

  int shiftSys = 1 + startIdA;
  int shiftTrn = 1 + startIdB;
  int translationSys = 1;
  int translationTrn = 1;

  if (abs(shift) & 1) {
    shiftSys = sizeA;
    if (shift < 0) {
      translationSys = sizeB;
    }
  }
  if ((abs(shift) & 2) >> 1) {
    shiftTrn = sizeB;

    if (shift < 0) {
      translationTrn = sizeA;
    }
  }

  count = 0;
  for (tSym = 0; tSym <= tRSym; tSym++) {
    for (i = startIdA; i < shiftSys; i+=translationSys) {
      countA = plaqHashSz[sizeA*i + siteA];
      countB = plaqHashSz[sizeA*i + siteB];
      hashListA = plaqHash[sizeA*i + siteA];
      hashListB = plaqHash[sizeA*i + siteB];

      for (a = startIdB; a < shiftTrn; a+=translationTrn) {
        for (k = 0; k < countA; k++) {
          id = hashListA[k];
          if (delta(cfgANew, sizeA, cfgB, sizeB, siteA, plaquetteBIdx[a*plaquetteSize+id], tSym)) {
            element = distWeights[distWeightIdx[sizeB*id + a]];
            inSumNew[count] -= element;
          }
          if (delta(cfgAOldReduced, 2, cfgB, sizeB, 0, plaquetteBIdx[a*plaquetteSize+id], tSym)) {
            element = distWeights[distWeightIdx[sizeB*id + a]];
            inSumNew[count] += element;
          }
        }
        if (siteA != siteB) {
          for (k = 0; k < countB; k++) {
            id = hashListB[k];
            if (delta(cfgANew, sizeA, cfgB, sizeB, siteB, plaquetteBIdx[a*plaquetteSize+id], tSym)) {
              element = distWeights[distWeightIdx[sizeB*id + a]];
              inSumNew[count] -= element;
            }
            if (delta(cfgAOldReduced, 2, cfgB, sizeB, 1, plaquetteBIdx[a*plaquetteSize+id], tSym)) {
              element = distWeights[distWeightIdx[sizeB*id + a]];
              inSumNew[count] += element;
            }
          }
        }
        count++;
      }
    }
  }
}

double ComputeExpKernel(const int *cfgA, const int sizeA, const int *cfgB,
                        const int sizeB, const int tRSym,
                        const int shift, const int startIdA, const int startIdB,
                        const double complex *inSum, const int centralDelta) {
  int i, a, tSym, count;
  int shiftSys = 1 + startIdA;
  int shiftTrn = 1 + startIdB;
  int translationSys = 1;
  int translationTrn = 1;
  double kernel = 0.0;

  if (abs(shift) & 1) {
    shiftSys = sizeA;
    if (shift < 0) {
      translationSys = sizeB;
    }
  }
  if ((abs(shift) & 2) >> 1) {
    shiftTrn = sizeB;

    if (shift < 0) {
      translationTrn = sizeA;
    }
  }

  count = 0;
  if (centralDelta) {
    for (tSym = 0; tSym <= tRSym; tSym++) {
      for (i = startIdA; i < shiftSys; i+=translationSys) {
        for (a = startIdB; a < shiftTrn; a+=translationTrn) {
          if (delta(cfgA, sizeA, cfgB, sizeB, i, a, tSym)) {
            kernel += exp(-creal(inSum[count]));
          }
          count++;
        }
      }
    }
  }
  else {
    for (tSym = 0; tSym <= tRSym; tSym++) {
      for (i = startIdA; i < shiftSys; i+=translationSys) {
        for (a = startIdB; a < shiftTrn; a+=translationTrn) {
          kernel += exp(-creal(inSum[count]));
          count++;
        }
      }
    }
  }
  return kernel;
}

double GPWExpKernel(const int *cfgA, const int *plaquetteAIdx, const int sizeA,
                    const int *cfgB, const int *plaquetteBIdx, const int sizeB,
                    const int tRSym, const int shift, const int startIdA,
                    const int startIdB, const int plaquetteSize,
                    const double complex *distWeights, const int numDistWeights,
                    const int *distWeightIdx, const int centralDelta,
                    double complex *workspace) {
  double complex *innerSum = workspace;

  ComputeInSumExp(innerSum, plaquetteAIdx, cfgA, sizeA, plaquetteBIdx,
                  cfgB, sizeB, plaquetteSize, distWeights,
                  distWeightIdx, shift, startIdA, startIdB, tRSym);

  return ComputeExpKernel(cfgA, sizeA, cfgB, sizeB, tRSym, shift, startIdA, startIdB,
                          innerSum, centralDelta);
}

void GPWExpKernelMat(const unsigned long *configsAUp,
                     const unsigned long *configsADown, const int *neighboursA,
                     const int sizeA, const int numA,
                     const unsigned long *configsBUp,
                     const unsigned long *configsBDown,
                     const int *neighboursB, const int sizeB, const int numB,
                     const double *distWeights, const int numDistWeights,
                     int **distWeightIdx, const int dim, const int rC,
                     const int tRSym, const int shift, const int startIdA,
                     const int startIdB, const int centralDelta,
                     const int symmetric, double *kernelMatr) {
  int i, j, workspaceSize;
  int **cfgsA, **cfgsB;
  int plaquetteSize;
  int *plaquetteAIdx, *plaquetteBIdx, *distList;

  double complex *distWeightsCompl = (double complex*) malloc(sizeof(double complex) *
                                                              numDistWeights);
  for (i = 0; i < numDistWeights; i++) {
    distWeightsCompl[i] = distWeights[i] + 0.0*I;
  }

  if (centralDelta) {
    plaquetteSize = SetupPlaquetteIdx(rC, neighboursA, sizeA, neighboursB,
                                      sizeB, dim, &plaquetteAIdx,
                                      &plaquetteBIdx, &distList, 0);
  }
  else {
    plaquetteSize = SetupPlaquetteIdx(rC, neighboursA, sizeA, neighboursB,
                                      sizeB, dim, &plaquetteAIdx,
                                      &plaquetteBIdx, &distList, 1);
  }

  cfgsA = (int**) malloc(sizeof(int*) * numA);

  for (i = 0; i < numA; i++) {
    cfgsA[i] = (int*) malloc(sizeof(int) * 2 * sizeA);

    for(j = 0; j < sizeA; j++) {
      cfgsA[i][j] = (configsAUp[i] >> j) & 1;
      cfgsA[i][j+sizeA] = (configsADown[i] >> j) & 1;
    }
  }

  workspaceSize = 1;
  if (abs(shift) & 1) {
    workspaceSize *= sizeA;
  }
  if ((abs(shift) & 2) >> 1) {
    workspaceSize *= sizeB;
  }
  if (tRSym) {
    workspaceSize *= 2;
  }

  if (!symmetric) {
    cfgsB = (int**) malloc(sizeof(int*) * numB);
    for (i = 0; i < numB; i++) {
     cfgsB[i] = (int*) malloc(sizeof(int) * 2 * sizeB);

     for(j = 0; j < sizeB; j++) {
       cfgsB[i][j] = (configsBUp[i] >> j) & 1;
       cfgsB[i][j+sizeB] = (configsBDown[i] >> j) & 1;
     }
    }

    #pragma omp parallel default(shared) private(i, j)
    {
      double complex *workspace = (double complex*)malloc(sizeof(double complex)*(workspaceSize));
      #pragma omp for
      for (i = 0; i < numA; i++) {
        for (j = 0; j < numB; j++) {
          kernelMatr[i*numB + j] = GPWExpKernel(cfgsA[i], plaquetteAIdx, sizeA,
                                                cfgsB[j], plaquetteBIdx, sizeB,
                                                tRSym, shift, startIdA, startIdB,
                                                plaquetteSize, distWeightsCompl,
                                                numDistWeights, distWeightIdx[j],
                                                centralDelta, workspace);
        }
      }
      free(workspace);
    }

    for (i = 0; i < numB; i++) {
     free(cfgsB[i]);
    }
    free(cfgsB);
  }

  else {
    #pragma omp parallel default(shared) private(i, j)
    {
      double complex *workspace = (double complex*)malloc(sizeof(double complex)*(workspaceSize));

      #pragma omp for
      for (i = 0; i < numA; i++) {
        for (j = 0; j <= i; j++) {
          kernelMatr[i*numA + j] = GPWExpKernel(cfgsA[i], plaquetteAIdx, sizeA,
                                                cfgsA[j], plaquetteAIdx, sizeA,
                                                tRSym, shift, startIdA, startIdB,
                                                plaquetteSize, distWeightsCompl,
                                                numDistWeights, distWeightIdx[j],
                                                centralDelta, workspace);
        }
      }
      free(workspace);
    }
  }

  for (i = 0; i < numA; i++) {
    free(cfgsA[i]);
  }

  free(cfgsA);

  FreeMemPlaquetteIdx(plaquetteAIdx, plaquetteBIdx, distList);

  free(distWeightsCompl);
}

void GPWExpKernelVec(const unsigned long *configsAUp, const unsigned long *configsADown,
                     const int *neighboursA, const int sizeA, const int numA,
                     const int *configRef, const int *neighboursRef, const int sizeRef,
                     const double *distWeights, const int numDistWeights,
                     const int *distWeightIdx, const int dim, const int rC,
                     const int tRSym, const int shift, const int startIdA,
                     const int startIdB, const int centralDelta,
                     double *kernelVec) {
  int i, j, workspaceSize;
  int **cfgsA;

  int plaquetteSize;
  int *plaquetteAIdx, *plaquetteBIdx, *distList;

  double complex *distWeightsCompl = (double complex*) malloc(sizeof(double complex) *
                                                              numDistWeights);
  for (i = 0; i < numDistWeights; i++) {
    distWeightsCompl[i] = distWeights[i] + 0.0*I;
  }

  if (centralDelta) {
    plaquetteSize = SetupPlaquetteIdx(rC, neighboursA, sizeA, neighboursRef,
                                      sizeRef, dim, &plaquetteAIdx,
                                      &plaquetteBIdx, &distList, 0);
  }
  else {
    plaquetteSize = SetupPlaquetteIdx(rC, neighboursA, sizeA, neighboursRef,
                                      sizeRef, dim, &plaquetteAIdx,
                                      &plaquetteBIdx, &distList, 1);
  }

  cfgsA = (int**) malloc(sizeof(int*) * numA);

  for (i = 0; i < numA; i++) {
    cfgsA[i] = (int*) malloc(sizeof(int) * 2 * sizeA);

    for(j = 0; j < sizeA; j++) {
      cfgsA[i][j] = (configsAUp[i] >> j) & 1;
      cfgsA[i][j+sizeA] = (configsADown[i] >> j) & 1;
    }
  }

  workspaceSize = 1;
  if (abs(shift) & 1) {
    workspaceSize *= sizeA;
  }
  if ((abs(shift) & 2) >> 1) {
    workspaceSize *= sizeRef;
  }
  if (tRSym) {
    workspaceSize *= 2;
  }

  #pragma omp parallel default(shared) private(i, j)
  {
    double complex *workspace = (double complex*)malloc(sizeof(double complex)*(workspaceSize));

    #pragma omp for
    for (i = 0; i < numA; i++) {
      kernelVec[i] = GPWExpKernel(cfgsA[i], plaquetteAIdx, sizeA, configRef,
                                  plaquetteBIdx, sizeRef, tRSym, shift,
                                  startIdA, startIdB, plaquetteSize, distWeightsCompl,
                                  numDistWeights, distWeightIdx, centralDelta,
                                  workspace);
    }
    free(workspace);
  }

  for (i = 0; i < numA; i++) {
    free(cfgsA[i]);
  }
  free(cfgsA);

  FreeMemPlaquetteIdx(plaquetteAIdx, plaquetteBIdx, distList);

  free(distWeightsCompl);
}


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
