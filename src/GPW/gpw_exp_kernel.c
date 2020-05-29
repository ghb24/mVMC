// TODO include license and description

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "gpw_exp_kernel.h"
#include "gpw_kernel.h"

void ComputeInSumExp(double *inSum, const int *plaquetteAIdx, const int *cfgA,
                     const int sizeA, const int *plaquetteBIdx, const int *cfgB,
                     const int sizeB, const int plaquetteSize, const double complex *distWeights,
                     const int *distWeightIdx, const int shift, const int startIdA, const int startIdB,
                     const int tRSym) {
  int i, a, k, iPlaqSize, aPlaqSize, tSym, count;
  double innerSum, element;

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

  iPlaqSize = 0;
  count = 0;

  for (tSym = 0; tSym <= tRSym; tSym++) {
    for (i = startIdA; i < shiftSys; i+=translationSys) {
      aPlaqSize = 0;
      for (a = startIdB; a < shiftTrn; a+=translationTrn) {
        innerSum = 0.0;
        for (k = 0; k < plaquetteSize; k++) {
          if (!delta(cfgA, sizeA, cfgB, sizeB, plaquetteAIdx[iPlaqSize+k], plaquetteBIdx[aPlaqSize+k], tSym)) {
            element = creal(distWeights[distWeightIdx[sizeB*k + a]]);
            innerSum += element*element;
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

void UpdateInSumExp(double *inSumNew, const int *cfgAOldReduced, const int *cfgANew,
                    const int *plaquetteAIdx, const int sizeA,
                    const int *plaquetteBIdx, const int *cfgB, const int sizeB,
                    const int plaquetteSize,
                    const double complex *distWeights,
                    const int *distWeightIdx, const int shift, const int startIdA,
                    const int startIdB,int **plaqHash, int *plaqHashSz, const int siteA,
                    const int siteB, const int tRSym) {
  int i, a, k, countA, countB, id, countAId, countBId, tSym, count;
  const int *hashListA, *hashListB;
  double element;

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

  countAId = siteA;
  countBId = siteB;

  for (tSym = 0; tSym <= tRSym; tSym++) {
    for (i = startIdA; i < shiftSys; i+=translationSys) {
      countA = plaqHashSz[countAId];
      countB = plaqHashSz[countBId];
      hashListA = plaqHash[countAId];
      hashListB = plaqHash[countBId];

      for (a = startIdB; a < shiftTrn; a+=translationTrn) {
        for (k = 0; k < countA; k++) {
          id = hashListA[k];
          if (delta(cfgANew, sizeA, cfgB, sizeB, siteA, plaquetteBIdx[a*plaquetteSize+id], tSym)) {
            element = creal(distWeights[distWeightIdx[sizeB*id + a]]);
            inSumNew[count] -= element*element;
          }
          if (delta(cfgAOldReduced, 2, cfgB, sizeB, 0, plaquetteBIdx[a*plaquetteSize+id], tSym)) {
            element = creal(distWeights[distWeightIdx[sizeB*id + a]]);
            inSumNew[count] += element*element;
          }
        }
        if (siteA != siteB) {
          for (k = 0; k < countB; k++) {
            id = hashListB[k];
            if (delta(cfgANew, sizeA, cfgB, sizeB, siteB, plaquetteBIdx[a*plaquetteSize+id], tSym)) {
              element = creal(distWeights[distWeightIdx[sizeB*id + a]]);
              inSumNew[count] -= element*element;
            }
            if (delta(cfgAOldReduced, 2, cfgB, sizeB, 1, plaquetteBIdx[a*plaquetteSize+id], tSym)) {
              element = creal(distWeights[distWeightIdx[sizeB*id + a]]);
              inSumNew[count] += element*element;
            }
          }
        }
        count++;
      }
      countAId += sizeA;
      countBId += sizeA;
    }
  }
}

double ComputeExpKernel(const int *cfgA, const int sizeA, const int *cfgB,
                        const int sizeB, const int tRSym,
                        const int shift, const int startIdA, const int startIdB,
                        const double *inSum, const int centralDelta) {
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
            kernel += exp(-inSum[count]);
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
          kernel += exp(-inSum[count]);
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
                    const double *distWeights, const int numDistWeights,
                    const int *distWeightIdx, const int centralDelta,
                    double *workspace) {
  int i;

  double *innerSum = workspace;
  double *innerSumFlipped = innerSum + sizeA*sizeB;

  double complex *distWeightsCompl = (double complex*) malloc(sizeof(double complex) *
                                                              numDistWeights);
  for (i = 0; i < numDistWeights; i++) {
    distWeightsCompl[i] = distWeights[i] + 0.0*I;
  }

  ComputeInSumExp(innerSum, plaquetteAIdx, cfgA, sizeA, plaquetteBIdx,
                  cfgB, sizeB, plaquetteSize, distWeightsCompl,
                  distWeightIdx, tRSym, shift, startIdA, startIdB);

  free(distWeightsCompl);

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
      double *workspace = (double*)malloc(sizeof(double)*(workspaceSize));
      #pragma omp for
      for (i = 0; i < numA; i++) {
        for (j = 0; j < numB; j++) {
          kernelMatr[i*numB + j] = GPWExpKernel(cfgsA[i], plaquetteAIdx, sizeA,
                                                cfgsB[j], plaquetteBIdx, sizeB,
                                                tRSym, shift, startIdA, startIdB,
                                                plaquetteSize, distWeights,
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
      double *workspace = (double*)malloc(sizeof(double)*(workspaceSize));

      #pragma omp for
      for (i = 0; i < numA; i++) {
        for (j = 0; j <= i; j++) {
          kernelMatr[i*numA + j] = GPWExpKernel(cfgsA[i], plaquetteAIdx, sizeA,
                                                cfgsA[j], plaquetteAIdx, sizeA,
                                                tRSym, shift, startIdA, startIdB,
                                                plaquetteSize, distWeights,
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
    double *workspace = (double*)malloc(sizeof(double)*(workspaceSize));

    #pragma omp for
    for (i = 0; i < numA; i++) {
      kernelVec[i] = GPWExpKernel(cfgsA[i], plaquetteAIdx, sizeA, configRef,
                                  plaquetteBIdx, sizeRef, tRSym, shift,
                                  startIdA, startIdB, plaquetteSize, distWeights,
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
}


void ComputeInSumExpBasisOpt(double *inSum, const int *plaquetteAIdx, const int sizeA,
                             const int plaquetteSize, const double complex *distWeights,
                             const int *eleNum, const int tRSym) {
  int i, k, occupationId, id, tSym,count;
  double innerSum, element;

  count = 0;
  for (tSym = 0; tSym <= tRSym; tSym++) {
    for (i = 0; i < sizeA; i++) {
      innerSum = 1.0;
      for (k = 0; k < plaquetteSize; k++) {
        if (tSym) {
          id = (1-eleNum[plaquetteAIdx[i*plaquetteSize+k]]) + 2 * (1-eleNum[plaquetteAIdx[i*plaquetteSize+k]+sizeA]);
        }
        else {
          id = eleNum[plaquetteAIdx[i*plaquetteSize+k]] + 2 * eleNum[plaquetteAIdx[i*plaquetteSize+k]+sizeA];
        }
        element = creal(distWeights[k*4+id]);
        innerSum *= element;
      }
      inSum[count] = innerSum;
      count++;
    }
  }
}

void UpdateInSumExpBasisOpt(double *inSumNew, const int *cfgOldReduced, const int *eleNum,
                            const int *plaquetteAIdx, const int sizeA, const int plaquetteSize,
                            const double complex *distWeights, int **plaqHash, const int ri,
                            const int rj, const int tRSym) {
  int i, a, k, occupationIdOld, occupationIdNew, tSym;
  double elementOld, elementNew;

  for (tSym = 0; tSym <= tRSym; tSym++) {
    if (tSym) {
      occupationIdOld = (1-cfgOldReduced[0]) + 2 * (1-cfgOldReduced[2]);
      occupationIdNew = (1-eleNum[ri]) + 2 * (1-eleNum[ri+sizeA]);
    }
    else {
      occupationIdOld = cfgOldReduced[0] + 2 * cfgOldReduced[2];
      occupationIdNew = eleNum[ri] + 2 * eleNum[ri+sizeA];
    }
    for (i = 0; i < sizeA; i++) {
      k = plaqHash[ri + i *sizeA][0];
      elementOld = creal(distWeights[k*4+occupationIdOld]);
      elementNew = creal(distWeights[k*4+occupationIdNew]);

      inSumNew[tSym * sizeA + i] /= elementOld;
      inSumNew[tSym * sizeA + i] *= elementNew;
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
      for (i = 0; i < sizeA; i++) {
        k = plaqHash[rj + i *sizeA][0];
        elementOld = creal(distWeights[k*4+occupationIdOld]);
        elementNew = creal(distWeights[k*4+occupationIdNew]);
        inSumNew[tSym * sizeA + i] /= elementOld;
        inSumNew[tSym * sizeA + i] *= elementNew;
      }
    }
  }
}

double ComputeExpKernelBasisOpt(const int size, const int tRSym,
                                const double *inSum) {
  int i, tSym, count;
  double kernel = 0.0;

  count = 0;
  for (tSym = 0; tSym <= tRSym; tSym++) {
    for (i = 0; i < size; i++) {
      kernel -= inSum[count];
      count++;
    }
  }
  return kernel;
}
