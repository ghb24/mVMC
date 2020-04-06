// TODO include license and description

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "gpw_exp_kernel.h"
#include "gpw_kernel.h"

void ComputeInSumExp(double *inSum, const int *plaquetteAIdx,
                     const int sizeA, const int *plaquetteBIdx, const int sizeB,
                     const int plaquetteSize, const double complex *distWeights,
                     const int *distWeightIdx) {
  int i, a, k, innerId, iPlaqSize, aPlaqSize;
  double innerSum, element;

  innerId = 0;
  iPlaqSize = 0;

  for (i = 0; i < sizeA; i++) {
    aPlaqSize = 0;
    for (a = 0; a < sizeB; a++) {
      innerSum = 0.0;
      for (k = 0; k < plaquetteSize; k++) {
        if (!signbit(inSum[plaquetteAIdx[iPlaqSize+k]*sizeB +
                           plaquetteBIdx[aPlaqSize+k]])) {
          element = creal(distWeights[distWeightIdx[sizeB*k + a]]);
          innerSum += element*element;
        }
      }
      inSum[innerId] = copysign(innerSum, inSum[innerId]);
      innerId++;
      aPlaqSize += plaquetteSize;
    }
    iPlaqSize += plaquetteSize;
  }
}

void UpdateInSumExp(double *inSumNew, const double *inSumOld,
                    const int *plaquetteAIdx, const int sizeA,
                    const int *plaquetteBIdx, const int sizeB,
                    const int plaquetteSize,
                    const double complex *distWeights,
                    const int *distWeightIdx, int **plaqHash,
                    int *plaqHashSz, const int siteA,
                    const int siteB) {
  int i, a, k, countA, countB, id, innerId, aPlaqSize, countAId, countBId;
  const int *hashListA, *hashListB;
  double element;

  const int plaqIdA = siteA*sizeB;
  const int plaqIdB = siteB*sizeB;

  innerId = 0;
  countAId = siteA;
  countBId = siteB;

  for (i = 0; i < sizeA; i++) {
    countA = plaqHashSz[countAId];
    countB = plaqHashSz[countBId];
    hashListA = plaqHash[countAId];
    hashListB = plaqHash[countBId];
    aPlaqSize = 0;

    for (a = 0; a < sizeB; a++) {
      for (k = 0; k < countA; k++) {
        id = hashListA[k];
        element = creal(distWeights[distWeightIdx[sizeB*id + a]]);
        element = element * element;
        if (!signbit(inSumNew[plaqIdA + plaquetteBIdx[aPlaqSize+id]])) {
          inSumNew[innerId] = copysign(fabs(inSumNew[innerId]) + element, inSumNew[innerId]);
        }
        if (!signbit(inSumOld[plaqIdA + plaquetteBIdx[aPlaqSize+id]])) {
          inSumNew[innerId] = copysign(fabs(inSumNew[innerId]) - element, inSumNew[innerId]);
        }
      }
      for (k = 0; k < countB; k++) {
        id = hashListB[k];
        element = creal(distWeights[distWeightIdx[sizeB*id + a]]);
        element = element * element;
        if (!signbit(inSumNew[plaqIdB + plaquetteBIdx[aPlaqSize+id]])) {
          inSumNew[innerId] = copysign(fabs(inSumNew[innerId]) + element, inSumNew[innerId]);
        }
        if (!signbit(inSumOld[plaqIdB + plaquetteBIdx[aPlaqSize+id]])) {
          inSumNew[innerId] = copysign(fabs(inSumNew[innerId]) - element, inSumNew[innerId]);
        }
      }
      aPlaqSize += plaquetteSize;
      innerId++;
    }
    countAId += sizeA;
    countBId += sizeA;
  }
}


double ComputeExpKernel(const int sizeA, const int sizeB, const int tRSym,
                        const int shift, const int startIdA, const int startIdB,
                        const double *inSum, const double *inSumFlipped) {
  int i, a;
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

  for (i = startIdA; i < shiftSys; i+=translationSys) {
    for (a = startIdB; a < shiftTrn; a+=translationTrn) {
      kernel += exp(-fabs(inSum[i*sizeB+a]));
    }
  }
  if (tRSym) {
    for (i = startIdA; i < shiftSys; i+=translationSys) {
      for (a = startIdB; a < shiftTrn; a+=translationTrn) {
        kernel += exp(-fabs(inSumFlipped[i*sizeB+a]));
      }
    }
    kernel /= 2.0;
  }
  return kernel;
}

double GPWExpKernel(const int *cfgA, const int *plaquetteAIdx, const int sizeA,
                    const int *cfgB, const int *plaquetteBIdx, const int sizeB,
                    const int tRSym, const int shift, const int startIdA,
                    const int startIdB, const int plaquetteSize,
                    const double *distWeights, const int numDistWeights,
                    const int *distWeightIdx, double *workspace) {
  int i;

  double *innerSum = workspace;
  double *innerSumFlipped = innerSum + sizeA*sizeB;

  double complex *distWeightsCompl = (double complex*) malloc(sizeof(double complex) *
                                                              numDistWeights);
  for (i = 0; i < numDistWeights; i++) {
    distWeightsCompl[i] = distWeights[i] + 0.0*I;
  }

  CalculatePairDelta(innerSum, cfgA, sizeA, cfgB, sizeB);
  ComputeInSumExp(innerSum, plaquetteAIdx, sizeA, plaquetteBIdx,
                  sizeB, plaquetteSize, distWeightsCompl, distWeightIdx);

  if (tRSym) {
    CalculatePairDeltaFlipped(innerSumFlipped, cfgA, sizeA, cfgB, sizeB);
    ComputeInSumExp(innerSumFlipped, plaquetteAIdx, sizeA,
                    plaquetteBIdx, sizeB, plaquetteSize, distWeightsCompl,
                    distWeightIdx);
  }
  free(distWeightsCompl);

  return ComputeExpKernel(sizeA, sizeB, tRSym, shift, startIdA, startIdB,
                          innerSum, innerSumFlipped);
}

void GPWExpKernelMat(const unsigned long *configsAUp,
                     const unsigned long *configsADown, const int *neighboursA,
                     const int sizeA, const int numA,
                     const unsigned long *configsBUp,
                     const unsigned long *configsBDown,
                     const int *neighboursB, const int sizeB, const int numB,
                     const double *distWeights, const int numDistWeights,
                     const int **distWeightIdx, const int dim, const int rC,
                     const int tRSym, const int shift, const int startIdA,
                     const int startIdB, const int symmetric, double *kernelMatr) {
  int i, j;
  int **cfgsA, **cfgsB;
  int plaquetteSize;
  int *plaquetteAIdx, *plaquetteBIdx, *distList;

  plaquetteSize = SetupPlaquetteIdx(rC, neighboursA, sizeA, neighboursB,
                                    sizeB, dim, &plaquetteAIdx,
                                    &plaquetteBIdx, &distList, 1);


  cfgsA = (int**) malloc(sizeof(int*) * numA);

  for (i = 0; i < numA; i++) {
    cfgsA[i] = (int*) malloc(sizeof(int) * 2 * sizeA);

    for(j = 0; j < sizeA; j++) {
      cfgsA[i][j] = (configsAUp[i] >> j) & 1;
      cfgsA[i][j+sizeA] = (configsADown[i] >> j) & 1;
    }
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
      double *workspace = (double*)malloc(sizeof(double)*(2*sizeA*sizeB));
      #pragma omp for
      for (i = 0; i < numA; i++) {
        for (j = 0; j < numB; j++) {
          kernelMatr[i*numB + j] = GPWExpKernel(cfgsA[i], plaquetteAIdx, sizeA,
                                                cfgsB[j], plaquetteBIdx, sizeB,
                                                tRSym, shift, startIdA, startIdB,
                                                plaquetteSize, distWeights,
                                                numDistWeights, distWeightIdx[j],
                                                workspace);
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
      double *workspace = (double*)malloc(sizeof(double)*(2*sizeA*sizeB));

      #pragma omp for
      for (i = 0; i < numA; i++) {
        for (j = 0; j <= i; j++) {
          kernelMatr[i*numA + j] = GPWExpKernel(cfgsA[i], plaquetteAIdx, sizeA,
                                                cfgsA[j], plaquetteAIdx, sizeA,
                                                tRSym, shift, startIdA, startIdB,
                                                plaquetteSize, distWeights,
                                                numDistWeights, distWeightIdx[j],
                                                workspace);
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
                     const int startIdB, double *kernelVec) {
  int i, j;
  int **cfgsA;

  int plaquetteSize;
  int *plaquetteAIdx, *plaquetteBIdx, *distList;

  plaquetteSize = SetupPlaquetteIdx(rC, neighboursA, sizeA, neighboursRef,
                                    sizeRef, dim, &plaquetteAIdx,
                                    &plaquetteBIdx, &distList, 1);
  cfgsA = (int**) malloc(sizeof(int*) * numA);

  for (i = 0; i < numA; i++) {
    cfgsA[i] = (int*) malloc(sizeof(int) * 2 * sizeA);

    for(j = 0; j < sizeA; j++) {
      cfgsA[i][j] = (configsAUp[i] >> j) & 1;
      cfgsA[i][j+sizeA] = (configsADown[i] >> j) & 1;
    }
  }

  #pragma omp parallel default(shared) private(i, j)
  {
    double *workspace = (double*)malloc(sizeof(double)*(2*sizeA*sizeRef));

    #pragma omp for
    for (i = 0; i < numA; i++) {
      kernelVec[i] = GPWExpKernel(cfgsA[i], plaquetteAIdx, sizeA, configRef,
                                  plaquetteBIdx, sizeRef, tRSym, shift,
                                  startIdA, startIdB, plaquetteSize, distWeights,
                                  numDistWeights, distWeightIdx, workspace);
    }
    free(workspace);
  }

  for (i = 0; i < numA; i++) {
    free(cfgsA[i]);
  }
  free(cfgsA);

  FreeMemPlaquetteIdx(plaquetteAIdx, plaquetteBIdx, distList);
}
