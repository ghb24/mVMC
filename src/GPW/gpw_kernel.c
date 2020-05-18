// TODO include license and description

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "gpw_kernel.h"

double GPWKernel1(const int *configA, const int sizeA, const int *configB,
                  const int sizeB, const int tRSym, const int shift,
                  const int startIdA, const int startIdB) {
  int countA[4] = {0, 0, 0, 0};
  int countB[4] = {0, 0, 0, 0};
  int up, down;
  int i;
  double kernel = 0.0;
  int shiftA = 1 + startIdA;
  int shiftB = 1 + startIdB;
  int translationA = 1;
  int translationB = 1;

  if (abs(shift) & 1) {
    shiftA = sizeA;
    if (shift < 0) {
      translationA = sizeB;
    }
  }
  if ((abs(shift) & 2) >> 1) {
    shiftB = sizeB;

    if (shift < 0) {
      translationB = sizeA;
    }
  }

  for (i = startIdA; i < shiftA; i+=translationA) {
    up = configA[i];
    down = configA[i+sizeA];

    countA[0] += up&(1-down);
    countA[1] += (1-up)&down;
    countA[2] += up&down;
    countA[3] += (1-up)&(1-down);
  }

  for (i = startIdB; i < shiftB; i+=translationB) {
    up = configB[i];
    down = configB[i+sizeB];

    countB[0] += up&(1-down);
    countB[1] += (1-up)&down;
    countB[2] += up&down;
    countB[3] += (1-up)&(1-down);
  }

  for (i = 0; i < 4; i++) {
    kernel += (double)(countA[i]*countB[i]);
  }

  // add kernel with one configuration flipped to ensure time reversal symmetry is respected
  if (tRSym) {
    kernel += (double)(countA[1]*countB[0] + countA[0]*countB[1] + countA[2]*countB[2] + countA[3]*countB[3]);
    kernel /= 2.0;
  }

  return kernel;
}

void GPWKernel1Mat(const unsigned long *configsAUp, const unsigned long *configsADown,
                   const int sizeA, const int numA, const unsigned long *configsBUp,
                   const unsigned long *configsBDown, const int sizeB, const int numB,
                   const int tRSym, const int shift, const int startIdA,
                   const int startIdB, const int symmetric, double *kernelMatr) {
  int i, j;
  int **cfgsA, **cfgsB;
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

    #pragma omp parallel for default(shared) private(i, j)
    for (i = 0; i < numA; i++) {
      for (j = 0; j < numB; j++) {
        kernelMatr[i*numB + j] = GPWKernel1(cfgsA[i], sizeA, cfgsB[j], sizeB,
                                            tRSym, shift, startIdA, startIdB);
      }
    }

    for (i = 0; i < numB; i++) {
     free(cfgsB[i]);
    }
    free(cfgsB);
  }

  else {
    #pragma omp parallel for default(shared) private(i, j)
    for (i = 0; i < numA; i++) {
      for (j = 0; j <= i; j++) {
        kernelMatr[i*numA + j] = GPWKernel1(cfgsA[i], sizeA, cfgsA[j], sizeA,
                                            tRSym, shift, startIdA, startIdB);
      }
    }
  }

  for (i = 0; i < numA; i++) {
   free(cfgsA[i]);
  }
  free(cfgsA);
}

void GPWKernel1Vec(const unsigned long *configsAUp, const unsigned long *configsADown,
                   const int sizeA, const int numA, const int *configRef,
                   const int sizeRef, const int tRSym, const int shift,
                   const int startIdA, const int startIdB, double *kernelVec) {
  int i, j;
  int **cfgsA;
  cfgsA = (int**) malloc(sizeof(int*) * numA);

  for (i = 0; i < numA; i++) {
    cfgsA[i] = (int*) malloc(sizeof(int) * 2 * sizeA);

    for(j = 0; j < sizeA; j++) {
      cfgsA[i][j] = (configsAUp[i] >> j) & 1;
      cfgsA[i][j+sizeA] = (configsADown[i] >> j) & 1;
    }
  }

  #pragma omp parallel for default(shared) private(i)
  for (i = 0; i < numA; i++) {
    kernelVec[i] = GPWKernel1(cfgsA[i], sizeA, configRef, sizeRef, tRSym,
                              shift, startIdA, startIdB);
  }

  for (i = 0; i < numA; i++) {
    free(cfgsA[i]);
  }

  free(cfgsA);
}

void CalculatePairDelta(double *inSum, const int *cfgA, const int sizeA,
                        const int *cfgB, const int sizeB) {
  int i, a;

  for (i = 0; i < sizeA; i++) {
    for (a = 0; a < sizeB; a++) {
      // TODO: can this be optimised by a bitwise comparison?
      if ((cfgA[i%sizeA]==cfgB[a%sizeB])&&
          (cfgA[i%sizeA+sizeA]==cfgB[a%sizeB+sizeB])) {
        inSum[i*sizeB+a] = copysign(inSum[i*sizeB+a], -1.0);
      }
      else {
        inSum[i*sizeB+a] = copysign(inSum[i*sizeB+a], 1.0);
      }
    }
  }
  return;
}

void CalculatePairDeltaFlipped(double *inSum, const int *cfgA, const int sizeA,
                               const int *cfgB, const int sizeB) {
  int i, a;

  for (i = 0; i < sizeA; i++) {
    for (a = 0; a < sizeB; a++) {
      // TODO: can this be optimised by a bitwise comparison?
      if ((cfgA[i%sizeA]==cfgB[a%sizeB+sizeB])&&
          (cfgA[i%sizeA+sizeA]==cfgB[a%sizeB])) {
        inSum[i*sizeB+a] = copysign(inSum[i*sizeB+a], -1.0);
      }
      else {
        inSum[i*sizeB+a] = copysign(inSum[i*sizeB+a], 1.0);
      }
    }
  }
  return;
}

int SetupPlaquetteIdx(const int rC, const int *neighboursA, const int sizeA,
                      const int *neighboursB, const int sizeB,
                      const int dim, int **plaquetteAIdx, int **plaquetteBIdx,
                      int **distList, int includeCentre) {
  int i, a, j, b, k, l, dist, count, tmpCount, d;

  int plaquetteSize, plaquetteCount, doubleCounting;
  int maxShellSize = 2;

  int *prevIndSys, *tmpPrevIndSys, *prevIndTrn, *tmpPrevIndTrn;
  int *visitedA, *visitedB;
  int *directions, *tmpDirections;

  plaquetteSize = 0;

  if (rC < 0) {
    if (abs(rC) < sizeA && abs(rC) < sizeB) {
      maxShellSize = abs(rC)-1;
      plaquetteSize = abs(rC)-1;
    }
    else if (sizeA < sizeB) {
      maxShellSize = sizeA-1;
      plaquetteSize = sizeA-1;
    }
    else {
      maxShellSize = sizeB-1;
      plaquetteSize = sizeB-1;
    }

    doubleCounting = 0;
  }

  else {
    if (dim == 2) {
      maxShellSize = 4*rC;
    }
    else if (dim == 3) {
      // TODO: double check if this is right
      maxShellSize = 4*rC*rC + 2;
    }

    if (dim == 1) {
      plaquetteSize = 2*rC;
    }
    else if (dim == 2) {
      for (i = 1; i <= rC; i++) {
        plaquetteSize += 4*i;
      }
    }
    else if (dim == 3) {
      // TODO: double check if this is right
      for (i = 1; i <= rC; i++) {
        plaquetteSize += (2+4*i*i);
      }
    }
    // double count contributions if plaquette size is determined from cutoff range
    doubleCounting = 1;
  }

  if (includeCentre) {
    maxShellSize++;
    plaquetteSize++;
  }

  *plaquetteAIdx = (int*)malloc((plaquetteSize*sizeA)*sizeof(int));
  *plaquetteBIdx = (int*)malloc((plaquetteSize*sizeB)*sizeof(int));
  *distList = (int*)malloc(plaquetteSize*sizeof(int));

  prevIndSys = (int*)malloc((4*maxShellSize+2*dim*maxShellSize + sizeA + sizeB)*sizeof(int));
  tmpPrevIndSys = prevIndSys + maxShellSize;
  prevIndTrn = tmpPrevIndSys + maxShellSize;
  tmpPrevIndTrn = prevIndTrn + maxShellSize;
  directions = tmpPrevIndTrn + maxShellSize;
  tmpDirections = directions + maxShellSize*dim;
  visitedA = tmpDirections + maxShellSize*dim;
  visitedB = visitedA + sizeA;

  for (i = 0; i < sizeA; i++) {
    for (a = 0; a < sizeB; a++) {
      for (k = 0; k < sizeA; k++) {
        visitedA[k] = 0;
      }

      for (k = 0; k < sizeB; k++) {
        visitedB[k] = 0;
      }

      // set up starting point (dist = 0)
      plaquetteCount = 0;
      count = 1;
      dist = 0;

      j = i;
      b = a;
      prevIndSys[0] = j;
      prevIndTrn[0] = b;
      visitedA[j] = 1;
      visitedB[b] = 1;

      for (d = 0; d < dim; d++) {
        directions[d] = 0;
      }

      if (includeCentre) {
        (*plaquetteAIdx)[i*plaquetteSize+plaquetteCount] = j;
        (*plaquetteBIdx)[a*plaquetteSize+plaquetteCount] = b;
        (*distList)[plaquetteCount] = dist;
        plaquetteCount++;
      }

      while (dist < abs(rC) || plaquetteCount < plaquetteSize) {
        if (plaquetteCount >= plaquetteSize) {
          break;
        }
        dist ++;
        tmpCount = 0;

        for (k = 0; k < count; k++) {
          if (plaquetteCount >= plaquetteSize) {
            break;
          }
          // add the corresponding neighbours
          for (d = 0; d < dim; d++) {

            // positive or negative direction
            int dir = directions[dim*k+d] >= 0 ? 0 : 1;
            // add neighbour in respective direction
            j = neighboursA[prevIndSys[k]*2*dim + d*2+dir];
            b = neighboursB[prevIndTrn[k]*2*dim + d*2+dir];

            if ((!visitedA[j] && !visitedB[b]) || doubleCounting) {
              visitedA[j] = 1;
              visitedB[b] = 1;
              tmpPrevIndSys[tmpCount] = j;
              tmpPrevIndTrn[tmpCount] = b;

              for(l = 0; l < dim; l++) {
                tmpDirections[dim*tmpCount+l] = directions[dim*k+l];
              }

              tmpDirections[dim*tmpCount+d] += (dir==0?1:-1);
              tmpCount++;

              (*plaquetteAIdx)[i*plaquetteSize+plaquetteCount] = j;
              (*plaquetteBIdx)[a*plaquetteSize+plaquetteCount] = b;
              (*distList)[plaquetteCount] = dist;
              plaquetteCount++;
            }

            if (plaquetteCount >= plaquetteSize) {
              break;
            }


            // add neighbour in opposing direction if coordinate = 0
            if (directions[dim*k+d] == 0) {
              dir = dir ^ 1;
              j = neighboursA[prevIndSys[k]*2*dim+d*2+dir];
              b = neighboursB[prevIndTrn[k]*2*dim+d*2+dir];

              if ((!visitedA[j] && !visitedB[b]) || doubleCounting) {
                visitedA[j] = 1;
                visitedB[b] = 1;
                tmpPrevIndSys[tmpCount] = j;
                tmpPrevIndTrn[tmpCount] = b;

                for (l = 0; l < dim; l++) {
                  tmpDirections[dim*tmpCount+l] = directions[k*dim+l];
                }

                tmpDirections[dim*tmpCount+d] += (dir==0?1:-1);
                tmpCount++;

                (*plaquetteAIdx)[i*plaquetteSize+plaquetteCount] = j;
                (*plaquetteBIdx)[a*plaquetteSize+plaquetteCount] = b;
                (*distList)[plaquetteCount] = dist;
                plaquetteCount++;
              }

              if (plaquetteCount >= plaquetteSize) {
                break;
              }
            }

            /* only continue in additional dimension if site in this dimension
               has coordinate 0 */
            else {
              break;
            }
          }
        }
        count = tmpCount;

        for (k = 0; k < tmpCount; k++) {
          prevIndSys[k] = tmpPrevIndSys[k];
          prevIndTrn[k] = tmpPrevIndTrn[k];

          for (l = 0; l < dim; l++) {
            directions[k*dim+l] = tmpDirections[k*dim+l];
          }
        }
      }
    }
  }

  free(prevIndSys);
  return plaquetteSize;
}

void FreeMemPlaquetteIdx(int *plaquetteAIdx, int *plaquetteBIdx,
                         int *distList) {
  free(distList);
  free(plaquetteBIdx);
  free(plaquetteAIdx);
}

void SetupPlaquetteHash(const int sysSize, const int plaqSize,
                        const int *plaqIdx, int ***plaqHash,
                        int **plaqHashSz) {
  int i, j, id;

  int *hashCount = (int*)(malloc(sizeof(int)*sysSize*sysSize));

  *plaqHash = (int**)(malloc(sizeof(int*)*sysSize*sysSize));
  *plaqHashSz = (int*)(malloc(sizeof(int)*sysSize*sysSize));

  for (i = 0; i < sysSize*sysSize; i++) {
    (*plaqHashSz)[i] = 0;
    hashCount[i] = 0;
  }

  for (i = 0; i < sysSize; i++) {
    for (j = 0; j < plaqSize; j++) {
      id = i*sysSize + plaqIdx[i*plaqSize + j];
      ((*plaqHashSz)[id])++;
    }
  }

  for (i = 0; i < sysSize*sysSize; i++) {
    (*plaqHash)[i] = (int*)(malloc(sizeof(int)*((*plaqHashSz)[i])));
  }

  for (i = 0; i < sysSize; i++) {
    for (j = 0; j < plaqSize; j++) {
      id = i*sysSize + plaqIdx[i*plaqSize + j];
      (*plaqHash)[id][hashCount[id]] = j;
      (hashCount[id])++;
    }
  }
  free(hashCount);
}

void FreeMemPlaquetteHash(const int sysSize, int **plaqHash, int *plaqHashSz) {
  int i;
  for (i = 0; i < sysSize*sysSize; i++) {
    free(plaqHash[i]);
  }

  free(plaqHashSz);
  free(plaqHash);
}

void ComputeInSum(double *inSum, const int *plaquetteAIdx,
                  const int sizeA, const int *plaquetteBIdx, const int sizeB,
                  const int plaquetteSize, const int *distList,
                  const int shift, const int startIdA, const int startIdB,
                  const double distWeightPower) {
  int i, a, k;
  double innerSum;

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

  for (i = startIdA; i < shiftSys; i+=translationSys) {
    for (a = startIdB; a < shiftTrn; a+=translationTrn) {
      innerSum = 0.0;
      for (k = 0; k < plaquetteSize; k++) {
        if (signbit(inSum[plaquetteAIdx[i*plaquetteSize+k]*sizeB +
                          plaquetteBIdx[a*plaquetteSize+k]])) {
          innerSum += 1.0/pow(distList[k], distWeightPower);
        }
      }
      inSum[i*sizeB + a] = copysign(innerSum, inSum[i*sizeB + a]);
    }
  }
}

void UpdateInSum(double *inSumNew, const double *inSumOld,
                 const int *plaquetteAIdx, const int sizeA,
                 const int *plaquetteBIdx, const int sizeB,
                 const int plaquetteSize, const int *distList,
                 const int shift, const int startIdA, const int startIdB,
                 const double distWeightPower, int **plaqHash,
                 int *plaqHashSz, const int siteA, const int siteB) {
  int i, a, k, countA, countB, id;
  const int *hashListA, *hashListB;

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

  for (i = startIdA; i < shiftSys; i+=translationSys) {
    countA = plaqHashSz[sizeA*i + siteA];
    countB = plaqHashSz[sizeA*i + siteB];
    hashListA = plaqHash[sizeA*i + siteA];
    hashListB = plaqHash[sizeA*i + siteB];

    for (a = startIdB; a < shiftTrn; a+=translationTrn) {
      for (k = 0; k < countA; k++) {
        id = hashListA[k];
        if (signbit(inSumNew[siteA*sizeB + plaquetteBIdx[a*plaquetteSize+id]])) {
          inSumNew[i*sizeB + a] = copysign(fabs(inSumNew[i*sizeB + a]) + 1.0/pow(distList[id], distWeightPower), inSumNew[i*sizeB + a]);
        }
        if (signbit(inSumOld[siteA*sizeB + plaquetteBIdx[a*plaquetteSize+id]])) {
          inSumNew[i*sizeB + a] = copysign(fabs(inSumNew[i*sizeB + a]) - 1.0/pow(distList[id], distWeightPower), inSumNew[i*sizeB + a]);
        }
      }
      if (siteA != siteB) {
        for (k = 0; k < countB; k++) {
          id = hashListB[k];
          if (signbit(inSumNew[siteB*sizeB + plaquetteBIdx[a*plaquetteSize+id]])) {
            inSumNew[i*sizeB + a] = copysign(fabs(inSumNew[i*sizeB + a]) + 1.0/pow(distList[id], distWeightPower), inSumNew[i*sizeB + a]);
          }
          if (signbit(inSumOld[siteB*sizeB + plaquetteBIdx[a*plaquetteSize+id]])) {
            inSumNew[i*sizeB + a] = copysign(fabs(inSumNew[i*sizeB + a]) - 1.0/pow(distList[id], distWeightPower), inSumNew[i*sizeB + a]);
          }
        }
      }
    }
  }
}


void UpdateDelta(double *inSumNew, const int *cfgA, const int sizeA,
                 const int *cfgB, const int sizeB, const int siteA,
                 const int siteB) {
  int a;

  for (a = 0; a < sizeB; a++) {
    if ((cfgA[siteA]==cfgB[a])&&
        (cfgA[siteA+sizeA]==cfgB[a+sizeB])) {
      inSumNew[siteA*sizeB+a] = copysign(inSumNew[siteA*sizeB+a], -1.0);
    }
    else {
      inSumNew[siteA*sizeB+a] = copysign(inSumNew[siteA*sizeB+a], 1.0);
    }

    if ((cfgA[siteB]==cfgB[a])&&
        (cfgA[siteB+sizeA]==cfgB[a+sizeB])) {
      inSumNew[siteB*sizeB+a] = copysign(inSumNew[siteB*sizeB+a], -1.0);
    }
    else {
      inSumNew[siteB*sizeB+a] = copysign(inSumNew[siteB*sizeB+a], 1.0);
    }
  }
}

void UpdateDeltaFlipped(double *inSumNew, const int *cfgA, const int sizeA,
                        const int *cfgB, const int sizeB, const int siteA,
                        const int siteB) {
  int a;

  for (a = 0; a < sizeB; a++) {
    if ((cfgA[siteA]==cfgB[a+sizeB])&&
        (cfgA[siteA+sizeA]==cfgB[a])) {
      inSumNew[siteA*sizeB+a] = copysign(inSumNew[siteA*sizeB+a], -1.0);
    }
    else {
      inSumNew[siteA*sizeB+a] = copysign(inSumNew[siteA*sizeB+a], 1.0);
    }

    if ((cfgA[siteB]==cfgB[a+sizeB])&&
        (cfgA[siteB+sizeA]==cfgB[a])) {
      inSumNew[siteB*sizeB+a] = copysign(inSumNew[siteB*sizeB+a], -1.0);
    }
    else {
      inSumNew[siteB*sizeB+a] = copysign(inSumNew[siteB*sizeB+a], 1.0);
    }
  }
}

double ComputeKernel(const int sizeA, const int sizeB, const double power,
                     const double theta, const double norm, const int tRSym,
                     const int shift, const int startIdA, const int startIdB,
                     const double *inSum, const double *inSumFlipped) {
  int i, a;
  int shiftSys = 1 + startIdA;
  int shiftTrn = 1 + startIdB;
  int translationSys = 1;
  int translationTrn = 1;
  double kernel = 0.0;
  const double scaledNorm = theta + norm/power;

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

  // looks ugly but we want speed here
  if (tRSym) {
    if (power < 0.0) {
      for (i = startIdA; i < shiftSys; i+=translationSys) {
        for (a = startIdB; a < shiftTrn; a+=translationTrn) {
          if (signbit(inSum[i*sizeB+a])) {
            kernel += exp(-1.0/theta * (norm - fabs(inSum[i*sizeB+a])));
          }
          if (signbit(inSumFlipped[i*sizeB+a])) {
            kernel += exp(-1.0/theta * (norm - fabs(inSumFlipped[i*sizeB+a])));
          }
        }
      }
    }
    else {
      for (i = startIdA; i < shiftSys; i+=translationSys) {
        for (a = startIdB; a < shiftTrn; a+=translationTrn) {
          if (signbit(inSum[i*sizeB+a])) {
            kernel += pow(((theta + fabs(inSum[i*sizeB+a])/power)/scaledNorm), power);
          }
          if (signbit(inSumFlipped[i*sizeB+a])) {
            kernel += pow(((theta + fabs(inSumFlipped[i*sizeB+a])/power)/scaledNorm), power);
          }
        }
      }
    }
    kernel /= 2.0;
  }
  else {
    if (power < 0.0) {
      for (i = startIdA; i < shiftSys; i+=translationSys) {
        for (a = startIdB; a < shiftTrn; a+=translationTrn) {
          if (signbit(inSum[i*sizeB+a])) {
            kernel += exp(-1.0/theta * (norm - fabs(inSum[i*sizeB+a])));
          }
        }
      }
    }
    else {
      for (i = startIdA; i < shiftSys; i+=translationSys) {
        for (a = startIdB; a < shiftTrn; a+=translationTrn) {
          if (signbit(inSum[i*sizeB+a])) {
            kernel += pow(((theta + fabs(inSum[i*sizeB+a])/power)/scaledNorm), power);
          }
        }
      }
    }
  }
  return kernel;
}

double ComputeKernDeriv(const int sizeA, const int sizeB, const double power,
                        const double theta, const double norm, const int tRSym,
                        const int shift, const int startIdA, const int startIdB,
                        const double *inSum, const double *inSumFlipped) {
  int i, a;
  int shiftSys = 1 + startIdA;
  int shiftTrn = 1 + startIdB;
  int translationSys = 1;
  int translationTrn = 1;
  const double scaledNorm = theta + norm/power;

  double kernDeriv = 0.0;

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

  if (power < 0.0) {
    for (i = startIdA; i < shiftSys; i+=translationSys) {
      for (a = startIdB; a < shiftTrn; a+=translationTrn) {
        if (signbit(inSum[i*sizeB+a])) {
          kernDeriv += exp(-1.0/theta * (norm - fabs(inSum[i*sizeB+a])))*
                       (norm-fabs(inSum[i*sizeB+a]));
        }
      }
    }
    if (tRSym) {
      for (i = startIdA; i < shiftSys; i+=translationSys) {
        for (a = startIdB; a < shiftTrn; a+=translationTrn) {
          if (signbit(inSumFlipped[i*sizeB+a])) {
            kernDeriv += exp(-1.0/theta * (norm - fabs(inSumFlipped[i*sizeB+a])))*
                         (norm-fabs(inSumFlipped[i*sizeB+a]));
          }
        }
      }
    kernDeriv /= 2.0;
    }
    kernDeriv /= (theta * theta);
  }
  else {
    for (i = startIdA; i < shiftSys; i+=translationSys) {
      for (a = startIdB; a < shiftTrn; a+=translationTrn) {
        if (signbit(inSum[i*sizeB+a])) {
          kernDeriv += pow(((theta + (fabs(inSum[i*sizeB+a])/power))/scaledNorm), power-1)*
                       ((norm - fabs(inSum[i*sizeB+a]))/power);
        }
      }
    }
    if (tRSym) {
      for (i = startIdA; i < shiftSys; i+=translationSys) {
        for (a = startIdB; a < shiftTrn; a+=translationTrn) {
          if (signbit(inSumFlipped[i*sizeB+a])) {
            kernDeriv += pow(((theta + (fabs(inSumFlipped[i*sizeB+a])/power))/scaledNorm), power-1)*
                         ((norm - fabs(inSumFlipped[i*sizeB+a]))/power);
          }
        }
      }
    kernDeriv /= 2.0;
    }
    kernDeriv *= power/(scaledNorm * scaledNorm);
  }
  return kernDeriv;
}

double ComputeKernelN(const int *cfgA, const int *plaquetteAIdx, const int sizeA,
                      const int *cfgB, const int *plaquetteBIdx, const int sizeB,
                      const int n, const int tRSym, const int shift,
                      const int startIdA, const int startIdB, const double *inSum,
                      const double *inSumFlipped) {
  if (n == 1) {
   return GPWKernel1(cfgA, sizeA, cfgB, sizeB, tRSym, shift, startIdA, startIdB);
  }

  else {
    int i, a, k, plaquetteMatches;

    int plaquetteSize = n-1;

    double kernel = 0.0;

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

    /* add kernel with one configuration flipped to ensure time reversal
       symmetry is respected */

    if (tRSym) {
      for (i = startIdA; i < shiftSys; i+=translationSys) {
        for (a = startIdB; a < shiftTrn; a+=translationTrn) {
          if (signbit(inSum[i*sizeB+a])) {
            plaquetteMatches = 1;
            for (k = 0; k < plaquetteSize; k++) {
              if (!signbit(inSum[plaquetteAIdx[i*plaquetteSize+k]*sizeB +
                                 plaquetteBIdx[a*plaquetteSize+k]])) {
                plaquetteMatches = 0;
                break;
              }
            }
            if (plaquetteMatches) {
              kernel += 1.0;
            }
          }
          if (signbit(inSumFlipped[i*sizeB+a])) {
            plaquetteMatches = 1;
            for (k = 0; k < plaquetteSize; k++) {
              if (!signbit(inSumFlipped[plaquetteAIdx[i*plaquetteSize+k]*sizeB +
                                        plaquetteBIdx[a*plaquetteSize+k]])) {
                plaquetteMatches = 0;
                break;
              }
            }
            if (plaquetteMatches) {
              kernel += 1.0;
            }
          }
        }
      }

      kernel /= 2.0;
    }
    else {
      for (i = startIdA; i < shiftSys; i+=translationSys) {
        for (a = startIdB; a < shiftTrn; a+=translationTrn) {
          if (signbit(inSum[i*sizeB+a])) {
            plaquetteMatches = 1;
            for (k = 0; k < plaquetteSize; k++) {
              if (!signbit(inSum[plaquetteAIdx[i*plaquetteSize+k]*sizeB +
                                 plaquetteBIdx[a*plaquetteSize+k]])) {
                plaquetteMatches = 0;
                break;
              }
            }
            if (plaquetteMatches) {
              kernel += 1.0;
            }
          }
        }
      }
    }
    return kernel;
  }
}

double GPWKernelN(const int *cfgA, const int *plaquetteAIdx, const int sizeA,
                  const int *cfgB, const int *plaquetteBIdx, const int sizeB,
                  const int dim, const int n, const int tRSym, const int shift,
                  const int startIdA, const int startIdB, double *workspace) {
  if (n == 1) {
   return GPWKernel1(cfgA, sizeA, cfgB, sizeB, tRSym, shift, startIdA, startIdB);
  }
  else {
    double *inSum = workspace;
    double *inSumFlipped = inSum + sizeA*sizeB;

    CalculatePairDelta(inSum, cfgA, sizeA, cfgB, sizeB);

    if (tRSym) {
      CalculatePairDeltaFlipped(inSumFlipped, cfgA, sizeA, cfgB, sizeB);
    }

    return ComputeKernelN(cfgA, plaquetteAIdx, sizeA, cfgB, plaquetteBIdx,
                          sizeB, n, tRSym, shift, startIdA, startIdB, inSum,
                          inSumFlipped);
  }
}

void GPWKernelNMat(const unsigned long *configsAUp,
                   const unsigned long *configsADown, const int *neighboursA,
                   const int sizeA, const int numA,
                   const unsigned long *configsBUp,
                   const unsigned long *configsBDown, const int *neighboursB,
                   const int sizeB, const int numB, const int dim, const int n,
                   const int tRSym, const int shift, const int startIdA,
                   const int startIdB, const int symmetric,
                   double *kernelMatr) {

  int i, j;
  int **cfgsA, **cfgsB;
  int *plaquetteAIdx, *plaquetteBIdx, *distList;

  SetupPlaquetteIdx(-n, neighboursA, sizeA, neighboursB, sizeB, dim,
                    &plaquetteAIdx, &plaquetteBIdx, &distList, 0);


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
          kernelMatr[i*numB + j] = GPWKernelN(cfgsA[i], plaquetteAIdx, sizeA,
                                              cfgsB[j], plaquetteBIdx, sizeB,
                                              dim, n, tRSym, shift, startIdA,
                                              startIdB, workspace);
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
          kernelMatr[i*numA + j] = GPWKernelN(cfgsA[i], plaquetteAIdx, sizeA,
                                              cfgsA[j], plaquetteAIdx, sizeA,
                                              dim, n, tRSym, shift, startIdA,
                                              startIdB, workspace);
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

void GPWKernelNVec(const unsigned long *configsAUp, const unsigned long *configsADown,
                   const int *neighboursA, const int sizeA, const int numA,
                   const int *configRef, const int *neighboursRef,
                   const int sizeRef, const int dim, const int n,
                   const int tRSym, const int shift, const int startIdA,
                   const int startIdB, double *kernelVec) {
  int i, j;
  int **cfgsA;
  int *plaquetteAIdx, *plaquetteBIdx, *distList;

  SetupPlaquetteIdx(-n, neighboursA, sizeA, neighboursRef, sizeRef, dim,
                    &plaquetteAIdx, &plaquetteBIdx, &distList, 0);

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
      kernelVec[i] = GPWKernelN(cfgsA[i], plaquetteAIdx, sizeA, configRef,
                                plaquetteBIdx, sizeRef, dim, n, tRSym, shift,
                                startIdA, startIdB, workspace);
    }
    free(workspace);
  }


  for (i = 0; i < numA; i++) {
    free(cfgsA[i]);
  }
  free(cfgsA);

  FreeMemPlaquetteIdx(plaquetteAIdx, plaquetteBIdx, distList);
}


double GPWKernel(const int *cfgA, const int *plaquetteAIdx, const int sizeA,
                 const int *cfgB, const int *plaquetteBIdx, const int sizeB,
                 const double power, const double theta, const double distWeightPower,
                 const int tRSym, const int shift, const int startIdA,
                 const int startIdB, const int plaquetteSize, const int *distList,
                 double *workspace) {
  int i, distWeightFlag;

  double *innerSum = workspace;
  double *innerSumFlipped = innerSum + sizeA*sizeB;

  double norm = 0.0;

  for (i = 0; i < plaquetteSize; i++) {
    norm += 1.0/pow(distList[i], distWeightPower);
  }

  CalculatePairDelta(innerSum, cfgA, sizeA, cfgB, sizeB);
  ComputeInSum(innerSum, plaquetteAIdx, sizeA, plaquetteBIdx,
               sizeB, plaquetteSize, distList, shift,
               startIdA, startIdB, distWeightPower);

  if (tRSym) {
    CalculatePairDeltaFlipped(innerSumFlipped, cfgA, sizeA, cfgB, sizeB);
    ComputeInSum(innerSumFlipped, plaquetteAIdx, sizeA,
                 plaquetteBIdx, sizeB, plaquetteSize, distList, shift,
                 startIdA, startIdB, distWeightPower);
  }

  return ComputeKernel(sizeA, sizeB, power, theta, norm, tRSym, shift,
                       startIdA, startIdB, innerSum, innerSumFlipped);
}

void GPWKernelMat(const unsigned long *configsAUp,
                  const unsigned long *configsADown, const int *neighboursA,
                  const int sizeA, const int numA,
                  const unsigned long *configsBUp,
                  const unsigned long *configsBDown,
                  const int *neighboursB, const int sizeB, const int numB,
                  const int dim, const double power, const int rC,
                  const double theta, const double distWeightPower,
                  const int tRSym, const int shift, const int startIdA,
                  const int startIdB, const int symmetric,
                  double *kernelMatr) {
  int i, j;
  int **cfgsA, **cfgsB;
  int plaquetteSize;
  int *plaquetteAIdx, *plaquetteBIdx, *distList;

  plaquetteSize = SetupPlaquetteIdx(rC, neighboursA, sizeA, neighboursB,
                                    sizeB, dim, &plaquetteAIdx,
                                    &plaquetteBIdx, &distList, 0);


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
          kernelMatr[i*numB + j] = GPWKernel(cfgsA[i], plaquetteAIdx, sizeA,
                                             cfgsB[j], plaquetteBIdx, sizeB,
                                             power, theta, distWeightPower, tRSym, shift,
                                             startIdA, startIdB, plaquetteSize,
                                             distList, workspace);
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
          kernelMatr[i*numA + j] = GPWKernel(cfgsA[i], plaquetteAIdx, sizeA,
                                             cfgsA[j], plaquetteAIdx, sizeA,
                                             power, theta, distWeightPower, tRSym, shift,
                                             startIdA, startIdB,
                                             plaquetteSize, distList,
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

void GPWKernelVec(const unsigned long *configsAUp,
                  const unsigned long *configsADown, const int *neighboursA,
                  const int sizeA, const int numA, const int *configRef,
                  const int *neighboursRef, const int sizeRef, const int dim,
                  const double power, const int rC, const double theta,
                  const double distWeightPower, const int tRSym, const int shift,
                  const int startIdA, const int startIdB, double *kernelVec) {
  int i, j;
  int **cfgsA;

  int plaquetteSize;
  int *plaquetteAIdx, *plaquetteBIdx, *distList;

  plaquetteSize = SetupPlaquetteIdx(rC, neighboursA, sizeA, neighboursRef,
                                    sizeRef, dim, &plaquetteAIdx,
                                    &plaquetteBIdx, &distList, 0);
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
      kernelVec[i] = GPWKernel(cfgsA[i], plaquetteAIdx, sizeA, configRef,
                               plaquetteBIdx, sizeRef, power, theta, distWeightPower, tRSym,
                               shift, startIdA, startIdB, plaquetteSize,
                               distList, workspace);
    }
    free(workspace);
  }

  for (i = 0; i < numA; i++) {
    free(cfgsA[i]);
  }
  free(cfgsA);

  FreeMemPlaquetteIdx(plaquetteAIdx, plaquetteBIdx, distList);
}
