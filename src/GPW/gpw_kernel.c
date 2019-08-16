// TODO include license and description

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "gpw_kernel.h"

double GPWKernel1(const int *configA, const int sizeA, const int *configB,
                  const int sizeB, const int tRSym, const int shift) {
  int countA[4] = {0, 0, 0, 0};
  int countB[4] = {0, 0, 0, 0};
  int up, down;
  int i;
  double kernel = 0.0;
  int shiftA = 1;
  int shiftB = 1;
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

  for (i = 0; i < shiftA; i+=translationA) {
    up = configA[i];
    down = configA[i+sizeA];

    countA[0] += up&(1-down);
    countA[1] += (1-up)&down;
    countA[2] += up&down;
    countA[3] += (1-up)&(1-down);
  }

  for (i = 0; i < shiftB; i+=translationB) {
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
                   const int tRSym, const int shift, const int symmetric,
                   double *kernelMatr) {
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
                                            tRSym, shift);
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
                                            tRSym, shift);
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
                   double *kernelVec) {
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
                              shift);
  }

  for (i = 0; i < numA; i++) {
    free(cfgsA[i]);
  }

  free(cfgsA);
}

// TODO: only compute necessary elements if no translational symmetry is required
void CalculatePairDelta(int *delta, const int *cfgA, const int sizeA,
                        const int *cfgB, const int sizeB) {
  int i, a;

  for (i = 0; i < sizeA; i++) {
    for (a = 0; a < sizeB; a++) {
      // TODO: can this be optimised by a bitwise comparison?
      if ((cfgA[i%sizeA]==cfgB[a%sizeB])&&
          (cfgA[i%sizeA+sizeA]==cfgB[a%sizeB+sizeB])) {
        delta[i*sizeB+a] = 1;
      }
      else {
        delta[i*sizeB+a] = 0;
      }
    }
  }
  return;
}

int SetupPlaquetteIdx(const int rC, const int *neighboursA, const int sizeA,
                      const int *neighboursB, const int sizeB,
                      const int dim, int **plaquetteAIdx, int **plaquetteBIdx,
                      int **distList) {
  int i, a, j, b, k, l, dist, count, tmpCount, d;

  int plaquetteSize, plaquetteCount;
  int maxShellSize = 2;

  int *prevIndSys, *tmpPrevIndSys, *prevIndTrn, *tmpPrevIndTrn;
  int *directions, *tmpDirections;

  plaquetteSize = 0;

  if (rC < 0) {
    maxShellSize = abs(rC)-1;
    plaquetteSize = abs(rC)-1;
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
  }


  *plaquetteAIdx = (int*)malloc((plaquetteSize*sizeA)*sizeof(int));
  *plaquetteBIdx = (int*)malloc((plaquetteSize*sizeB)*sizeof(int));
  *distList = (int*)malloc(plaquetteSize*sizeof(int));

  prevIndSys = (int*)malloc((4*maxShellSize+2*dim*maxShellSize)*sizeof(int));
  tmpPrevIndSys = prevIndSys + maxShellSize;
  prevIndTrn = tmpPrevIndSys + maxShellSize;
  tmpPrevIndTrn = prevIndTrn + maxShellSize;
  directions = tmpPrevIndTrn + maxShellSize;
  tmpDirections = directions + maxShellSize*dim;

  for (i = 0; i < sizeA; i++) {
    for (a = 0; a < sizeB; a++) {
      // set up starting point (dist = 0)
      plaquetteCount = 0;
      count = 1;
      dist = 0;

      j = i;
      b = a;
      prevIndSys[0] = j;
      prevIndTrn[0] = b;

      for (d = 0; d < dim; d++) {
        directions[d] = 0;
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

            tmpPrevIndSys[tmpCount] = j;
            tmpPrevIndTrn[tmpCount] = b;

            for(l = 0; l < dim; l++) {
              tmpDirections[dim*tmpCount+l] = directions[dim*k+l];
            }

            tmpDirections[dim*tmpCount+d] += (dir==0?1:-1);
            tmpCount ++;

            (*plaquetteAIdx)[i*plaquetteSize+plaquetteCount] = j;
            (*plaquetteBIdx)[a*plaquetteSize+plaquetteCount] = b;
            (*distList)[plaquetteCount] = dist;
            plaquetteCount += 1;

            if (plaquetteCount >= plaquetteSize) {
              break;
            }


            // add neighbour in opposing direction if coordinate = 0
            if (directions[dim*k+d] == 0) {
              dir = dir ^ 1;
              j = neighboursA[prevIndSys[k]*2*dim+d*2+dir];
              b = neighboursB[prevIndTrn[k]*2*dim+d*2+dir];

              tmpPrevIndSys[tmpCount] = j;
              tmpPrevIndTrn[tmpCount] = b;

              for (l = 0; l < dim; l++) {
                tmpDirections[dim*tmpCount+l] = directions[k*dim+l];
              }

              tmpDirections[dim*tmpCount+d] += (dir==0?1:-1);
              tmpCount ++;

              (*plaquetteAIdx)[i*plaquetteSize+plaquetteCount] = j;
              (*plaquetteBIdx)[a*plaquetteSize+plaquetteCount] = b;
              (*distList)[plaquetteCount] = dist;
              plaquetteCount += 1;

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

double GPWKernelN(const int *cfgA, const int *plaquetteAIdx, const int sizeA,
                  const int *cfgB, const int *plaquetteBIdx, const int sizeB,
                  const int dim, const int n, const int tRSym, const int shift,
                  int *workspace) {
  if (n == 1) {
   return GPWKernel1(cfgA, sizeA, cfgB, sizeB, tRSym, shift);
  }
  else {
    int i, a, k, j, b, plaquetteMatches;

    int plaquetteSize = n-1;

    double kernel = 0.0;

    int shiftSys = 1;
    int shiftTrn = 1;
    int translationSys = 1;
    int translationTrn = 1;

    int *delta = workspace;
    int *deltaFlipped = delta + sizeB*sizeA;
    int *configFlipped = deltaFlipped + sizeB*sizeA;

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

    CalculatePairDelta(delta, cfgA, sizeA, cfgB, sizeB);

    
    /* add kernel with one configuration flipped to ensure time reversal
       symmetry is respected */

    if (tRSym) {
      // TODO can this be done more efficiently?
      if (sizeB <= sizeA) {
        for (i = 0; i < sizeB; i++) {
          configFlipped[i] = cfgB[i+sizeB];
          configFlipped[i+sizeB] = cfgB[i];
        }
        CalculatePairDelta(deltaFlipped, cfgA, sizeA, configFlipped, sizeB);
      }
      else {
        for (i = 0; i < sizeA; i++) {
          configFlipped[i] = cfgA[i+sizeA];
          configFlipped[i+sizeA] = cfgA[i];
        }
        CalculatePairDelta(deltaFlipped, configFlipped, sizeA, cfgB, sizeB);
      }

      for (i = 0; i < shiftSys; i+=translationSys) {
        for (a = 0; a < shiftTrn; a+=translationTrn) {
          if (delta[i*sizeB+a]) {
            plaquetteMatches = 1;
            for (k = 0; k < plaquetteSize; k++) {
              if (!delta[plaquetteAIdx[i*plaquetteSize+k]*sizeB +
                         plaquetteBIdx[a*plaquetteSize+k]]) {
                plaquetteMatches = 0;
                break;
              }
            }
            if (plaquetteMatches) {
              kernel += 1.0;
            }
          }
          if (deltaFlipped[i*sizeB+a]) {
            plaquetteMatches = 1;
            for (k = 0; k < plaquetteSize; k++) {
              if (!deltaFlipped[plaquetteAIdx[i*plaquetteSize+k]*sizeB +
                                plaquetteBIdx[a*plaquetteSize+k]]) {
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
      for (i = 0; i < shiftSys; i+=translationSys) {
        for (a = 0; a < shiftTrn; a+=translationTrn) {
          if (delta[i*sizeB+a]) {
            plaquetteMatches = 1;
            for (k = 0; k < plaquetteSize; k++) {
              if (!delta[plaquetteAIdx[i*plaquetteSize+k]*sizeB +
                         plaquetteBIdx[a*plaquetteSize+k]]) {
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

void GPWKernelNMat(const unsigned long *configsAUp,
                   const unsigned long *configsADown, const int *neighboursA,
                   const int sizeA, const int numA,
                   const unsigned long *configsBUp,
                   const unsigned long *configsBDown, const int *neighboursB,
                   const int sizeB, const int numB, const int dim, const int n,
                   const int tRSym, const int shift, const int symmetric,
                   double *kernelMatr) {

  int i, j;
  int **cfgsA, **cfgsB;
  int *plaquetteAIdx, *plaquetteBIdx, *distList;

  SetupPlaquetteIdx(-n, neighboursA, sizeA, neighboursB, sizeB, dim,
                    &plaquetteAIdx, &plaquetteBIdx, &distList);


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
      int sizeSmall = sizeA;
      int *workspace;

      if (sizeB < sizeSmall) {
        sizeSmall = sizeB;
      }
      workspace = (int*)malloc(sizeof(int)*(2*sizeA*sizeB+2*sizeSmall));

      #pragma omp for
      for (i = 0; i < numA; i++) {
        for (j = 0; j < numB; j++) {
          kernelMatr[i*numB + j] = GPWKernelN(cfgsA[i], plaquetteAIdx, sizeA,
                                              cfgsB[j], plaquetteBIdx, sizeB,
                                              dim, n, tRSym, shift, workspace);
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
      int sizeSmall = sizeA;
      int *workspace;

      if (sizeB < sizeSmall) {
        sizeSmall = sizeB;
      }
      workspace = (int*)malloc(sizeof(int)*(2*sizeA*sizeB+2*sizeSmall));

      #pragma omp for
      for (i = 0; i < numA; i++) {
        for (j = 0; j <= i; j++) {
          kernelMatr[i*numA + j] = GPWKernelN(cfgsA[i], plaquetteAIdx, sizeA,
                                              cfgsA[j], plaquetteAIdx, sizeA,
                                              dim, n, tRSym, shift, workspace);
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
                   const int tRSym, const int shift, double *kernelVec){
  int i, j;
  int **cfgsA;
  int *plaquetteAIdx, *plaquetteBIdx, *distList;

  SetupPlaquetteIdx(-n, neighboursA, sizeA, neighboursRef, sizeRef, dim,
                    &plaquetteAIdx, &plaquetteBIdx, &distList);

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
    int sizeSmall = sizeA;
    int *workspace;

    if (sizeRef < sizeSmall) {
      sizeSmall = sizeRef;
    }
    workspace = (int*)malloc(sizeof(int)*(2*sizeA*sizeRef+2*sizeSmall));

    #pragma omp for
    for (i = 0; i < numA; i++) {
      kernelVec[i] = GPWKernelN(cfgsA[i], plaquetteAIdx, sizeA, configRef,
                                plaquetteBIdx, sizeRef, dim, n, tRSym, shift,
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



double GPWKernel(const int *cfgA, const int *plaquetteAIdx, const int sizeA,
                 const int *cfgB, const int *plaquetteBIdx, const int sizeB,
                 const int dim, const int power, const double theta0,
                 const double thetaC, const int tRSym, const int shift,
                 const int plaquetteSize, const int *distList,
                 int *workspace) {
  int i, a, k, j, b;

  double kernel = 0.0;
  int *delta = workspace;
  int *deltaFlipped = delta + sizeB*sizeA;
  int *configFlipped = deltaFlipped + sizeB*sizeA;

  int shiftSys = 1;
  int shiftTrn = 1;
  int translationSys = 1;
  int translationTrn = 1;

  double innerSum;
  double norm = 1.0;

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

  // compute norm TODO: do this in a preprocessing step to speed up things
  for (k = 0; k < plaquetteSize; k++) {
    norm += 1.0/distList[k];
  }
  if (power > 0) {
    norm = (theta0 + 1.0/(power)*norm);
  }

  CalculatePairDelta(delta, cfgA, sizeA, cfgB, sizeB);

  // looks ugly but we want speed here
  if (tRSym) {
    if (sizeB <= sizeA) {
      for (i = 0; i < sizeB; i++) {
        configFlipped[i] = cfgB[i+sizeB];
        configFlipped[i+sizeB] = cfgB[i];
      }
      CalculatePairDelta(deltaFlipped, cfgA, sizeA, configFlipped, sizeB);
    }
    else {
      for (i = 0; i < sizeA; i++) {
        configFlipped[i] = cfgA[i+sizeA];
        configFlipped[i+sizeA] = cfgA[i];
      }
      CalculatePairDelta(deltaFlipped, configFlipped, sizeA, cfgB, sizeB);
    }

    if (power == -1) {
      for (i = 0; i < shiftSys; i+=translationSys) {
        for (a = 0; a < shiftTrn; a+=translationTrn) {
          if (delta[i*sizeB+a]) {
            innerSum = 1.0;
            for (k = 0; k < plaquetteSize; k++) {
              if (delta[plaquetteAIdx[i*plaquetteSize+k]*sizeB +
                        plaquetteBIdx[a*plaquetteSize+k]]) {
                innerSum += 1.0/distList[k];
              }
            }
            kernel += exp(-1.0/theta0 * (norm - innerSum));
          }
          if (deltaFlipped[i*sizeB+a]) {
            innerSum = 1.0;
            for (k = 0; k < plaquetteSize; k++) {
              if (deltaFlipped[plaquetteAIdx[i*plaquetteSize+k]*sizeB +
                               plaquetteBIdx[a*plaquetteSize+k]]) {
                innerSum += 1.0/distList[k];
              }
            }
            kernel += exp(-1.0/theta0 * (norm - innerSum));
          }
        }
      }
    }
    else {
      for (i = 0; i < shiftSys; i+=translationSys) {
        for (a = 0; a < shiftTrn; a+=translationTrn) {
          if (delta[i*sizeB+a]) {
            innerSum = 1.0;
            for (k = 0; k < plaquetteSize; k++) {
              if (delta[plaquetteAIdx[i*plaquetteSize+k]*sizeB +
                        plaquetteBIdx[a*plaquetteSize+k]]) {
                innerSum += 1.0/distList[k];
              }
            }
            kernel += pow(((theta0 + 1.0/(power) * innerSum)/norm), power);
          }
          if (deltaFlipped[i*sizeB+a]) {
            innerSum = 1.0;
            for (k = 0; k < plaquetteSize; k++) {
              if (deltaFlipped[plaquetteAIdx[i*plaquetteSize+k]*sizeB +
                               plaquetteBIdx[a*plaquetteSize+k]]) {
                innerSum += 1.0/distList[k];
              }
            }
            kernel += pow(((theta0 + 1.0/(power) * innerSum)/norm), power);
          }
        }
      }
    }
    kernel /= 2.0;
  }
  else {
    if (power == -1) {
      for (i = 0; i < shiftSys; i+=translationSys) {
        for (a = 0; a < shiftTrn; a+=translationTrn) {
          if (delta[i*sizeB+a]) {
            innerSum = 1.0;
            for (k = 0; k < plaquetteSize; k++) {
              if (delta[plaquetteAIdx[i*plaquetteSize+k]*sizeB +
                        plaquetteBIdx[a*plaquetteSize+k]]) {
                innerSum += 1.0/distList[k];
              }
            }
            kernel += exp(-1.0/theta0 * (norm - innerSum));
          }
        }
      }
    }
    else {
      for (i = 0; i < shiftSys; i+=translationSys) {
        for (a = 0; a < shiftTrn; a+=translationTrn) {
          if (delta[i*sizeB+a]) {
            innerSum = 1.0;
            for (k = 0; k < plaquetteSize; k++) {
              if (delta[plaquetteAIdx[i*plaquetteSize+k]*sizeB +
                        plaquetteBIdx[a*plaquetteSize+k]]) {
                innerSum += 1.0/distList[k];
              }
            }
            kernel += pow(((theta0 + 1.0/(power) * innerSum)/norm), power);
          }
        }
      }
    }
  }
  return kernel;
}

void GPWKernelMat(const unsigned long *configsAUp,
                  const unsigned long *configsADown, const int *neighboursA,
                  const int sizeA, const int numA,
                  const unsigned long *configsBUp,
                  const unsigned long *configsBDown,
                  const int *neighboursB, const int sizeB, const int numB,
                  const int dim, const int power, const int rC,
                  const double theta0, const double thetaC,
                  const int tRSym, const int shift, const int symmetric,
                  double *kernelMatr) {
  int i, j;
  int **cfgsA, **cfgsB;
  int plaquetteSize;
  int *plaquetteAIdx, *plaquetteBIdx, *distList;

  plaquetteSize = SetupPlaquetteIdx(rC, neighboursA, sizeA, neighboursB,
                                    sizeB, dim, &plaquetteAIdx,
                                    &plaquetteBIdx, &distList);


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
      int sizeSmall = sizeA;
      int *workspace;

      if (sizeB < sizeSmall) {
        sizeSmall = sizeB;
      }
      workspace = (int*)malloc(sizeof(int)*(2*sizeA*sizeB+2*sizeSmall));
      #pragma omp for
      for (i = 0; i < numA; i++) {
        for (j = 0; j < numB; j++) {
          kernelMatr[i*numB + j] = GPWKernel(cfgsA[i], plaquetteAIdx, sizeA,
                                            cfgsB[j], plaquetteBIdx, sizeB,
                                            dim, power, theta0, thetaC, tRSym,
                                            shift, plaquetteSize, distList,
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
      int sizeSmall = sizeA;
      int *workspace;

      if (sizeB < sizeSmall) {
        sizeSmall = sizeB;
      }
      workspace = (int*)malloc(sizeof(int)*(2*sizeA*sizeB+2*sizeSmall));
      #pragma omp for
      for (i = 0; i < numA; i++) {
        for (j = 0; j <= i; j++) {
          kernelMatr[i*numA + j] = GPWKernel(cfgsA[i], plaquetteAIdx, sizeA,
                                             cfgsA[j], plaquetteAIdx, sizeA,
                                             dim, power, theta0, thetaC, tRSym,
                                             shift, plaquetteSize, distList,
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
                  const int power, const int rC, const double theta0,
                  const double thetaC, const int tRSym, const int shift,
                  double *kernelVec) {
  int i, j;
  int **cfgsA;

  int plaquetteSize;
  int *plaquetteAIdx, *plaquetteBIdx, *distList;

  plaquetteSize = SetupPlaquetteIdx(rC, neighboursA, sizeA, neighboursRef,
                                    sizeRef, dim, &plaquetteAIdx,
                                    &plaquetteBIdx, &distList);
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
    int sizeSmall = sizeA;
    int *workspace;

    if (sizeRef < sizeSmall) {
      sizeSmall = sizeRef;
    }
    workspace = (int*)malloc(sizeof(int)*(2*sizeA*sizeRef+2*sizeSmall));

    #pragma omp for
    for (i = 0; i < numA; i++) {
      kernelVec[i] = GPWKernel(cfgsA[i], plaquetteAIdx, sizeA, configRef,
                               plaquetteBIdx, sizeRef, dim, power, theta0,
                               thetaC, tRSym, shift, plaquetteSize, distList,
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
