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

  if (shift & 1) {
    shiftA = sizeA;
  }
  if ((shift & 2) >> 1) {
    shiftB = sizeB;
  }

  for (i = 0; i < shiftA; i++) {
    up = configA[i];
    down = configA[i+sizeA];

    countA[0] += up&(1-down);
    countA[1] += (1-up)&down;
    countA[2] += up&down;
    countA[3] += (1-up)&(1-down);
  }

  for (i = 0; i < shiftB; i++) {
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

void GPWKernel1Mat(const int *configsAUp, const int *configsADown,
                   const int sizeA, const int numA, const int *configsBUp,
                   const int *configsBDown, const int sizeB, const int numB,
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

void GPWKernel1Vec(const int *configsAUp, const int *configsADown,
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

int CalculatePlaquette(const int i, const int a, const int *delta, const int sysSize,
                       const int trnSize, const int dim, const int *sysNeighbours,
                       const int *trnNeighbours, const int n, int *workspace) {
  int k, l, orderId, count, tmpCount, d;

  // TODO: store this in bitstrings
  int *visitedSys = workspace;
  int *visitedTrn = visitedSys + sysSize;

  int *prevIndSys = visitedTrn + trnSize;
  int *tmpPrevIndSys = prevIndSys + n;
  int *prevIndTrn = tmpPrevIndSys + n;
  int *tmpPrevIndTrn = prevIndTrn + n;

  int *directions = tmpPrevIndTrn + n;
  int *tmpDirections = directions + n*dim;

  // set up starting point
  int j = i;
  int b = a;

  if (!delta[j*trnSize+b]) {
    return 0;
  }

  for(k = 0; k < sysSize + trnSize; k++) {
    visitedSys[k] = 0;
  }

  count = 1;
  orderId = 1;

  prevIndSys[0] = j;
  prevIndTrn[0] = b;
  visitedSys[j] = 1;
  visitedTrn[b] = 1;

  for (d = 0; d < dim; d++) {
    directions[d] = 0;
  }

  while (orderId < n && count > 0) {
    tmpCount = 0;

    for (k = 0; k < count; k++) {
      // add the corresponding neighbours
      for (d = 0; d < dim; d++) {
        int dir = directions[dim*k+d] >= 0 ? 0 : 1; // positive or negative direction

        // add neighbour in respective direction
        j = sysNeighbours[prevIndSys[k]*2*dim + d*2+dir];
        b = trnNeighbours[prevIndTrn[k]*2*dim + d*2+dir];

        if (!delta[j*trnSize+b]) {
          return 0;
        }

        else if (!visitedSys[j] && !visitedTrn[b]) {
          tmpPrevIndSys[tmpCount] = j;
          tmpPrevIndTrn[tmpCount] = b;

          visitedSys[j] = 1;
          visitedTrn[b] = 1;

          for(l = 0; l < dim; l++) {
            tmpDirections[dim*tmpCount+l] = directions[dim*k+l];
          }

          tmpDirections[dim*tmpCount+d] += (dir==0?1:-1);
          tmpCount++;
          orderId++;

          if (orderId >= n) {
            return 1;
          }
        }

        // add neighbour in opposing direction if coordinate = 0
        if (directions[dim*k+d] == 0) {
          dir = dir ^ 1;
          j = sysNeighbours[prevIndSys[k]*2*dim+d*2+dir];
          b = trnNeighbours[prevIndTrn[k]*2*dim+d*2+dir];

          if (!delta[j*trnSize+b]) {
            return 0;
          }

          else if (!visitedSys[j] && !visitedTrn[b]) {
            tmpPrevIndSys[tmpCount] = j;
            tmpPrevIndTrn[tmpCount] = b;
            visitedSys[j] = 1;
            visitedTrn[b] = 1;

            for (l = 0; l < dim; l++) {
              tmpDirections[dim*tmpCount+l] = directions[k*dim+l];
            }

            tmpDirections[dim*tmpCount+d] += (dir==0?1:-1);
            tmpCount++;
            orderId++;

            if (orderId >= n) {
              return 1;
            }
          }
        }

        // only continue in additional dimension if site in this dimension has coordinate 0
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
  return 1;
}

double GPWKernelN(const int *configA, const int *neighboursA, const int sizeA,
                  const int *configB, const int *neighboursB, const int sizeB,
                  const int dim, const int n, const int tRSym,
                  const int shift) {
  if (n == 1) {
    return GPWKernel1(configA, sizeA, configB, sizeB, tRSym, shift);
  }
  else {
    int comp;
    int i, j, a, b, k;
    int *configFlipped;
    double kernel = 0.0;
    int shiftA = 1;
    int shiftB = 1;

    int *delta = (int*)malloc(sizeof(int)*(sizeA*sizeB));

    // TODO: use less workspace
    int *workspace = (int*)malloc((sizeA+sizeB+4*n+2*dim*n)*sizeof(int));

    if (shift & 1) {
      shiftA = sizeA;
    }
    if ((shift & 2) >> 1) {
      shiftB = sizeB;
    }

    CalculatePairDelta(delta, configA, sizeA, configB, sizeB);

    for (i = 0; i < shiftA; i++) {
      for (a = 0; a < shiftB; a++) {
        if (CalculatePlaquette(i, a, delta, sizeA, sizeB, dim, neighboursA,
                               neighboursB, n, workspace)) {
          kernel += 1.0;
        }
      }
    }

    // add kernel with one configuration flipped to ensure time reversal symmetry is respected
    if (tRSym) {
      // TODO can this be done more efficiently?
      if (sizeA <= sizeB) {
        configFlipped = (int*)malloc(sizeof(int)*(2*sizeA));
        for (i = 0; i < sizeA; i++) {
          configFlipped[i] = configA[i+sizeA];
          configFlipped[i+sizeA] = configA[i];
        }
        CalculatePairDelta(delta, configFlipped, sizeA, configB, sizeB);
      }
      else {
        configFlipped = (int*)malloc(sizeof(int)*(2*sizeB));
        for (i = 0; i < sizeB; i++) {
          configFlipped[i] = configB[i+sizeB];
          configFlipped[i+sizeB] = configB[i];
        }
        CalculatePairDelta(delta, configA, sizeA, configFlipped, sizeB);
      }

      for (i = 0; i < shiftA; i++) {
        for (a = 0; a < shiftB; a++) {
          if (CalculatePlaquette(i, a, delta, sizeA, sizeB, dim, neighboursA,
                                 neighboursB, n, workspace)) {
            kernel += 1.0;
          }
        }
      }
      kernel /= 2.0;
      free(configFlipped);
    }

    free(workspace);
    free(delta);
    return kernel;
  }
}

void GPWKernelNMat(const int *configsAUp, const int *configsADown,
                   const int *neighboursA, const int sizeA, const int numA,
                   const int *configsBUp, const int *configsBDown,
                   const int *neighboursB, const int sizeB, const int numB,
                   const int dim, const int n, const int tRSym, const int shift,
                   const int symmetric, double *kernelMatr) {
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
        kernelMatr[i*numB + j] = GPWKernelN(cfgsA[i], neighboursA, sizeA,
                                            cfgsB[j], neighboursB, sizeB,dim,
                                            n, tRSym, shift);
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
        kernelMatr[i*numA + j] = GPWKernelN(cfgsA[i], neighboursA, sizeA,
                                            cfgsA[j], neighboursA, sizeA, dim,
                                            n, tRSym, shift);
      }
    }
  }

  for (i = 0; i < numA; i++) {
    free(cfgsA[i]);
  }

  free(cfgsA);
}

void GPWKernelNVec(const int *configsAUp, const int *configsADown,
                   const int *neighboursA, const int sizeA, const int numA,
                   const int *configRef, const int *neighboursRef,
                   const int sizeRef, const int dim, const int n,
                   const int tRSym, const int shift, double *kernelVec){
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
    kernelVec[i] = GPWKernelN(cfgsA[i], neighboursA, sizeA, cfgsA[j],
                              neighboursA, sizeA, dim, n, tRSym, shift);
  }

  for (i = 0; i < numA; i++) {
    free(cfgsA[i]);
  }

  free(cfgsA);
}

// TODO: only compute necessary elements if no translational symmetry is required
void CalculatePairDelta(int *delta, const int *sysCfg, const int sysSize, const int *trnCfg, const int trnSize) {
  int i, a;

  for (i = 0; i < sysSize; i++) {
    for (a = 0; a < trnSize; a++) {
      // TODO: can this be optimised by a bitwise comparison?
      if ((sysCfg[i%sysSize]==trnCfg[a%trnSize])&&(sysCfg[i%sysSize+sysSize]==trnCfg[a%trnSize+trnSize])) {
        delta[i*trnSize+a] = 1;
      }
      else {
        delta[i*trnSize+a] = 0;
      }
    }
  }
  return;
}

double CalculateInnerSum(const int i, const int a, const int *delta, const int sysSize,
                         const int trnSize, const int dim, const int *sysNeighbours,
                         const int *trnNeighbours, const int rC, const double thetaC,
                         int *workspace) {
  int j, b, k, l, dist, count, tmpCount, d;
  double rangeSum = 0.0;

  int multiple = 2*dim*rC;

  // TODO: store this in bitstrings
  int *visitedSys = workspace;
  int *visitedTrn = visitedSys + sysSize;

  for(k = 0; k < sysSize + trnSize; k++) {
    visitedSys[k] = 0;
  }


  int *prevIndSys = visitedTrn + trnSize;
  int *tmpPrevIndSys = prevIndSys + multiple;
  int *prevIndTrn = tmpPrevIndSys + multiple;
  int *tmpPrevIndTrn = prevIndTrn + multiple;

  int *directions = tmpPrevIndTrn + multiple;
  int *tmpDirections = directions + multiple*dim;



  // set up starting point (dist = 0)
  count = 1;
  dist = 0;

  j = i;
  b = a;
  prevIndSys[0] = j;
  prevIndTrn[0] = b;
  visitedSys[j] = 1;
  visitedTrn[b] = 1;

  for (d = 0; d < dim; d++) {
    directions[d] = 0;
  }

  if (delta[j*trnSize+b]) {
    rangeSum += 1;
  }

  while (dist < rC && count > 0) {
    dist ++;
    tmpCount = 0;

    for (k = 0; k < count; k++) {
      // add the corresponding neighbours
      for (d = 0; d < dim; d++) {
        int dir = directions[dim*k+d] >= 0 ? 0 : 1; // positive or negative direction

        // add neighbour in respective direction
        j = sysNeighbours[prevIndSys[k]*2*dim + d*2+dir];
        b = trnNeighbours[prevIndTrn[k]*2*dim + d*2+dir];

        if (!visitedSys[j] && !visitedTrn[b]) {
          tmpPrevIndSys[tmpCount] = j;
          tmpPrevIndTrn[tmpCount] = b;
          visitedSys[j] = 1;
          visitedTrn[b] = 1;

          for(l = 0; l < dim; l++) {
            tmpDirections[dim*tmpCount+l] = directions[dim*k+l];
          }

          tmpDirections[dim*tmpCount+d] += (dir==0?1:-1);
          tmpCount ++;

          if (delta[j*trnSize+b]) {
            rangeSum += 1.0/dist; //exp(-dist*dist/thetaC);
          }
        }

        // add neighbour in opposing direction if coordinate = 0
        if (directions[dim*k+d] == 0) {
          dir = dir ^ 1;
          j = sysNeighbours[prevIndSys[k]*2*dim+d*2+dir];
          b = trnNeighbours[prevIndTrn[k]*2*dim+d*2+dir];

          if (!visitedSys[j] && !visitedTrn[b]) {
            tmpPrevIndSys[tmpCount] = j;
            tmpPrevIndTrn[tmpCount] = b;
            visitedSys[j] = 1;
            visitedTrn[b] = 1;

            for (l = 0; l < dim; l++) {
              tmpDirections[dim*tmpCount+l] = directions[k*dim+l];
            }

            tmpDirections[dim*tmpCount+d] += (dir==0?1:-1);
            tmpCount ++;

            if (delta[j*trnSize+b]) {
              rangeSum += 1.0/dist; //exp(-dist*dist/thetaC);
            }
          }
        }

        // only continue in additional dimension if site in this dimension has coordinate 0
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


  return rangeSum;
}

double GPWKernel(const int *sysCfg, const int *sysNeighbours, const int sysSize,
                 const int *trnCfg, const int *trnNeighbours, const int trnSize,
                 const int dim, const int power, const int rC,
                 const double theta0, const double thetaC, const int tRSym,
                 const int shift) {
  int i, a, k;

  double kernel = 0.0;
  int *configFlipped;
  int *delta = (int*)malloc(sizeof(int)*(trnSize*sysSize));
  int *workspace = (int*)malloc((trnSize+sysSize+8*dim*rC+4*dim*rC*dim)*sizeof(int));

  int shiftSys = 1;
  int shiftTrn = 1;

  double norm = 1.0;

  if (shift & 1) {
    shiftSys = sysSize;
  }
  if ((shift & 2) >> 1) {
    shiftTrn = trnSize;
  }

  // normalise
  for (i = 0; i < trnSize*sysSize; i++) {
    delta[i] = 1;
  }
  if (power == -1) {
  	norm = (CalculateInnerSum(0, 0, delta, sysSize, trnSize, dim, sysNeighbours, trnNeighbours, rC, thetaC, workspace));
  }
  else {
  	norm = (theta0 + 1.0/(power) * CalculateInnerSum(0, 0, delta, sysSize, trnSize, dim, sysNeighbours, trnNeighbours, rC, thetaC, workspace));
  }

  CalculatePairDelta(delta, sysCfg, sysSize, trnCfg, trnSize);

  if (power == -1) {
	for (i = 0; i < shiftSys; i++) {
	  for (a = 0; a < shiftTrn; a++) {
	    if (delta[i*trnSize+a]) {

	      kernel += exp( - 1.0/theta0 * (norm - CalculateInnerSum(i, a, delta, sysSize, trnSize, dim, sysNeighbours, trnNeighbours, rC, thetaC, workspace)) ) ;
	    }
	  }
	}
  }
  else {
  	for (i = 0; i < shiftSys; i++) {
  	  for (a = 0; a < shiftTrn; a++) {
  	    if (delta[i*trnSize+a]) {
  	      kernel += pow(((theta0 + 1.0/(power) * CalculateInnerSum(i, a, delta, sysSize, trnSize, dim, sysNeighbours, trnNeighbours, rC, thetaC, workspace))/norm), power);
  	    }
  	  }
  	}
  }

  // add kernel with one configuration flipped to ensure time reversal symmetry is respected
  if (tRSym) {
    // TODO can this be done more efficiently?
    if (trnSize <= sysSize) {
      configFlipped = (int*)malloc(sizeof(int)*(2*trnSize));
      for (i = 0; i < trnSize; i++) {
        configFlipped[i] = trnCfg[i+trnSize];
        configFlipped[i+trnSize] = trnCfg[i];
      }
      CalculatePairDelta(delta, sysCfg, sysSize, configFlipped, trnSize);
    }
    else {
      configFlipped = (int*)malloc(sizeof(int)*(2*sysSize));
      for (i = 0; i < sysSize; i++) {
        configFlipped[i] = sysCfg[i+sysSize];
        configFlipped[i+sysSize] = sysCfg[i];
      }
      CalculatePairDelta(delta, configFlipped, sysSize, trnCfg, trnSize);
    }

    if (power == -1) {
  	for (i = 0; i < shiftSys; i++) {
  	  for (a = 0; a < shiftTrn; a++) {
  	    if (delta[i*trnSize+a]) {
  	      kernel += exp( - 1.0/theta0 * (norm - CalculateInnerSum(i, a, delta, sysSize, trnSize, dim, sysNeighbours, trnNeighbours, rC, thetaC, workspace)) ) ;
  	    }
  	  }
  	}
    }
    else {
    	for (i = 0; i < shiftSys; i++) {
    	  for (a = 0; a < shiftTrn; a++) {
    	    if (delta[i*trnSize+a]) {
    	      kernel += pow(((theta0 + 1.0/(power) * CalculateInnerSum(i, a, delta, sysSize, trnSize, dim, sysNeighbours, trnNeighbours, rC, thetaC, workspace))/norm), power);
    	    }
    	  }
    	}
    }

    kernel /= 2.0;
    free(configFlipped);
  }

  free(workspace);
  free(delta);

  return kernel;
}

void GPWKernelMat(const int *configsAUp, const int *configsADown,
                  const int *neighboursA, const int sizeA, const int numA,
                  const int *configsBUp, const int *configsBDown,
                  const int *neighboursB, const int sizeB, const int numB,
                  const int dim, const int power, const int rC,
                  const double theta0, const double thetaC,
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
		  kernelMatr[i*numB + j] = GPWKernel(cfgsA[i], neighboursA, sizeA,
		                                     cfgsB[j], neighboursB, sizeB, dim,
                                         power, rC, theta0, thetaC, tRSym,
                                         shift);
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
        kernelMatr[i*numA + j] = GPWKernel(cfgsA[i], neighboursA, sizeA,
                                           cfgsA[j], neighboursA, sizeA, dim,
                                           power, rC, theta0, thetaC, tRSym,
                                           shift);
      }
    }
  }

  for (i = 0; i < numA; i++) {
    free(cfgsA[i]);
  }

  free(cfgsA);
}

void GPWKernelVec(const int *configsAUp, const int *configsADown,
                  const int *neighboursA, const int sizeA, const int numA,
                  const int *configRef, const int *neighboursRef,
                  const int sizeRef, const int dim, const int power,
                  const int rC, const double theta0, const double thetaC,
                  const int tRSym, const int shift, double *kernelVec) {
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
      kernelVec[i] = GPWKernel(cfgsA[i], neighboursA, sizeA, configRef,
                               neighboursRef, sizeRef, dim, power, rC,
                               theta0, thetaC, tRSym, shift);
  }

  for (i = 0; i < numA; i++) {
    free(cfgsA[i]);
  }

  free(cfgsA);
}
