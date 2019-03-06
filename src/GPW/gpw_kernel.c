// TODO include license and description

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "gpw_kernel.h"

double GPWKernel1(const int *configA, const int sizeA, const int *configB, const int sizeB) {
  int countA[4] = {0, 0, 0, 0};
  int countB[4] = {0, 0, 0, 0};
  int up, down;
  int i;
  double kernel = 0.0;

  for (i = 0; i < sizeA; i++) {
    up = configA[i];
    down = configA[i+sizeA];

    countA[0] += up&(1-down);
    countA[1] += (1-up)&down;
    countA[2] += up&down;
    countA[3] += (1-up)&(1-down);
  }

  for (i = 0; i < sizeB; i++) {
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

  return kernel;
}

// TODO: extension to more sophisticated lattices/unit cells, currently only 1D chain with indices ordered according to chain topology
double GPWKernel2(const int *configA, const int sizeA, const int *configB, const int sizeB) {
  int countA[16] = {0};
  int countB[16] = {0};

  int up, down, up_n, down_n;
  int i,j;
  double kernel = 0.0;

  for (i = 0; i < sizeA; i++) {
    up = configA[i];
    up_n = configA[(i+1)%sizeA];

    down = configA[i+sizeA];
    down_n = configA[(i+1)%sizeA+sizeA];

    for (j = 0; j < 16; j++) {
      int k = (j % 4 < 2) ? 0 : 1;
      int l = (j % 8 < 4) ? 0 : 1;
      int m = (j < 8) ? 0 : 1;

      countA[j] += (j%2-up)&(k-down)&(l-up_n)&(m-down_n);
    }
  }

  for (i = 0; i < sizeB; i++) {
    up = configB[i];
    up_n = configB[(i+1)%sizeB];

    down = configB[i+sizeB];
    down_n = configB[(i+1)%sizeB+sizeB];

    for (j = 0; j < 16; j++) {
      int k = (j % 4 < 2) ? 0 : 1;
      int l = (j % 8 < 4) ? 0 : 1;
      int m = (j < 8) ? 0 : 1;

      countB[j] += (j%2-up)&(k-down)&(l-up_n)&(m-down_n);
    }
  }

  for (i = 0; i < 16; i++) {
    kernel += (double)(countA[i]*countB[i]);
  }

  return kernel;
}

/* TODO: extension to more sophisticated lattices/unit cells, currently only 1D chain with
indices ordered according to chain topology*/
double GPWKernelN(const int *configA, const int sizeA, const int *configB, const int sizeB, const int n) {
  if (n == 1) {
    return GPWKernel1(configA, sizeA, configB, sizeB);
  }

  /* TODO: detailed performance comparison for n = 2, this general method is faster
  for small lattices but for larger lattices the GPWKernel2 should be used */
  else {
    int comp;
    int i, j, a, b, k;
    double kernel=0.0;

    int *delta = (int*)malloc(sizeof(int)*(sizeA*sizeB));
    CalculatePairDelta(delta, configA, sizeA, configB, sizeB);

    for (i = 0; i < sizeA; i++) {
      for (a = 0; a < sizeB; a++) {
        for (k = 0; k < n; k++) {
          j = (i+k)%sizeA;
          b = (a+k)%sizeB;
          comp = delta[j*sizeB+b];
          if(!comp) {
            break;
          }
        }
        if (comp) {
          kernel += 1.0;
        }
      }
    }

    free(delta);
    return kernel;
  }
}


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
                         const int *trnNeighbours, const int rC, const double theta0,
                         const double thetaC, int *workspace) {
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
            rangeSum += exp(-dist*dist/thetaC);
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
              rangeSum += exp(-dist*dist/thetaC);
            }
          }
        }

        // only continue in additional dension if site in this dension has coordinate 0
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
                 const int dim, const int rC, const double theta0, const double thetaC) {
  int i, a, power, k;

  double kernel = 0.0;
  int *delta = (int*)malloc(sizeof(int)*(trnSize*sysSize));
  int *workspace = (int*)malloc((trnSize+sysSize+8*dim*rC+4*dim*rC*dim)*sizeof(int));

  power = (trnSize <= sysSize) ? trnSize-1 : sysSize-1;

  CalculatePairDelta(delta, sysCfg, sysSize, trnCfg, trnSize);

  for (i = 0; i < sysSize; i++) {
    for (a = 0; a < trnSize; a++) {
      if (delta[i*trnSize+a]) {
        kernel += pow((1.0 + theta0/power * CalculateInnerSum(i, a, delta, sysSize, trnSize, dim, sysNeighbours, trnNeighbours, rC, theta0, thetaC, workspace)), power);
      }
    }
  }

  free(workspace);
  free(delta);

  return kernel;
}
