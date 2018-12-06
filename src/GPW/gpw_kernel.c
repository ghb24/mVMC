// TODO include license and description

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "gpw_kernel.h"

double GPWKernel1(const int *configA, const int sizeA, const int *configB, const int sizeB) {
  int countA[4]={0,0,0,0};
  int countB[4]={0,0,0,0};
  int up,down;
  int i;
  double kernel=0.0;

  for(i=0;i<sizeA;i++) {
    up = configA[i];
    down = configA[i+sizeA];

    countA[0] += up&(1-down);
    countA[1] += (1-up)&down;
    countA[2] += up&down;
    countA[3] += (1-up)&(1-down);
  }

  for(i=0;i<sizeB;i++) {
    up = configB[i];
    down = configB[i+sizeB];

    countB[0] += up&(1-down);
    countB[1] += (1-up)&down;
    countB[2] += up&down;
    countB[3] += (1-up)&(1-down);
  }

  for(i=0;i<4;i++) {
    kernel += (double)(countA[i]*countB[i]);
  }

  return kernel;
}

// TODO: extension to more sophisticated lattices/unit cells, currently only 1D chain with indices ordered according to chain topology
double GPWKernel2(const int *configA, const int sizeA, const int *configB, const int sizeB) {
  int countA[16] = {0};
  int countB[16] = {0};

  int up,down,up_n,down_n;
  int i,j;
  double kernel=0.0;

  for(i=0;i<sizeA;i++) {
    up = configA[i];
    up_n = configA[(i+1)%sizeA];

    down = configA[i+sizeA];
    down_n = configA[(i+1)%sizeA+sizeA];

    for(j=0;j<16;j++) {
      int k = (j % 4 < 2) ? 0 : 1;
      int l = (j % 8 < 4) ? 0 : 1;
      int m = (j < 8) ? 0 : 1;

      countA[j] += (j%2-up)&(k-down)&(l-up_n)&(m-down_n);
    }
  }

  for(i=0;i<sizeB;i++) {
    up = configB[i];
    up_n = configB[(i+1)%sizeB];

    down = configB[i+sizeB];
    down_n = configB[(i+1)%sizeB+sizeB];

    for(j=0;j<16;j++) {
      int k = (j % 4 < 2) ? 0 : 1;
      int l = (j % 8 < 4) ? 0 : 1;
      int m = (j < 8) ? 0 : 1;

      countB[j] += (j%2-up)&(k-down)&(l-up_n)&(m-down_n);
    }
  }

  for(i=0;i<16;i++) {
    kernel += (double)(countA[i]*countB[i]);
  }

  return kernel;
}

// TODO: extension to more sophisticated lattices/unit cells, currently only 1D chain with indices ordered according to chain topology
double GPWKernelN(const int *configA, const int sizeA, const int *configB, const int sizeB, const int n) {
  if (n == 1) {
    return GPWKernel1(configA, sizeA, configB, sizeB);
  }

  // TODO: detailed performance comparison for n = 2, atm this method is faster for the small training lattice sizes but maybe GPWKernel2 should be used
  else {
    int comp;
    int i,j,k;
    double kernel=0.0;

    for(i=0;i<sizeA;i++) {
      for(j=0;j<sizeB;j++) {
        for(k=0;k<n;k++) {
          comp = (configA[(i+k)%sizeA]==configB[(j+k)%sizeB])&&(configA[(i+k)%sizeA+sizeA]==configB[(j+k)%sizeB+sizeB]);
          if(comp == 0) {
            break;
          }
        }
        if(comp != 0) {
          kernel += 1.0;
        }
      }
    }

    return kernel;
  }
}


void CalculatePairDelta(int *delta, const int *sysCfg, const int sysSize, const int *trnCfg, const int trnSize) {
  int i,a;

  for (i=0;i<sysSize;i++) {
    for (a=0;a<trnSize;a++) {
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

double CalculateRangeSum(const int i, const int a, const int *delta, const int sysSize,
                         const int trnSize, const int d, int **neighbours, const int rC,
                         const double theta0, const double thetaC, int *workspace) {
  int j,b,k,l,dist,count,tmpCount,dim;
  double rangeSum = 0.0;

  int multiple = 2*d*rC;

  int **neighboursSys = neighbours;
  int **neighboursTrn = neighbours + sysSize;

  // TODO: store this in bitstrings
  int *visitedSys = workspace;
  int *visitedTrn = visitedSys+sysSize;

  for(k=0;k<sysSize+trnSize;k++) {
    visitedSys[k] = 0;
  }


  int *prevIndSys = visitedTrn+trnSize;
  int *tmpPrevIndSys = prevIndSys + multiple;
  int *prevIndTrn = tmpPrevIndSys + multiple;
  int *tmpPrevIndTrn = prevIndTrn + multiple;

  int *directions = tmpPrevIndTrn + multiple;
  int *tmpDirections = directions + multiple*d;



  // set up starting point (dist = 0)
  count = 1;
  dist = 0;

  j = i;
  b = a;
  prevIndSys[0] = j;
  prevIndTrn[0] = b;
  visitedSys[j] = 1;
  visitedTrn[b] = 1;

  for(dim=0; dim<d; dim++) {
    directions[dim] = 0;
  }

  if(delta[j*trnSize+b]) {
    rangeSum += 1;
  }

  while(dist < rC && count > 0) {
    dist ++;
    tmpCount = 0;

    for(k=0; k<count; k++) {

      // add the corresponding neighbours
      for(dim=0; dim<d; dim++) {
        int dir = directions[d*k+dim]>=0?0:1; // positive or negative direction

        // add neighbour in respective direction
        j = neighboursSys[prevIndSys[k]][dim*2+dir];
        b = neighboursTrn[prevIndTrn[k]][dim*2+dir];

        if(!visitedSys[j] && !visitedTrn[b]) {

          tmpPrevIndSys[tmpCount] = j;
          tmpPrevIndTrn[tmpCount] = b;
          visitedSys[j] = 1;
          visitedTrn[b] = 1;

          for(l=0; l<d; l++) {
            tmpDirections[d*tmpCount+l] = directions[d*tmpCount+l];
          }
          tmpDirections[d*tmpCount+dim] += (dir==0?1:-1);
          tmpCount ++;


          if(delta[j*trnSize+b]) {
            rangeSum += exp(-dist*dist/thetaC);
          }
        }

        // add neighbour in opposing direction if coordinate = 0
        if(directions[d*k+dim] == 0) {
          dir = dir ^ 1;
          j = neighboursSys[prevIndSys[k]][dim*2+dir];
          b = neighboursTrn[prevIndTrn[k]][dim*2+dir];

          if(!visitedSys[j] && !visitedTrn[b]) {
            tmpPrevIndSys[tmpCount] = j;
            tmpPrevIndTrn[tmpCount] = b;
            visitedSys[j] = 1;
            visitedTrn[b] = 1;
            for(l=0; l<d; l++) {
              tmpDirections[d*tmpCount+l] = directions[k*d+l];
            }
            tmpDirections[d*tmpCount+dim] += (dir==0?1:-1);
            tmpCount ++;

            if(delta[j*trnSize+b]) {
              rangeSum += exp(-dist*dist/thetaC);
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

    for (k=0;k<tmpCount;k++) {
      prevIndSys[k] = tmpPrevIndSys[k];
      prevIndTrn[k] = tmpPrevIndTrn[k];

      for(l=0; l<d; l++) {
        directions[k*d+l] = tmpDirections[k*d+l];
      }
    }

  }

  rangeSum += exp(1/theta0);

  return rangeSum;
}

double GPWKernel(const double kernelOld, const double *termMatr, const int updateCount) {
  int i;
  double kernel = kernelOld;

  for (i=0;i<updateCount;i++) {
    kernel += termMatr[i];
  }

  return kernel;
}

double GPWKernelInPlace(const int *sysCfg, const int sysSize, const int *trnCfg,
                        const int trnSize, const int rC, const double theta0, const double thetaC) {
  int i, a, power, k;

  double kernel = 0.0;
  int *delta = (int*)malloc(sizeof(int)*trnSize*sysSize);

  int d = 1;

  int *workspace = (int*)malloc((trnSize+sysSize+8*d*rC+4*d*rC*d)*sizeof(int));



  int **neighboursSys = (int**)malloc((sysSize+trnSize)*sizeof(int*));
  int **neighboursTrn = neighboursSys+sysSize;

  for(i=0; i<sysSize; i++) {
    neighboursSys[i] = (int*)malloc(2*sizeof(int));

    neighboursSys[i][0] = (i+1)>=0?(i+1)%sysSize:(i+1)%sysSize+sysSize;
    neighboursSys[i][1] = (i-1)>=0?(i-1)%sysSize:(i-1)%sysSize+sysSize;
  }

  for(i=0; i<trnSize; i++) {
    neighboursTrn[i] = (int*)malloc(2*sizeof(int));

    neighboursTrn[i][0] = (i+1)>=0?(i+1)%trnSize:(i+1)%trnSize+trnSize;
    neighboursTrn[i][1] = (i-1)>=0?(i-1)%trnSize:(i-1)%trnSize+trnSize;
  }



  if (trnSize <= sysSize) {
    power = trnSize-1;
  }

  else {
    power = sysSize-1;
  }

  CalculatePairDelta(delta, sysCfg, sysSize, trnCfg, trnSize);

  for (i=0;i<sysSize;i++) {
    for (a=0;a<trnSize;a++) {
      if (delta[i*trnSize+a]) {
        kernel += pow(CalculateRangeSum(i, a, delta, sysSize, trnSize, d, neighboursSys, rC, theta0, thetaC, workspace), power);
      }
    }
  }


  for(i=0; i<trnSize; i++) {
    free(neighboursTrn[i]);
  }

  for(i=0; i<sysSize; i++) {
    free(neighboursSys[i]);
  }

  free(neighboursSys);
  free(workspace);
  free(delta);

  return kernel;
}
