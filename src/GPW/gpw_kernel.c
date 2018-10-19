// TODO include license and description

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "gpw_kernel.h"

int GPWKernel1(const int *configA, const int sizeA, const int *configB, const int sizeB) {
  int countA[4]={0,0,0,0};
  int countB[4]={0,0,0,0};
  int up,down;
  int i;
  int kernel=0;

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
    kernel += countA[i]*countB[i];
  }

  return kernel;
}
