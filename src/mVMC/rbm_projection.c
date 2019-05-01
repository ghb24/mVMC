/*
TODO: Add License + Description*/
#include "global.h"
#include "math.h"
#include "rbm_projection.h"

double complex RBMHiddenLayerSum(const int i, const int *eleNum) {
  int j, idx;
  double complex sum;

  sum = RBMVar[RBMNVisibleIdx+i];

  for(j = 0; j < Nsite2; j++) {
    idx = RBMWeightMatrIdx[j][i];
    sum += RBMVar[RBMNVisibleIdx+RBMNHiddenIdx+idx] * (2*eleNum[j]-1);
  }

  return sum;
}


double complex RBMVal(const int *eleNum) {
  int i, j;
  double complex result = 1.0;
  double complex sum = 0.0;

  if(RBMNVisibleIdx > 0) {
    #pragma omp parallel for default(shared) private(i) reduction(+:sum)
    for(i = 0; i < Nsite2; i++) {
      sum += RBMVar[RBMVisIdx[i]]*(2*eleNum[i]-1);
    }

    result = cexp(sum);
  }

  #pragma omp parallel for default(shared) private(i,sum) reduction(*:result)
  for(i = 0; i < RBMNHiddenIdx; i++) {
    sum = RBMHiddenLayerSum(i, eleNum);
    result *= (cexp(sum) + cexp(-sum))/2.0;
  }

  return result;
}
