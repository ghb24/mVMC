/*
TODO: Add License + Description*/
#include "global.h"
#include "math.h"
#include "rbm_projection.h"

double complex RBMHiddenLayerSum(const int f, const int i, const int *eleNum) {
  int j, idx, targetBasis;
  double complex sum;

  sum = RBMVar[RBMNVisibleIdx+f];

  for(j = 0; j < Nsite; j++) {
    targetBasis = (j + i) % Nsite;
    idx = RBMWeightMatrIdx[j][f];
    sum += RBMVar[RBMNVisibleIdx+RBMNHiddenIdx+idx] * (2*eleNum[targetBasis]-1);

    targetBasis = (j + i) % Nsite + Nsite;
    idx = RBMWeightMatrIdx[j+Nsite][f];
    sum += RBMVar[RBMNVisibleIdx+RBMNHiddenIdx+idx] * (2*eleNum[targetBasis]-1);
  }

  return sum;
}


double complex RBMVal(const int *eleNum) {
  int i, j, f;
  double complex result = 1.0;
  double complex sum = 0.0;

  #pragma omp parallel for default(shared) private(f,i) reduction(+:sum)
  for(f = 0; f < RBMNVisibleIdx; f++) {
    for(i = 0; i < Nsite2; i++) {
      sum += RBMVar[f]*(2*eleNum[i]-1)*Nsite;
    }
  }
  result = cexp(sum);

  #pragma omp parallel for default(shared) private(f,i,sum) reduction(*:result)
  for(f = 0; f < RBMNHiddenIdx; f++) {
    for(i = 0; i < Nsite; i++) {
      sum = RBMHiddenLayerSum(f, i, eleNum);
      result *= (cexp(sum) + cexp(-sum))/2.0;
    }
  }

  return result;
}
