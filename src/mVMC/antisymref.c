#include "./include/antisymref.h"
#include "./include/global.h"

int ComputeRefState(const int *eleIdx, const int *eleNum, int *workspace) {
  int i, j, k, pos, totalCount, cycleLength, posNew, minIdx, totalCycle, evenCycles, translationId, trId;
  int *master = workspace;
  int *masterCandidate = master + Nsite2;
  int *eleCfg = master + Nsite2;
  int trSym = 0;
  int sign = 1;

  return 1;

  if (NGPWIdx > 0) {
    if(GPWTRSym[0] == 1) {
      trSym = 1;
    }
  }


  #pragma omp parallel for default(shared) private(i)
  for(i = 0; i < Nsite2; i++) {
    masterCandidate[i] = -1;
  }

  for (i = 0; i < NMPTrans; i++) {
    for (j = 0; j <= trSym; j++) {
      #pragma omp parallel for default(shared) private(k)
      for (k = 0; k < Nsite; k++) {
        master[QPTrans[i][k] + j * Nsite] = eleNum[k];
        master[QPTrans[i][k] + (1-j) * Nsite] = eleNum[k+Nsite];
      }
      pos = 0;
      while (master[pos] == masterCandidate[pos]) {
        pos++;
      }

      // accept config
      if (master[pos] > masterCandidate[pos]) {
        memcpy(masterCandidate, master, sizeof(int)*Nsite2);
        translationId = i;
        trId = j;
      }
    }
  }

  pos = 0;
  for(i = 0; i < Nsite2; i++) {
    if (masterCandidate[i]) {
      master[pos] = i;
      pos++;
    }
    eleCfg[i] = -1;
  }

  for (i = 0; i < Ne; i++) {
    eleCfg[QPTrans[translationId][eleIdx[i]] + trId * Nsite] = i;
    if (APFlag) {
      sign *= QPTransSgn[translationId][eleIdx[i]];
    }
  }

  for (i = Ne; i < Nsize; i++) {
    eleCfg[QPTrans[translationId][eleIdx[i]] + (1 - trId) * Nsite] = i;
    if (APFlag) {
      sign *= QPTransSgn[translationId][eleIdx[i]];
    }
  }

  totalCount = 0;
  minIdx = 0;
  evenCycles = 0;
  while (totalCount < Nsize) {
    pos = minIdx;
    while (master[pos] == -1) {
      pos++;
      minIdx++;
    }

    cycleLength = 0;
    while (master[pos] != -1) {
      totalCount++;
      cycleLength++;
      posNew = eleCfg[master[pos]];
      master[pos] = -1;
      pos = posNew;
    }

    if (cycleLength%2 == 0) {
      evenCycles++;
    }
  }


  if (evenCycles%2 == 0) {
    return sign;
  }
  else {
    return -sign;
  }
}


int UpdateRefState(const int *eleIdx, const int *eleNum, int *workspace) {
  return ComputeRefState(eleIdx, eleNum, workspace);
}
