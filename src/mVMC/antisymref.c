#include "./include/antisymref.h"
#include "./include/global.h"

int ComputeRefState(const int *eleIdx, const int *eleNum, int *workspace) {
  int i, j, k, pos, totalCount, cycleLength, posNew, minIdx, evenCycles, translationId, trId;
  int *master = workspace;
  int *masterCandidate = master + Nsite2;
  int *eleCfg = master + Nsite2;
  int trSym = 0;
  int sign = 1;

  if (AlternativeBasisOrdering == 2) {
    for (i = 0; i < Nsite2; i++) {
      masterCandidate[i] = eleNum[i];
    }
  }
  else {
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
  }

  if (AlternativeBasisOrdering == 2) {
     //Marshall sign rule
    for (i = 0; i < Nsite; i++) {
      master[i] = -1;
    }

    master[0] = 0;
    totalCount = 1;

    while(totalCount < Nsite) {
      for(j = 0; j < NCoulombInter; j++) {
        pos = CoulombInter[j][0];
        posNew = CoulombInter[j][1];

        if (master[pos] != -1 && master[posNew] == -1) {
          master[posNew] = 1 - master[pos];
          totalCount ++;
        }

        if (master[posNew] != -1 && master[pos] == -1) {
          master[pos] = 1 - master[posNew];
          totalCount ++;
        }
      }
    }

    totalCount = 0;
    for (i = 0; i < Nsite; i++) {
      if (master[i] == 0) {
        totalCount += masterCandidate[i];
      }
    }

    if (totalCount%2 != 0) {
      sign = -1;
    }
  }

  pos = 0;
  if (AlternativeBasisOrdering > 0) {
    for(i = 0; i < Nsite; i++) {
      for (j = 0; j < 2; j ++) {
        if (masterCandidate[i + (1-j) * Nsite]) {
          master[pos] = 2 * i + j; /* here we use our strange convention that
                                    we want down, up ordering to be consistent with
                                    the notation */
          pos++;
        }
        eleCfg[i + (1-j) * Nsite] = -1;
      }
    }
  }
  else {
    for(i = 0; i < Nsite2; i++) {
      if (masterCandidate[i]) {
        master[pos] = i;
        pos++;
      }
      eleCfg[i] = -1;
    }
  }

  for (i = 0; i < Ne; i++) {
    if (AlternativeBasisOrdering == 1) {
      eleCfg[2 * QPTrans[translationId][eleIdx[i]] + (1-trId)] = i;
    }
    else if (AlternativeBasisOrdering == 2) {
      eleCfg[2 * eleIdx[i] + 1] = i;
    }
    else {
      eleCfg[QPTrans[translationId][eleIdx[i]] + trId * Nsite] = i;
    }
    if (APFlag) {
      sign *= QPTransSgn[translationId][eleIdx[i]];
    }
  }

  for (i = Ne; i < Nsize; i++) {
    if (AlternativeBasisOrdering == 1) {
      eleCfg[2 * QPTrans[translationId][eleIdx[i]] + trId] = i;
    }
    else if (AlternativeBasisOrdering == 2) {
      eleCfg[2 * eleIdx[i]] = i;
    }
    else {
      eleCfg[QPTrans[translationId][eleIdx[i]] + (1 - trId) * Nsite] = i;
    }
    if (APFlag) {
      sign *= QPTransSgn[translationId][eleIdx[i]];
    }
    if (trId) {
      sign *= -1;
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

int ComputeRefState_fsz(const int *eleIdx, const int *eleNum, const int *eleSpn, int *workspace) {
  int i, j, k, pos, totalCount, cycleLength, posNew, minIdx, evenCycles, translationId, trId, spin;
  int *master = workspace;
  int *masterCandidate = master + Nsite2;
  int *eleCfg = master + Nsite2;
  int trSym = 0;
  int sign = 1;
  int nTrans;

  if (AlternativeBasisOrdering == 2) {
    nTrans = 1;
    trSym = 0;
  }
  else {
    nTrans = NMPTrans;
    trSym = 0;
    if (NGPWIdx > 0) {
      if(GPWTRSym[0] == 1) {
        trSym = 1;
      }
    }
  }

  #pragma omp parallel for default(shared) private(i)
  for(i = 0; i < Nsite2; i++) {
    masterCandidate[i] = -1;
  }

  for (i = 0; i < nTrans; i++) {
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

  if (AlternativeBasisOrdering == 2) {
     //Marshall sign rule
    for (i = 0; i < Nsite; i++) {
      master[i] = -1;
    }

    master[0] = 0;
    totalCount = 1;

    while(totalCount < Nsite) {
      for(j = 0; j < NCoulombInter; j++) {
        pos = CoulombInter[j][0];
        posNew = CoulombInter[j][1];

        if (master[pos] != -1 && master[posNew] == -1) {
          master[posNew] = 1 - master[pos];
          totalCount ++;
        }

        if (master[posNew] != -1 && master[pos] == -1) {
          master[pos] = 1 - master[posNew];
          totalCount ++;
        }
      }
    }

    totalCount = 0;
    for (i = 0; i < Nsite; i++) {
      if (master[i] == 0) {
        totalCount += eleNum[i];
      }
    }

    if (totalCount%2 != 0) {
      sign = -1;
    }
  }

  pos = 0;
  if (AlternativeBasisOrdering > 0) {
    for(i = 0; i < Nsite; i++) {
      for (j = 0; j < 2; j ++) {
        if (masterCandidate[i + (1-j) * Nsite]) {
          master[pos] = 2 * i + j; /* here we use our strange convention that
                                    we want down, up ordering to be consistent with
                                    the notation */
          pos++;
        }
        eleCfg[i + (1-j) * Nsite] = -1;
      }
    }
  }
  else {
    for(i = 0; i < Nsite2; i++) {
      if (masterCandidate[i]) {
        master[pos] = i;
        pos++;
      }
      eleCfg[i] = -1;
    }
  }

  for (i = 0; i < Nsize; i++) {
    spin = eleSpn[i];
    if (spin) {
      if (AlternativeBasisOrdering == 1) {
        eleCfg[2 * QPTrans[translationId][eleIdx[i]] + trId] = i;
      }
      else if (AlternativeBasisOrdering == 2) {
        eleCfg[2 * eleIdx[i]] = i;
      }
      else {
        eleCfg[QPTrans[translationId][eleIdx[i]] + (1 - trId) * Nsite] = i;
      }
      if (trId) {
        sign *= -1;
      }
    }
    else {
      if (AlternativeBasisOrdering == 1) {
        eleCfg[2 * QPTrans[translationId][eleIdx[i]] + (1-trId)] = i;
      }
      else if (AlternativeBasisOrdering == 2) {
        eleCfg[2 * eleIdx[i] + 1] = i;
      }
      else {
        eleCfg[QPTrans[translationId][eleIdx[i]] + trId * Nsite] = i;
      }
    }
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


int UpdateRefState_fsz(const int *eleIdx, const int *eleNum, const int *eleSpn, int *workspace) {
  return ComputeRefState_fsz(eleIdx, eleNum, eleSpn, workspace);
}
