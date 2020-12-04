/*
TODO: Add License + Description*/
#include "global.h"
#include "QGPS.h"
#include "include/global.h"
#include <omp.h>

double complex LogQGPSVal(const double complex *QGPSAmplitude) {
  return clog(QGPSVal(QGPSAmplitude));
}

double complex QGPSVal(const double complex *QGPSAmplitude) {
  return QGPSAmplitude[0];
}

double complex LogQGPSRatio(const double complex *QGPSAmplitudeNew, const double complex *QGPSAmplitudeOld) {
  return (LogQGPSVal(QGPSAmplitudeNew)-LogQGPSVal(QGPSAmplitudeOld));
}

double complex QGPSRatio(const double complex *QGPSAmplitudeNew, const double complex *QGPSAmplitudeOld) {
  return (QGPSVal(QGPSAmplitudeNew)/QGPSVal(QGPSAmplitudeOld));
}

void ComputeQGPSAmplitude(double complex *QGPSAmplitude, const double complex *eleGPWInSum) {
  int i, tSym, l;
  double complex expansionargument;
  double complex tmpValue;
  double complex value;
  int factorial = 1;

  int shiftSys = 1;
  if (abs(GPWShift[0]) & 1) {
    shiftSys = Nsite;
  }

  if(QGPSSymMode) {
    value = 0.0;

    #pragma omp parallel for default(shared) private(tSym, i, expansionargument, tmpValue, factorial, l) reduction(+:value)
    for (tSym = 0; tSym <= GPWTRSym[0]; tSym++) {
      for (i = 0; i < shiftSys; i++) {
        expansionargument = 0.0;
        
        for(l=0;l<NGPWIdx;l++) {
          expansionargument += eleGPWInSum[l*(GPWTRSym[0]+1)*shiftSys + tSym * shiftSys + i];
        }

        if (GPWExpansionOrder == -1) {
          tmpValue = cexp(expansionargument);
        }

        else {
          tmpValue = 0.0;
          factorial = 1;
          for(l = 0; l <= GPWExpansionOrder; l++) {
            tmpValue += cpow(expansionargument, l)/factorial;
            factorial *= (l+1);
          }
        }

        value += tmpValue;
      }
    }
  }
  else {
    value = 1.0;

    #pragma omp parallel for default(shared) private(tSym, i, expansionargument, tmpValue, factorial, l) reduction(*:value)
    for (tSym = 0; tSym <= GPWTRSym[0]; tSym++) {
      for (i = 0; i < shiftSys; i++) {
        expansionargument = 0.0;
        
        for(l=0;l<NGPWIdx;l++) {
          expansionargument += eleGPWInSum[l*(GPWTRSym[0]+1)*shiftSys + tSym * shiftSys + i];
        }

        if (GPWExpansionOrder == -1) {
          tmpValue = cexp(expansionargument);
        }

        else {
          tmpValue = 0.0;
          factorial = 1;
          for(l = 0; l <= GPWExpansionOrder; l++) {
            tmpValue += cpow(expansionargument, l)/factorial;
            factorial *= (l+1);
          }
        }

        value *= tmpValue;
      }
    }
  }
  QGPSAmplitude[0] = value;
  return;
}

void CalculateQGPSInsum(double complex *eleGPWInSum, const int *eleNum) {
  int shiftSys = 1;

  int plaquetteSize = GPWPlaquetteSizes[0];
  int *plaquetteAIdx = GPWSysPlaquetteIdx[0];

  if (abs(GPWShift[0]) & 1) {
    shiftSys = Nsite;
  }

  #pragma omp parallel default(shared)
  {
    int i, j, tSym, k;
    double complex innerSum;
    double complex element;
    int id;
    #pragma omp for
    for (tSym = 0; tSym <= GPWTRSym[0]; tSym++) {
      for (j = 0; j < shiftSys; j++) {
        for(i=0;i<NGPWIdx;i++) {
          innerSum = 1.0;
          for (k = 0; k < plaquetteSize; k++) {
            if (tSym) {
              id = (1-eleNum[plaquetteAIdx[j*plaquetteSize+k]]);
              if (LocSpn[plaquetteAIdx[j*plaquetteSize+k]] != 1) {
                id += 2 * (1-eleNum[plaquetteAIdx[j*plaquetteSize+k]+Nsite]);
              }
            }
            else {
              id = eleNum[plaquetteAIdx[j*plaquetteSize+k]];
              if (LocSpn[plaquetteAIdx[j*plaquetteSize+k]] != 1) {
                id += 2 * eleNum[plaquetteAIdx[j*plaquetteSize+k]+Nsite];
              }
            }
            element = GPWDistWeights[4*plaquetteSize*i + k*4 + id];
            innerSum *= element;
          }
          eleGPWInSum[i*(GPWTRSym[0]+1)*shiftSys + tSym * shiftSys + j] = innerSum;
        }
      }
    }
  }
  return;
}

void UpdateQGPSInSum(const int ri, const int rj, const int *cfgOldReduced,
                     double complex *eleGPWInSumNew, const double complex *eleGPWInSumOld,
                     const int *eleNum) {

  int shiftSys = 1;
  int count = 0;

  int plaquetteSize = GPWPlaquetteSizes[0];

  int *plaquetteAIdx = GPWSysPlaquetteIdx[0];
  int **plaqHash = GPWSysPlaqHash[0];



  if (abs(GPWShift[0]) & 1) {
    shiftSys = Nsite;
  }

  memcpy(eleGPWInSumNew, eleGPWInSumOld, sizeof(double complex)*GPWInSumSize);

  #pragma omp parallel default(shared)
  {
    int i, j, tSym, occupationIdOld, occupationIdNew, k;
    double complex elementOld;
    double complex elementNew;
    #pragma omp for
    for (tSym = 0; tSym <= GPWTRSym[0]; tSym++) {
      if (tSym) {
        occupationIdOld = (1-cfgOldReduced[0]);
        occupationIdNew = (1-eleNum[ri]);
        if (LocSpn[ri] != 1) {
          occupationIdOld += 2 * (1-cfgOldReduced[2]);
          occupationIdNew += 2 * (1-eleNum[ri+Nsite]);
        }
      }
      else {
        occupationIdOld = cfgOldReduced[0];
        occupationIdNew = eleNum[ri];
        if (LocSpn[ri] != 1) {
          occupationIdOld += 2 * cfgOldReduced[2];
          occupationIdNew += 2 * eleNum[ri+Nsite];
        }
      }
      for (j = 0; j < shiftSys; j++) {
        for(i=0;i<NGPWIdx;i++) {
          k = plaqHash[ri +j *Nsite][0];
          elementOld = GPWDistWeights[4*plaquetteSize*i + k*4 + occupationIdOld];
          elementNew = GPWDistWeights[4*plaquetteSize*i + k*4 + occupationIdNew];

          if (cabs(elementNew) > 0.0 || cabs(elementOld) > 0.0) {
            eleGPWInSumNew[i*(GPWTRSym[0]+1)*shiftSys + tSym * shiftSys + j] /= elementOld;
            eleGPWInSumNew[i*(GPWTRSym[0]+1)*shiftSys + tSym * shiftSys + j] *= elementNew;
          }
        }
      }
      if (ri != rj) {
        if (tSym) {
          occupationIdOld = (1-cfgOldReduced[1]);
          occupationIdNew = (1-eleNum[rj]);
          if (LocSpn[ri] != 1) {
            occupationIdOld += 2 * (1-cfgOldReduced[3]);
            occupationIdNew += 2 * (1-eleNum[rj+Nsite]);
          }
        }
        else {
          occupationIdOld = cfgOldReduced[1];
          occupationIdNew = eleNum[rj];
          if (LocSpn[ri] != 1) {
            occupationIdOld += 2 * cfgOldReduced[3];
            occupationIdNew += 2 * eleNum[rj+Nsite];
          }
        }
        for (j = 0; j < shiftSys; j++) {
          for(i=0;i<NGPWIdx;i++) {
            k = plaqHash[rj + j *Nsite][0];
            elementOld = GPWDistWeights[4*plaquetteSize*i + k*4 + occupationIdOld];
            elementNew = GPWDistWeights[4*plaquetteSize*i + k*4 + occupationIdNew];
            
            if (cabs(elementNew) > 0.0 || cabs(elementOld) > 0.0) {
              eleGPWInSumNew[i*(GPWTRSym[0]+1)*shiftSys + tSym * shiftSys + j] /= elementOld;
              eleGPWInSumNew[i*(GPWTRSym[0]+1)*shiftSys + tSym * shiftSys + j] *= elementNew;
            }
          }
        }
      }
    }
  }
  return;
}

void calculateQGPSderivative(double complex *derivative, const double complex *QGPSAmplitude,
                             const double complex *eleGPWInSum, const int *eleNum) {
  int i, j, k, l, occId, tSym, id;
  double complex expansionargument;
  double complex prefactor;
  double complex innerderiv;
  int factorial;
  int trnSize, plaqSize;
  int shiftSys, translationSys;
  int *sysPlaquetteIdx;
  int offset = NGPWIdx + NGPWTrnLat;
  double complex *distWeightDeriv = derivative + 2*(NGPWIdx + NGPWTrnLat);
  
  #pragma omp parallel for default(shared) private(i)
  for (i = 0; i < NGPWDistWeights; i++) {
    distWeightDeriv[i*2] = 0.0;
  }

  trnSize = GPWTrnSize[0];
  plaqSize = GPWPlaquetteSizes[0];
  sysPlaquetteIdx = GPWSysPlaquetteIdx[0];

  shiftSys = 1;
  translationSys = 1;

  if (abs(GPWShift[0]) & 1) {
    shiftSys = Nsite;
    if (GPWShift[0] < 0) {
      translationSys = trnSize;
    }
  }

  if (QGPSSymMode) {
    for (tSym = 0; tSym <= GPWTRSym[0]; tSym++) {
      for (l = 0; l < shiftSys; l++) {
        expansionargument = 0.0;

        #pragma omp parallel for default(shared) private(i) reduction(+:expansionargument)
        for(i=0;i<NGPWIdx;i++) {
          expansionargument += eleGPWInSum[i*(GPWTRSym[0]+1)*shiftSys + tSym * shiftSys + l];
        }

        if (GPWExpansionOrder == -1) {
          prefactor = cexp(expansionargument);
        }

        else {
          prefactor = 0.0;
          factorial = 1;
          for(l = 0; l <= GPWExpansionOrder; l++) {
            prefactor += cpow(expansionargument, l)/factorial;
            factorial *= (l+1);
          }
        }

        #pragma omp parallel for default(shared) private(i, j, k, occId, id, innerderiv)
        for (i = 0; i < NGPWIdx; i++) {
          for (k = 0; k < plaqSize; k++) {
            if (tSym) {
              occId = (1-eleNum[sysPlaquetteIdx[l*plaqSize+k]]);
              if (LocSpn[sysPlaquetteIdx[l*plaqSize+k]] != 1) {
                occId += 2 * (1-eleNum[sysPlaquetteIdx[l*plaqSize+k]+Nsite]);
              }
            }
            else {
              occId = eleNum[sysPlaquetteIdx[l*plaqSize+k]];
              if (LocSpn[sysPlaquetteIdx[l*plaqSize+k]] != 1) {
                occId += 2 * eleNum[sysPlaquetteIdx[l*plaqSize+k]+Nsite];
              }
            }
            if (cabs(eleGPWInSum[i*(GPWTRSym[0]+1)*shiftSys + tSym * shiftSys + l]) > 0.0) {
              innerderiv = eleGPWInSum[i*(GPWTRSym[0]+1)*shiftSys + tSym * shiftSys + l] / GPWDistWeights[4*plaqSize*i + 4*k + occId];
            }
            else {
              innerderiv = 1.0;
              for (j = 0; j < plaqSize; j++) {
                if (j != k) {
                  if (tSym) {
                    id = (1-eleNum[sysPlaquetteIdx[l*plaqSize+j]]);
                    if(LocSpn[sysPlaquetteIdx[l*plaqSize+j]] != 1) {
                      id += 2 * (1-eleNum[sysPlaquetteIdx[l*plaqSize+j]+Nsite]);
                    }
                  }
                  else {
                    id = eleNum[sysPlaquetteIdx[l*plaqSize+j]];
                    if(LocSpn[sysPlaquetteIdx[l*plaqSize+j]] != 1) {
                      id += 2 * eleNum[sysPlaquetteIdx[l*plaqSize+j]+Nsite];
                    }
                  }
                  innerderiv *= GPWDistWeights[4*plaqSize*i + j*4 + id];
                }
              }
            }
            distWeightDeriv[2*(4*plaqSize*i + 4*k + occId)] += prefactor*innerderiv;
          }
        }
      }
    }

    prefactor = QGPSVal(QGPSAmplitude);

    for (i = 0; i < NGPWDistWeights; i++) {
        distWeightDeriv[i*2] /= prefactor;
    }
  }
  else {
    for (tSym = 0; tSym <= GPWTRSym[0]; tSym++) {
      for (l = 0; l < shiftSys; l++) {
        #pragma omp parallel for default(shared) private(i, j, k, occId, id, innerderiv)
        for (i = 0; i < NGPWIdx; i++) {
          for (k = 0; k < plaqSize; k++) {
            if (tSym) {
              occId = (1-eleNum[sysPlaquetteIdx[l*plaqSize+k]]);
              if (LocSpn[sysPlaquetteIdx[l*plaqSize+k]] != 1) {
                occId += 2 * (1-eleNum[sysPlaquetteIdx[l*plaqSize+k]+Nsite]);
              }
            }
            else {
              occId = eleNum[sysPlaquetteIdx[l*plaqSize+k]];
              if (LocSpn[sysPlaquetteIdx[l*plaqSize+k]] != 1) {
                occId += 2 * eleNum[sysPlaquetteIdx[l*plaqSize+k]+Nsite];
              }
            }
            if (cabs(GPWDistWeights[4*plaqSize*i + 4*k + occId]) > 0.0) {
              innerderiv = eleGPWInSum[i*(GPWTRSym[0]+1)*shiftSys + tSym * shiftSys + l] / GPWDistWeights[4*plaqSize*i + 4*k + occId];
            }
            else {
              innerderiv = 1.0;
              for (j = 0; j < plaqSize; j++) {
                if (j != k) {
                  if (tSym) {
                    id = (1-eleNum[sysPlaquetteIdx[l*plaqSize+j]]);
                    if(LocSpn[sysPlaquetteIdx[l*plaqSize+j]] != 1) {
                      id += 2 * (1-eleNum[sysPlaquetteIdx[l*plaqSize+j]+Nsite]);
                    }
                  }
                  else {
                    id = eleNum[sysPlaquetteIdx[l*plaqSize+j]];
                    if(LocSpn[sysPlaquetteIdx[l*plaqSize+j]] != 1) {
                      id += 2 * eleNum[sysPlaquetteIdx[l*plaqSize+j]+Nsite];
                    }
                  }
                  innerderiv *= GPWDistWeights[4*plaqSize*i + j*4 + id];
                }
              }
            }
            distWeightDeriv[2*(4*plaqSize*i + 4*k + occId)] += innerderiv;
          }
        }
      }
    }

    if (GPWExpansionOrder >= 0) {
      expansionargument = 0.0 + 0.0*I;
      #pragma omp parallel for default(shared) private(i, tSym, l) reduction(+:expansionargument)
      for(i=0;i<NGPWIdx;i++) {
        for (tSym = 0; tSym <= GPWTRSym[0]; tSym++) {
          for (l = 0; l < shiftSys; l++) {
            expansionargument += eleGPWInSum[i*(GPWTRSym[0]+1)*shiftSys + tSym * shiftSys + l];
          }
        }
      }

      prefactor = 0.0;
      factorial = 1;
      for (i = 0; i < GPWExpansionOrder; i++) {
        prefactor += cpow(expansionargument, i)/factorial;
        factorial *= (i+1);
      }

      prefactor /= QGPSVal(QGPSAmplitude);

      #pragma omp parallel for default(shared) private(i)
      for (i = 0; i < NGPWDistWeights; i++) {
        distWeightDeriv[i*2] *= prefactor;
      }
    }
  }

  #pragma omp parallel for default(shared) private(i)
  for (i = 0; i < NGPWDistWeights; i++) {
    distWeightDeriv[i*2 + 1] = I*distWeightDeriv[i*2];
  }

  return;
}