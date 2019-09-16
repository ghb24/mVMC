/*
  mVMC - A numerical solver package for a wide range of quantum lattice models based on many-variable Variational Monte Carlo method
  Copyright (C) 2016 The University of Tokyo, All rights reserved.

This program is developed based on the mVMC-mini program
(https://github.com/fiber-miniapp/mVMC-mini)
which follows "The BSD 3-Clause License".

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU General Public License for more details. 

  You should have received a copy of the GNU General Public License 
  along with this program. If not, see http://www.gnu.org/licenses/. 
*/
/*-------------------------------------------------------------
 * Variational Monte Carlo
 * make sample
 *-------------------------------------------------------------
 * by Satoshi Morita
 *-------------------------------------------------------------*/
#include "vmcmake_real.h"
#ifndef _SRC_VMCMAKE_REAL
#define _SRC_VMCMAKE_REAL

#include "global.h"
#include "slater.h"
#include "matrix.h"
#include "pfupdate_real.h"
#include "qp_real.h"
#include "splitloop.h"
#include "vmcmake.h"

void VMCMakeSample_real(MPI_Comm comm) {
  int outStep, nOutStep;
  int inStep, nInStep;
  UpdateType updateType;
  int mi, mj, ri, rj, s, t, i;
  int nAccept = 0;
  int sample;

  double logIpOld, logIpNew; /* logarithm of inner product <phi|L|x> */ // is this ok ? TBC
  int projCntNew[NProj];
  double eleGPWKernNew[NGPWIdx];
  int *eleGPWDeltaNew;
  double *eleGPWInSumNew;
  double complex rbmValOld, rbmValNew; /* value of the RBM projector */
  double pfMNew_real[NQPFull];
  double x, w; // TBC x will be complex number

  int qpStart, qpEnd;
  int rejectFlag;
  int rank, size;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  eleGPWDeltaNew = (int*)malloc(Nsite*GPWTrnCfgSz*sizeof(int));
  eleGPWInSumNew = (double*)malloc(Nsite*GPWTrnCfgSz*sizeof(double));

  SplitLoop(&qpStart, &qpEnd, NQPFull, rank, size);

  StartTimer(30);
  if (BurnFlag == 0) {
    makeInitialSample(TmpEleIdx, TmpEleCfg, TmpEleNum, TmpEleProjCnt,
                      TmpEleGPWKern, TmpEleGPWDelta, TmpEleGPWInSum, qpStart, qpEnd, comm);
  } else {
    copyFromBurnSample(TmpEleIdx, TmpEleCfg, TmpEleNum, TmpEleProjCnt, TmpEleGPWKern, TmpEleGPWDelta, TmpEleGPWInSum);
  }

  if (UseOrbital) {
    CalculateMAll_real(TmpEleIdx, qpStart, qpEnd);
    // printf("DEBUG: maker1: PfM=%lf\n",creal(PfM[0]));
    logIpOld = CalculateLogIP_real(PfM_real, qpStart, qpEnd, comm);
    if (!isfinite(logIpOld)) {
      if (rank == 0) fprintf(stderr, "warning: VMCMakeSample remakeSample logIpOld=%e\n", creal(logIpOld)); //TBC
      makeInitialSample(TmpEleIdx, TmpEleCfg, TmpEleNum, TmpEleProjCnt,
                        TmpEleGPWKern, TmpEleGPWDelta, TmpEleGPWInSum,
                        qpStart, qpEnd, comm);
      CalculateMAll_real(TmpEleIdx, qpStart, qpEnd);
      //printf("DEBUG: maker2: PfM=%lf\n",creal(PfM[0]));
      logIpOld = CalculateLogIP_real(PfM_real, qpStart, qpEnd, comm);
      BurnFlag = 0;
    }
  }
  rbmValOld = RBMVal(TmpEleNum);
  StopTimer(30);

  nOutStep = (BurnFlag == 0) ? NVMCWarmUp + NVMCSample : NVMCSample + 1;
  nInStep = NVMCInterval * Nsite;

  for (i = 0; i < Counter_max; i++) Counter[i] = 0;  /* reset counter */

  for (outStep = 0; outStep < nOutStep; outStep++) {
    for (inStep = 0; inStep < nInStep; inStep++) {

      updateType = getUpdateType(NExUpdatePath);

      if (updateType == HOPPING) { /* hopping */
        Counter[0]++;

        StartTimer(31);
        makeCandidate_hopping(&mi, &ri, &rj, &s, &rejectFlag,
                              TmpEleIdx, TmpEleCfg);
        StopTimer(31);

        if (rejectFlag) continue;

        StartTimer(32);
        StartTimer(60);
        /* The mi-th electron with spin s hops to site rj */
        updateEleConfig(mi, ri, rj, s, TmpEleIdx, TmpEleCfg, TmpEleNum);
        UpdateProjCnt(ri, rj, s, projCntNew, TmpEleProjCnt, TmpEleNum);
        UpdateGPWKern(ri, rj, eleGPWKernNew, eleGPWDeltaNew, eleGPWInSumNew,
                      TmpEleGPWKern, TmpEleGPWDelta, TmpEleGPWInSum, TmpEleNum);
        StopTimer(60);
        if (UseOrbital) {
          StartTimer(61);
          //CalculateNewPfM2(mi,s,pfMNew,TmpEleIdx,qpStart,qpEnd);
          CalculateNewPfM2_real(mi, s, pfMNew_real, TmpEleIdx, qpStart, qpEnd);
          //printf("DEBUG: out %d in %d pfMNew=%lf \n",outStep,inStep,creal(pfMNew[0]));
          StopTimer(61);

          StartTimer(62);

          /* calculate inner product <phi|L|x> */
          //logIpNew = CalculateLogIP_fcmp(pfMNew,qpStart,qpEnd,comm);
          logIpNew = CalculateLogIP_real(pfMNew_real, qpStart, qpEnd, comm);
          StopTimer(62);
        }

        rbmValNew = RBMVal(TmpEleNum);

        /* Metroplis */
        x = LogProjRatio(projCntNew, TmpEleProjCnt);
        x += LogGPWRatio(eleGPWKernNew, TmpEleGPWKern);
        if (UseOrbital) {
          w = exp(2.0 * (x + (logIpNew - logIpOld)));
        }
        else {
          w = exp(2.0 * (x));
        }
        w *= pow(cabs(rbmValNew/rbmValOld),2);

        // printf("%f \n", cabs(rbmValNew/rbmValOld));
        if (!isfinite(w)) w = -1.0; /* should be rejected */

        if (w > genrand_real2()) { /* accept */
          // UpdateMAll will change SlaterElm, InvM (including PfM)
          StartTimer(63);
          if (UseOrbital) {
            UpdateMAll_real(mi, s, TmpEleIdx, qpStart, qpEnd);
            //            UpdateMAll(mi,s,TmpEleIdx,qpStart,qpEnd);
          }
          StopTimer(63);

          for (i = 0; i < NProj; i++) TmpEleProjCnt[i] = projCntNew[i];
          for (i = 0; i < NGPWIdx; i++) TmpEleGPWKern[i] = eleGPWKernNew[i];
          for (i = 0; i < Nsite*GPWTrnCfgSz;i ++) TmpEleGPWDelta[i] = eleGPWDeltaNew[i];
          for (i = 0; i < GPWTrnCfgSz*Nsite; i++) TmpEleGPWInSum[i] = eleGPWInSumNew[i];
          logIpOld = logIpNew;
          rbmValOld = rbmValNew;
          nAccept++;
          Counter[1]++;
        } else { /* reject */
          revertEleConfig(mi, ri, rj, s, TmpEleIdx, TmpEleCfg, TmpEleNum);
        }
        StopTimer(32);

      } else if (updateType == EXCHANGE) { /* exchange */
        Counter[2]++;

        StartTimer(31);
        makeCandidate_exchange(&mi, &ri, &rj, &s, &rejectFlag,
                               TmpEleIdx, TmpEleCfg, TmpEleNum);
        StopTimer(31);

        if (rejectFlag) continue;

        StartTimer(33);
        StartTimer(65);

        /* The mi-th electron with spin s exchanges with the electron on site rj with spin 1-s */
        t = 1 - s;
        mj = TmpEleCfg[rj + t * Nsite];

        /* The mi-th electron with spin s hops to rj */
        updateEleConfig(mi, ri, rj, s, TmpEleIdx, TmpEleCfg, TmpEleNum);
        UpdateProjCnt(ri, rj, s, projCntNew, TmpEleProjCnt, TmpEleNum);
        /* The mj-th electron with spin t hops to ri */
        updateEleConfig(mj, rj, ri, t, TmpEleIdx, TmpEleCfg, TmpEleNum);
        UpdateProjCnt(rj, ri, t, projCntNew, projCntNew, TmpEleNum);
        UpdateGPWKern(rj, ri, eleGPWKernNew, eleGPWDeltaNew, eleGPWInSumNew,
                      TmpEleGPWKern, TmpEleGPWDelta, TmpEleGPWInSum, TmpEleNum);

        StopTimer(65);
        if (UseOrbital) {
          StartTimer(66);

          CalculateNewPfMTwo2_real(mi, s, mj, t, pfMNew_real, TmpEleIdx, qpStart, qpEnd);
          StopTimer(66);
          StartTimer(67);

          /* calculate inner product <phi|L|x> */
          logIpNew = CalculateLogIP_real(pfMNew_real, qpStart, qpEnd, comm);
          StopTimer(67);
        }

        rbmValNew = RBMVal(TmpEleNum);

        /* Metroplis */
        x = LogProjRatio(projCntNew, TmpEleProjCnt);
        x += LogGPWRatio(eleGPWKernNew, TmpEleGPWKern);
        if (UseOrbital) {
          w = exp(2.0 * (x + (logIpNew - logIpOld)));
        }
        else {
          w = exp(2.0 * (x));
        }
        w *= pow(cabs(rbmValNew/rbmValOld),2);
        if (!isfinite(w)) w = -1.0; /* should be rejected */

        if (w > genrand_real2()) { /* accept */
          StartTimer(68);
          if (UseOrbital) {
            UpdateMAllTwo_real(mi, s, mj, t, ri, rj, TmpEleIdx, qpStart, qpEnd);
          }
          StopTimer(68);

          for (i = 0; i < NProj; i++) TmpEleProjCnt[i] = projCntNew[i];
          for (i = 0; i < NGPWIdx; i++) TmpEleGPWKern[i] = eleGPWKernNew[i];
          for(i = 0; i < Nsite*GPWTrnCfgSz; i++) TmpEleGPWDelta[i] = eleGPWDeltaNew[i];
          for(i = 0; i < GPWTrnCfgSz*Nsite; i++) TmpEleGPWInSum[i] = eleGPWInSumNew[i];
          logIpOld = logIpNew;
          rbmValOld = rbmValNew;
          nAccept++;
          Counter[3]++;
        } else { /* reject */
          revertEleConfig(mj, rj, ri, t, TmpEleIdx, TmpEleCfg, TmpEleNum);
          revertEleConfig(mi, ri, rj, s, TmpEleIdx, TmpEleCfg, TmpEleNum);
        }
        StopTimer(33);
      }

      if (nAccept > Nsite) {
        StartTimer(34);
        if (UseOrbital) {
          /* recal PfM and InvM */
          CalculateMAll_real(TmpEleIdx, qpStart, qpEnd);
          //printf("DEBUG: maker3: PfM=%lf\n",creal(PfM[0]));
          logIpOld = CalculateLogIP_real(PfM_real, qpStart, qpEnd, comm);
        }
        StopTimer(34);
        nAccept = 0;
      }
    } /* end of instep */

    StartTimer(35);
    /* save Electron Configuration */
    if (outStep >= nOutStep - NVMCSample) {
      sample = outStep - (nOutStep - NVMCSample);
      saveEleConfig(sample, logIpOld, rbmValOld, TmpEleIdx, TmpEleCfg, TmpEleNum,
                    TmpEleProjCnt, TmpEleGPWKern, TmpEleGPWDelta, TmpEleGPWInSum);
    }
    StopTimer(35);

  } /* end of outstep */

  copyToBurnSample(TmpEleIdx, TmpEleCfg, TmpEleNum, TmpEleProjCnt, TmpEleGPWKern,
                   TmpEleGPWDelta, TmpEleGPWInSum);
  BurnFlag = 1;

  free(eleGPWInSumNew);
  free(eleGPWDeltaNew); 

  return;
}

int makeInitialSample_real(int *eleIdx, int *eleCfg, int *eleNum, int *eleProjCnt, double *eleGPWKern,
                           int *eleGPWDelta, double *eleGPWInSum, const int qpStart, const int qpEnd,
                           MPI_Comm comm) {
  const int nsize = Nsize;
  const int nsite2 = Nsite2;
  int flag = 1, flagRdc, loop = 0;
  int ri, mi, si, msi, rsi;
  int rank, size;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  do {
    /* initialize */
#pragma omp parallel for default(shared) private(msi)
    for (msi = 0; msi < nsize; msi++) eleIdx[msi] = -1;
#pragma omp parallel for default(shared) private(rsi)
    for (rsi = 0; rsi < nsite2; rsi++) eleCfg[rsi] = -1;

    /* local spin */
    for (ri = 0; ri < Nsite; ri++) {
      if (LocSpn[ri] == 1) {
        do {
          mi = gen_rand32() % Ne;
          si = (genrand_real2() < 0.5) ? 0 : 1;
        } while (eleIdx[mi + si * Ne] != -1);
        eleCfg[ri + si * Nsite] = mi;
        eleIdx[mi + si * Ne] = ri;
      }
    }

    /* itinerant electron */
    for (si = 0; si < 2; si++) {
      for (mi = 0; mi < Ne; mi++) {
        if (eleIdx[mi + si * Ne] == -1) {
          do {
            ri = gen_rand32() % Nsite;
          } while (eleCfg[ri + si * Nsite] != -1 || LocSpn[ri] == 1);
          eleCfg[ri + si * Nsite] = mi;
          eleIdx[mi + si * Ne] = ri;
        }
      }
    }

    /* EleNum */
#pragma omp parallel for default(shared) private(rsi)
#pragma loop noalias
    for (rsi = 0; rsi < nsite2; rsi++) {
      eleNum[rsi] = (eleCfg[rsi] < 0) ? 0 : 1;
    }

    MakeProjCnt(eleProjCnt, eleNum);
    CalculateGPWKern(eleGPWKern, eleGPWDelta, eleGPWInSum, eleNum);

    if (UseOrbital) {
      flag = CalculateMAll_real(eleIdx, qpStart, qpEnd);
      //printf("DEBUG: maker4: PfM=%lf\n",creal(PfM[0]));
      if (size > 1) {
        MPI_Allreduce(&flag, &flagRdc, 1, MPI_INT, MPI_MAX, comm);
        flag = flagRdc;
      }
    }
    else {
      flag = -1;
    }

    loop++;
    if (loop > 100) {
      if (rank == 0) fprintf(stderr, "error: makeInitialSample: Too many loops\n");
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
  } while (flag > 0);

  return 0;
}

void VMC_BF_MakeSample_real(MPI_Comm comm) {
  int outStep, nOutStep;
  int inStep, nInStep;
  UpdateType updateType;
  int mi, mj, ri, rj, s, t, i;
  int nAccept = 0;
  int sample;
  int tmp_i;

  double logIpOld, logIpNew; /* logarithm of inner product <phi|L|x> */ // is this ok ? TBC
  int projCntNew[NProj];
  int projBFCntNew[16 * Nsite * Nrange]; // For BackFlow
  int msaTmp[NQPFull * Nsite], icount[NQPFull]; // For BackFlow
  double pfMNew_real[NQPFull];
  double x, w; // TBC x will be complex number

  int qpStart, qpEnd;
  int rejectFlag;
  int rank, size;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  SplitLoop(&qpStart, &qpEnd, NQPFull, rank, size);

  StartTimer(30);
  if (BurnFlag == 0) {
    makeInitialSampleBF(TmpEleIdx, TmpEleCfg, TmpEleNum, TmpEleProjCnt, TmpEleProjBFCnt,
                        qpStart, qpEnd, comm);
    //makeInitialSample(TmpEleIdx,TmpEleCfg,TmpEleNum,TmpEleProjCnt,
    //                  qpStart,qpEnd,comm);
  } else {
    //copyFromBurnSample(TmpEleIdx,TmpEleCfg,TmpEleNum,TmpEleProjCnt);
    copyFromBurnSampleBF(TmpEleIdx);
    MakeSlaterElmBF_fcmp(TmpEleNum, TmpEleProjBFCnt);
#pragma omp parallel for default(shared) private(tmp_i)
    for(tmp_i=0;tmp_i<NQPFull*(2*Nsite)*(2*Nsite);tmp_i++) SlaterElm_real[tmp_i]= creal(SlaterElm[tmp_i]);
  }

  CalculateMAll_BF_real(TmpEleIdx, qpStart, qpEnd);
  // printf("DEBUG: maker1: PfM=%lf\n",creal(PfM[0]));
  logIpOld = CalculateLogIP_real(PfM_real, qpStart, qpEnd, comm);
  if (!isfinite(logIpOld)) {
    if (rank == 0) fprintf(stderr, "waring: VMCMakeSample remakeSample logIpOld=%e\n", creal(logIpOld)); //TBC
    //    makeInitialSample(TmpEleIdx,TmpEleCfg,TmpEleNum,TmpEleProjCnt,
    //                    qpStart,qpEnd,comm);
    makeInitialSampleBF(TmpEleIdx, TmpEleCfg, TmpEleNum, TmpEleProjCnt, TmpEleProjBFCnt,
                        qpStart, qpEnd, comm);

    CalculateMAll_BF_real(TmpEleIdx, qpStart, qpEnd);
    //printf("DEBUG: maker2: PfM=%lf\n",creal(PfM[0]));
    logIpOld = CalculateLogIP_real(PfM_real, qpStart, qpEnd, comm);
    BurnFlag = 0;
  }
  StopTimer(30);

  nOutStep = (BurnFlag == 0) ? NVMCWarmUp + NVMCSample : NVMCSample + 1;
  nInStep = NVMCInterval * Nsite;

  for (i = 0; i < Counter_max; i++) Counter[i] = 0;  /* reset counter */

  for (outStep = 0; outStep < nOutStep; outStep++) {
    for (inStep = 0; inStep < nInStep; inStep++) {

      updateType = getUpdateType(NExUpdatePath);

      if (updateType == HOPPING) { /* hopping */
        Counter[0]++;

        StartTimer(31);
        makeCandidate_hopping(&mi, &ri, &rj, &s, &rejectFlag,
                              TmpEleIdx, TmpEleCfg);
        StopTimer(31);

        if (rejectFlag) continue;

        StartTimer(32);
        StartTimer(60);
        /* The mi-th electron with spin s hops to site rj */
        updateEleConfig(mi, ri, rj, s, TmpEleIdx, TmpEleCfg, TmpEleNum);
        UpdateProjCnt(ri, rj, s, projCntNew, TmpEleProjCnt, TmpEleNum);
        MakeProjBFCnt(projBFCntNew, TmpEleNum);
        StopTimer(60);
        UpdateSlaterElmBF_fcmp(mi, ri, rj, s, TmpEleCfg, TmpEleNum, projBFCntNew, msaTmp, icount,
                               SlaterElmBF);
#pragma omp parallel for default(shared) private(tmp_i)
        for(tmp_i=0;tmp_i<NQPFull*(2*Nsite)*(2*Nsite);tmp_i++) SlaterElm_real[tmp_i]= creal(SlaterElm[tmp_i]);

        StartTimer(61);
        //CalculateNewPfM2(mi,s,pfMNew,TmpEleIdx,qpStart,qpEnd);
        //CalculateNewPfM2_real(mi,s,pfMNew_real,TmpEleIdx,qpStart,qpEnd);
        CalculateNewPfMBF_real(icount, msaTmp, pfMNew_real, TmpEleIdx, qpStart, qpEnd, SlaterElmBF_real);

        //printf("DEBUG: out %d in %d pfMNew=%lf \n",outStep,inStep,creal(pfMNew[0]));
        StopTimer(61);

        StartTimer(62);
        /* calculate inner product <phi|L|x> */
        //logIpNew = CalculateLogIP_fcmp(pfMNew,qpStart,qpEnd,comm);
        logIpNew = CalculateLogIP_real(pfMNew_real, qpStart, qpEnd, comm);
        StopTimer(62);

        /* Metroplis */
        x = LogProjRatio(projCntNew, TmpEleProjCnt);
        w = exp(2.0 * (x + (logIpNew - logIpOld)));
        if (!isfinite(w)) w = -1.0; /* should be rejected */

        if (w > genrand_real2()) { /* accept */
          // UpdateMAll will change SlaterElm, InvM (including PfM)
          StartTimer(63);
          UpdateMAll_BF_real(icount, msaTmp, PfM_real, TmpEleIdx, qpStart, qpEnd);
          //          UpdateMAll_real(mi,s,TmpEleIdx,qpStart,qpEnd);
          //            UpdateMAll(mi,s,TmpEleIdx,qpStart,qpEnd);
          StopTimer(63);

          for (i = 0; i < NProj; i++) TmpEleProjCnt[i] = projCntNew[i];
          for (i = 0; i < 16 * Nsite * Nrange; i++) TmpEleProjBFCnt[i] = projBFCntNew[i];
          logIpOld = logIpNew;
          nAccept++;
          Counter[1]++;
        } else { /* reject */
          revertEleConfig(mi, ri, rj, s, TmpEleIdx, TmpEleCfg, TmpEleNum);
          //TODO: Add Timer
          UpdateSlaterElmBF_fcmp(mi, rj, ri, s, TmpEleCfg, TmpEleNum, TmpEleProjBFCnt, msaTmp, icount,
                                 SlaterElmBF);
#pragma omp parallel for default(shared) private(tmp_i)
          for(tmp_i=0;tmp_i<NQPFull*(2*Nsite)*(2*Nsite);tmp_i++) SlaterElm_real[tmp_i]= creal(SlaterElm[tmp_i]);
        }
        StopTimer(32);

      } else if (updateType == EXCHANGE) { /* exchange */
        Counter[2]++;

        StartTimer(31);
        makeCandidate_exchange(&mi, &ri, &rj, &s, &rejectFlag,
                               TmpEleIdx, TmpEleCfg, TmpEleNum);
        StopTimer(31);

        if (rejectFlag) continue;

        StartTimer(33);
        StartTimer(65);

        /* The mi-th electron with spin s exchanges with the electron on site rj with spin 1-s */
        t = 1 - s;
        mj = TmpEleCfg[rj + t * Nsite];

        /* The mi-th electron with spin s hops to rj */
        updateEleConfig(mi, ri, rj, s, TmpEleIdx, TmpEleCfg, TmpEleNum);
        UpdateProjCnt(ri, rj, s, projCntNew, TmpEleProjCnt, TmpEleNum);
        /* The mj-th electron with spin t hops to ri */
        updateEleConfig(mj, rj, ri, t, TmpEleIdx, TmpEleCfg, TmpEleNum);
        UpdateProjCnt(rj, ri, t, projCntNew, projCntNew, TmpEleNum);

        StopTimer(65);
        StartTimer(66);

        CalculateNewPfMTwo2_real(mi, s, mj, t, pfMNew_real, TmpEleIdx, qpStart, qpEnd);
        StopTimer(66);
        StartTimer(67);

        /* calculate inner product <phi|L|x> */
        logIpNew = CalculateLogIP_real(pfMNew_real, qpStart, qpEnd, comm);

        StopTimer(67);

        /* Metroplis */
        x = LogProjRatio(projCntNew, TmpEleProjCnt);
        w = exp(2.0 * (x + (logIpNew - logIpOld))); //TBC
        if (!isfinite(w)) w = -1.0; /* should be rejected */

        if (w > genrand_real2()) { /* accept */
          StartTimer(68);
          UpdateMAllTwo_real(mi, s, mj, t, ri, rj, TmpEleIdx, qpStart, qpEnd);
          StopTimer(68);

          for (i = 0; i < NProj; i++) TmpEleProjCnt[i] = projCntNew[i];
          logIpOld = logIpNew;
          nAccept++;
          Counter[3]++;
        } else { /* reject */
          revertEleConfig(mj, rj, ri, t, TmpEleIdx, TmpEleCfg, TmpEleNum);
          revertEleConfig(mi, ri, rj, s, TmpEleIdx, TmpEleCfg, TmpEleNum);
        }
        StopTimer(33);
      }

      if (nAccept > Nsite) {
        StartTimer(34);
        /* recal PfM and InvM */
        //CalculateMAll_real(TmpEleIdx,qpStart,qpEnd);
        //printf("DEBUG: maker3: PfM=%lf\n",creal(PfM[0]));
        CalculateMAll_BF_real(TmpEleIdx, qpStart, qpEnd);
        logIpOld = CalculateLogIP_real(PfM_real, qpStart, qpEnd, comm);
        StopTimer(34);
        nAccept = 0;
      }
    } /* end of instep */

    StartTimer(35);
    /* save Electron Configuration */
    if (outStep >= nOutStep - NVMCSample) {
      sample = outStep - (nOutStep - NVMCSample);
      saveEleConfigBF(sample, logIpOld, TmpEleIdx, TmpEleCfg, TmpEleNum, TmpEleProjCnt, TmpEleProjBFCnt);
      //      saveEleConfig(sample,logIpOld,TmpEleIdx,TmpEleCfg,TmpEleNum,TmpEleProjCnt);
    }
    StopTimer(35);

  } /* end of outstep */

  //copyToBurnSample(TmpEleIdx,TmpEleCfg,TmpEleNum,TmpEleProjCnt);
  copyToBurnSampleBF(TmpEleIdx);
  BurnFlag = 1;
  return;
}

int makeInitialSampleBF_real(int *eleIdx, int *eleCfg, int *eleNum, int *eleProjCnt, int *eleProjBFCnt,
                             const int qpStart, const int qpEnd, MPI_Comm comm) {
  const int nsize = Nsize;
  const int nsite2 = Nsite2;
  int flag = 1, flagRdc, loop = 0;
  int ri, mi, si, msi, rsi;
  int rank, size;
  int tmp_i;

  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  do {
    /* initialize */
#pragma omp parallel for default(shared) private(msi)
    for (msi = 0; msi < nsize; msi++) eleIdx[msi] = -1;
#pragma omp parallel for default(shared) private(rsi)
    for (rsi = 0; rsi < nsite2; rsi++) eleCfg[rsi] = -1;

    /* local spin */
    for (ri = 0; ri < Nsite; ri++) {
      if (LocSpn[ri] == 1) {
        do {
          mi = gen_rand32() % Ne;
          si = (genrand_real2() < 0.5) ? 0 : 1;
        } while (eleIdx[mi + si * Ne] != -1);
        eleCfg[ri + si * Nsite] = mi;
        eleIdx[mi + si * Ne] = ri;
      }
    }

    /* itinerant electron */
    for (si = 0; si < 2; si++) {
      for (mi = 0; mi < Ne; mi++) {
        if (eleIdx[mi + si * Ne] == -1) {
          do {
            ri = gen_rand32() % Nsite;
          } while (eleCfg[ri + si * Nsite] != -1 || LocSpn[ri] == 1);
          eleCfg[ri + si * Nsite] = mi;
          eleIdx[mi + si * Ne] = ri;
        }
      }
    }

    /* EleNum */
#pragma omp parallel for default(shared) private(rsi)
#pragma loop noalias
    for (rsi = 0; rsi < nsite2; rsi++) {
      eleNum[rsi] = (eleCfg[rsi] < 0) ? 0 : 1;
    }

    MakeProjCnt(eleProjCnt, eleNum);
    MakeProjBFCnt(eleProjBFCnt, eleNum);

    MakeSlaterElmBF_fcmp(eleNum, eleProjBFCnt);
#pragma omp parallel for default(shared) private(tmp_i)
    for(tmp_i=0;tmp_i<NQPFull*(2*Nsite)*(2*Nsite);tmp_i++) SlaterElm_real[tmp_i]= creal(SlaterElm[tmp_i]);

    flag = CalculateMAll_BF_real(eleIdx, qpStart, qpEnd);
    //printf("DEBUG: maker4: PfM=%lf\n",creal(PfM[0]));
    if (size > 1) {
      MPI_Allreduce(&flag, &flagRdc, 1, MPI_INT, MPI_MAX, comm);
      flag = flagRdc;
    }

    loop++;
    if (loop > 100) {
      if (rank == 0) fprintf(stderr, "error: makeInitialSample: Too many loops\n");
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
  } while (flag > 0);

  return 0;
}

#endif
