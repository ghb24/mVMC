/*
mVMC - A numerical solver package for a wide range of quantum lattice models based on many-variable Variational Monte Carlo method
Copyright (C) 2016 Takahiro Misawa, Satoshi Morita, Takahiro Ohgoe, Kota Ido, Mitsuaki Kawamura, Takeo Kato, Masatoshi Imada.

his program is developed based on the mVMC-mini program
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

void VMCMakeSample_fsz(MPI_Comm comm);
int makeInitialSample_fsz(int *eleIdx, int *eleCfg, int *eleNum, int *eleProjCnt,int *eleSpn,
                      const int qpStart, const int qpEnd, MPI_Comm comm);
void copyFromBurnSample_fsz(int *eleIdx, int *eleCfg, int *eleNum, int *eleProjCnt,int *eleSpn);
void copyToBurnSample_fsz(const int *eleIdx, const int *eleCfg, const int *eleNum, const int *eleProjCnt,const int *eleSpn);
void saveEleConfig_fsz(const int sample, const double complex logIp,
                   const int *eleIdx, const int *eleCfg, const int *eleNum, const int *eleProjCnt,const int *eleSpn);
//void sortEleConfig(int *eleIdx, int *eleCfg, const int *eleNum);
//void ReduceCounter(MPI_Comm comm);
void makeCandidate_hopping_fsz(int *mi_, int *ri_, int *rj_, int *s_,int *t_, int *rejectFlag_,
                           const int *eleIdx, const int *eleCfg,const int *eleNum,const int *eleSpn);

void makeCandidate_exchange_fsz(int *mi_, int *ri_, int *rj_, int *s_, int *rejectFlag_,
                            const int *eleIdx, const int *eleCfg, const int *eleNum,const int *eleSpn);
void updateEleConfig_fsz(int mi, int org_r, int dst_r, int org_spn,int dst_spn,
                     int *eleIdx, int *eleCfg, int *eleNum, int *eleSpn) ;
void revertEleConfig_fsz(int mi, int org_ri, int dst_r, int org_spn,int dst_spn,
                     int *eleIdx, int *eleCfg, int *eleNum,int *eleSpn);
void CheckEleConfig_fsz(int *eleIdx, int *eleCfg, int *eleNum,int *eleSpn,MPI_Comm comm);
//typedef enum {HOPPING, EXCHANGE, NONE} UpdateType;
//UpdateType getUpdateType(int path);

void VMCMakeSample_fsz(MPI_Comm comm) {
  int outStep,nOutStep;
  int inStep,nInStep;
  UpdateType updateType;
  int mi,mj,ri,rj,s,t,i;
  int nAccept=0;
  int sample;

  double complex logIpOld,logIpNew; /* logarithm of inner product <phi|L|x> */ // is this ok ? TBC
  int projCntNew[NProj];
  double complex pfMNew[NQPFull];
  double x,w; // TBC x will be complex number

  int qpStart,qpEnd;
  int rejectFlag;
  int rank,size;
  int tmp_mi,tmp_ri,tmp_num;//fsz DEBUG
  MPI_Comm_size(comm,&size);
  MPI_Comm_rank(comm,&rank);

  SplitLoop(&qpStart,&qpEnd,NQPFull,rank,size);

  StartTimer(30);
  if(BurnFlag==0) {
    //printf("DEBUG: make1: \n");
    makeInitialSample_fsz(TmpEleIdx,TmpEleCfg,TmpEleNum,TmpEleProjCnt,TmpEleSpn,
                      qpStart,qpEnd,comm);
//DEBUG
    CheckEleConfig_fsz(TmpEleIdx,TmpEleCfg,TmpEleNum,TmpEleSpn,comm);
//DEBUG
  } else {
    copyFromBurnSample_fsz(TmpEleIdx,TmpEleCfg,TmpEleNum,TmpEleProjCnt,TmpEleSpn) ;//fsz
  }
  
  CalculateMAll_fsz(TmpEleIdx,TmpEleSpn,qpStart,qpEnd);
 // printf("DEBUG: maker1: PfM=%lf\n",creal(PfM[0]));
  logIpOld = CalculateLogIP_fcmp(PfM,qpStart,qpEnd,comm);
  if( !isfinite(creal(logIpOld) + cimag(logIpOld)) ) {
    if(rank==0) fprintf(stderr,"waring: VMCMakeSample remakeSample logIpOld=%e\n",creal(logIpOld)); //TBC
    makeInitialSample_fsz(TmpEleIdx,TmpEleCfg,TmpEleNum,TmpEleProjCnt,TmpEleSpn,
                      qpStart,qpEnd,comm);
    CalculateMAll_fsz(TmpEleIdx,TmpEleSpn,qpStart,qpEnd);
    //printf("DEBUG: maker2: PfM=%lf\n",creal(PfM[0]));
    logIpOld = CalculateLogIP_fcmp(PfM,qpStart,qpEnd,comm);
    BurnFlag = 0;
  }
  StopTimer(30);

  nOutStep = (BurnFlag==0) ? NVMCWarmUp+NVMCSample : NVMCSample+1;
  nInStep = NVMCInterval * Nsite;

  for(i=0;i<4;i++) Counter[i]=0;  /* reset counter */

  for(outStep=0;outStep<nOutStep;outStep++) {
    for(inStep=0;inStep<nInStep;inStep++) {
//DEBUG
      CheckEleConfig_fsz(TmpEleIdx,TmpEleCfg,TmpEleNum,TmpEleSpn,comm);
//DEBUG
      updateType = getUpdateType(NExUpdatePath);

      if(updateType==HOPPING) { /* hopping */
        Counter[0]++;

        StartTimer(31);
        makeCandidate_hopping_fsz(&mi, &ri, &rj, &s,&t, &rejectFlag,
                              TmpEleIdx, TmpEleCfg,TmpEleNum,TmpEleSpn);
        StopTimer(31);

        if(rejectFlag) continue; 

        StartTimer(32);
          StartTimer(60);
        /* The mi-th electron with spin s hops to site rj with t */
        updateEleConfig_fsz(mi,ri,rj,s,t,TmpEleIdx,TmpEleCfg,TmpEleNum,TmpEleSpn);
        if(s==t){
          UpdateProjCnt(ri,rj,s,projCntNew,TmpEleProjCnt,TmpEleNum);
        }else{
          UpdateProjCnt_fsz(ri,rj,s,t,projCntNew,TmpEleProjCnt,TmpEleNum);
        }   
          StopTimer(60);
          StartTimer(61);
        CalculateNewPfM2_fsz(mi,t,pfMNew,TmpEleIdx,TmpEleSpn,qpStart,qpEnd); // fsz: s->t 
          StopTimer(61);

          StartTimer(62);
        /* calculate inner product <phi|L|x> */
        //logIpNew = CalculateLogIP_fcmp(pfMNew,qpStart,qpEnd,comm);
        logIpNew = CalculateLogIP_fcmp(pfMNew,qpStart,qpEnd,comm);
          StopTimer(62);

        /* Metroplis */
        x = LogProjRatio(projCntNew,TmpEleProjCnt);
        w = exp(2.0*(x+creal(logIpNew-logIpOld)));
        if( !isfinite(w) ) w = -1.0; /* should be rejected */

        if(w > genrand_real2()) { /* accept */
            // UpdateMAll will change SlaterElm, InvM (including PfM)
            StartTimer(63);
            UpdateMAll_fsz(mi,t,TmpEleIdx,TmpEleSpn,qpStart,qpEnd); // fsz : s->t
            StopTimer(63);

          for(i=0;i<NProj;i++) TmpEleProjCnt[i] = projCntNew[i];
          logIpOld = logIpNew;
          nAccept++;
          Counter[1]++;
        } else { /* reject */ //(ri,s) <- (rj,t)
          revertEleConfig_fsz(mi,ri,rj,s,t,TmpEleIdx,TmpEleCfg,TmpEleNum,TmpEleSpn);
        }
        StopTimer(32);
      } else if(updateType==EXCHANGE) { /* exchange */
        Counter[2]++;

        StartTimer(31);
        makeCandidate_exchange_fsz(&mi, &ri, &rj, &s, &rejectFlag,
                               TmpEleIdx, TmpEleCfg, TmpEleNum,TmpEleSpn);
        StopTimer(31);
        if(rejectFlag) continue;

        StartTimer(33);
        StartTimer(65);

        /* The mi-th electron with spin s exchanges with the electron on site rj with spin 1-s */
        t = 1-s;
        mj = TmpEleCfg[rj+t*Nsite];

        /* The mi-th electron with spin s hops to rj */
        updateEleConfig_fsz(mi,ri,rj,s,s,TmpEleIdx,TmpEleCfg,TmpEleNum,TmpEleSpn);
        UpdateProjCnt(ri,rj,s,projCntNew,TmpEleProjCnt,TmpEleNum);
        /* The mj-th electron with spin t hops to ri */
        updateEleConfig_fsz(mj,rj,ri,t,t,TmpEleIdx,TmpEleCfg,TmpEleNum,TmpEleSpn);
        UpdateProjCnt(rj,ri,t,projCntNew,projCntNew,TmpEleNum);




        StopTimer(65);
        StartTimer(66);

        CalculateNewPfMTwo2_fsz(mi, s, mj, t, pfMNew, TmpEleIdx,TmpEleSpn, qpStart, qpEnd);
        StopTimer(66);
        StartTimer(67);

        /* calculate inner product <phi|L|x> */
        logIpNew = CalculateLogIP_fcmp(pfMNew,qpStart,qpEnd,comm);

        StopTimer(67);

        /* Metroplis */
        x = LogProjRatio(projCntNew,TmpEleProjCnt);
        w = exp(2.0*(x+creal(logIpNew-logIpOld))); //TBC
        if( !isfinite(w) ) w = -1.0; /* should be rejected */

        if(w > genrand_real2()) { /* accept */
          StartTimer(68);
          UpdateMAllTwo_fsz(mi, s, mj, t, ri, rj, TmpEleIdx,TmpEleSpn,qpStart,qpEnd);
          StopTimer(68);

          for(i=0;i<NProj;i++) TmpEleProjCnt[i] = projCntNew[i];
          logIpOld = logIpNew;
          nAccept++;
          Counter[3]++;
        } else { /* reject */
          revertEleConfig_fsz(mj,rj,ri,t,t,TmpEleIdx,TmpEleCfg,TmpEleNum,TmpEleSpn);
          revertEleConfig_fsz(mi,ri,rj,s,s,TmpEleIdx,TmpEleCfg,TmpEleNum,TmpEleSpn);
        }
        StopTimer(33);
      }

      if(nAccept>Nsite) {
        StartTimer(34);
        /* recal PfM and InvM */
        CalculateMAll_fsz(TmpEleIdx,TmpEleSpn,qpStart,qpEnd);
        //printf("DEBUG: maker3: PfM=%lf\n",creal(PfM[0]));
        logIpOld = CalculateLogIP_fcmp(PfM,qpStart,qpEnd,comm);
        StopTimer(34);
        nAccept=0;
      }
    } /* end of instep */

    StartTimer(35);
    /* save Electron Configuration */
    if(outStep >= nOutStep-NVMCSample) {
      sample = outStep-(nOutStep-NVMCSample);
      saveEleConfig_fsz(sample,logIpOld,TmpEleIdx,TmpEleCfg,TmpEleNum,TmpEleProjCnt,TmpEleSpn);
    }
    StopTimer(35);

  } /* end of outstep */

  copyToBurnSample_fsz(TmpEleIdx,TmpEleCfg,TmpEleNum,TmpEleProjCnt,TmpEleSpn);
  BurnFlag=1;
  return;
}

int makeInitialSample_fsz(int *eleIdx, int *eleCfg, int *eleNum, int *eleProjCnt,int *eleSpn,
                      const int qpStart, const int qpEnd, MPI_Comm comm) {
  const int nsize = Nsize;
  const int nsite2 = Nsite2;
  int flag=1,flagRdc,loop=0;
  int ri,mi,si,msi,rsi;
  int rank,size;
  MPI_Comm_size(comm,&size);
  MPI_Comm_rank(comm,&rank);
  
  do {
    /* initialize */
    #pragma omp parallel for default(shared) private(msi)
    for(msi=0;msi<nsize;msi++) eleIdx[msi] = -1;
    #pragma omp parallel for default(shared) private(msi)
    for(msi=0;msi<nsize;msi++) eleSpn[msi] = -1;
    #pragma omp parallel for default(shared) private(rsi)
    for(rsi=0;rsi<nsite2;rsi++) eleCfg[rsi] = -1;
    
    /* local spin */
    //note: Sz is not conserved quantity but initially we take Sz=0 
    for(ri=0;ri<Nsite;ri++) {
      if(LocSpn[ri]==1) {
        do {
          //mi = gen_rand32()%nsize;
          mi = gen_rand32()%Ne;
          si = (genrand_real2()<0.5) ? 0 : 1;
        } while(eleIdx[mi+si*Ne]!=-1); // seeking empty site
        eleCfg[ri+si*Nsite] = mi+si*Ne;
        eleIdx[mi+si*Ne]    = ri;
        eleSpn[mi+si*Ne]    = si;
      }
    }
    ///for(mi=0;mi<2*Ne;mi++){
    //  printf("DEBUG: %d %d %d\n",mi,eleIdx[mi],eleSpn[mi]);
    //}
    
    /* itinerant electron */
    for(si=0;si<2;si++) { //note: Sz is not conserved quantity but initially we take Sz=0 
      for(mi=0;mi<Ne;mi++) { 
        if(eleIdx[mi+si*Ne]== -1) {
          do {
            ri = gen_rand32()%Nsite;
          } while (eleCfg[ri+si*Nsite]!= -1 || LocSpn[ri]==1); // seeking empty and itinerant site
          eleCfg[ri+si*Nsite]     = mi+si*Ne; // buggged 4/26
          eleIdx[mi+si*Ne]        = ri;
          eleSpn[mi+si*Ne]        = si;
        }
      }
    }
    
    /* EleNum */
    #pragma omp parallel for default(shared) private(rsi)
    #pragma loop noalias
    for(rsi=0;rsi<nsite2;rsi++) {
      eleNum[rsi] = (eleCfg[rsi] < 0) ? 0 : 1;
    }
    
    MakeProjCnt(eleProjCnt,eleNum); // this function does not change even for fsz

    flag = CalculateMAll_fsz(eleIdx,eleSpn,qpStart,qpEnd);
    //printf("DEBUG: make4: PfM=%lf\n",creal(PfM[0]));
    if(size>1) {
      MPI_Allreduce(&flag,&flagRdc,1,MPI_INT,MPI_MAX,comm);
      flag = flagRdc;
    }

    loop++;
    if(loop>100) {
      if(rank==0) fprintf(stderr, "error: makeInitialSample: Too many loops\n");
      MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE);
    }
  } while (flag>0);
 
  return 0;
}

void copyFromBurnSample_fsz(int *eleIdx, int *eleCfg, int *eleNum, int *eleProjCnt,int *eleSpn) {
  int i,n;
  const int *burnEleIdx = BurnEleIdx;// BurnEleIdx is global
  n = Nsize + 2*Nsite + 2*Nsite + NProj+Nsite;//fsz
  #pragma loop noalias
  for(i=0;i<n;i++) eleIdx[i] = burnEleIdx[i]; 
  return;
}

void copyToBurnSample_fsz(const int *eleIdx, const int *eleCfg, const int *eleNum, const int *eleProjCnt,const int *eleSpn) {
  int i,n;
  int *burnEleIdx = BurnEleIdx;
  n = Nsize + 2*Nsite + 2*Nsite + NProj+Nsite;//fsz
  #pragma loop noalias
  for(i=0;i<n;i++) burnEleIdx[i] = eleIdx[i];
  return;
}

void saveEleConfig_fsz(const int sample, const double complex logIp,
                   const int *eleIdx, const int *eleCfg, const int *eleNum, const int *eleProjCnt,const int *eleSpn) {
  int i,offset;
  double x;
  const int nsize=Nsize;
  const int nsite2 = Nsite2;
  const int nProj = NProj;

  offset = sample*nsize;
  #pragma loop noalias
  for(i=0;i<nsize;i++) EleIdx[offset+i] = eleIdx[i];
  offset = sample*nsite2;
  #pragma loop noalias
  for(i=0;i<nsite2;i++) EleCfg[offset+i] = eleCfg[i];
  #pragma loop noalias
  for(i=0;i<nsite2;i++) EleNum[offset+i] = eleNum[i];
  offset = sample*nProj;
  #pragma loop noalias
  for(i=0;i<nProj;i++) EleProjCnt[offset+i] = eleProjCnt[i];
  offset = sample*nsize;
  #pragma loop noalias
  for(i=0;i<nsize;i++) EleSpn[offset+i] = eleSpn[i];
  
  x = LogProjVal(eleProjCnt);
  logSqPfFullSlater[sample] = 2.0*(x+creal(logIp));//TBC
  
  return;
}

//void sortEleConfig(int *eleIdx, int *eleCfg, const int *eleNum) {
/*   int ri,mi=0; */
/*   for(ri=0;ri<Nsite;ri++) { */
/*     if(eleNum[ri]>0) { */
/*       eleCfg[ri]=mi; */
/*       eleIdx[mi]=ri; */
/*       mi++; */
/*     } else { */
/*       eleCfg[ri]=-1; */
/*     } */
/*   } */
/*   mi=0; */
/*   for(ri=0;ri<Nsite;ri++) { */
/*     if(eleNum[ri+Nsite]>0) { */
/*       eleCfg[ri+Nsite]=mi; */
/*       eleIdx[mi+Ne]=ri; */
/*       mi++; */
/*     } else { */
/*       eleCfg[ri+Nsite]=-1; */
/*     } */
/*   } */

//  return;
//}
/*
void ReduceCounter(MPI_Comm comm) {
  #ifdef _mpi_use
  int n=4;
  int recv[n];
  int i;
  int rank,size;
  MPI_Comm_size(comm,&size);
  MPI_Comm_rank(comm,&rank);

  MPI_Allreduce(Counter,recv,n,MPI_INT,MPI_SUM,comm);
  if(rank==0) {
    for(i=0;i<n;i++) Counter[i] = recv[i];
  }
  #endif
  return;
}
*/

// mi (ri,s) -> mi (rj,t)
void makeCandidate_hopping_fsz(int *mi_, int *ri_, int *rj_, int *s_,int *t_, int *rejectFlag_,
                           const int *eleIdx, const int *eleCfg,const int *eleNum,const int *eleSpn) {
  const int icnt_max = Nsite*Nsite;
  int icnt;
  int mi, ri, rj, s, flag,tmp_rj;
  int t; //fsz

  flag = 0; // FALSE
  do {
    mi = gen_rand32()%Nsize;
    s  = eleSpn[mi] ; //fsz 
    t  = (genrand_real2()<0.5) ? s : 1-s; //fsz
    ri = eleIdx[mi];  //fsz
  } while (LocSpn[ri] == 1);

  icnt = 0;
  do {
    rj = gen_rand32()%Nsite;
    if(icnt> icnt_max){
      flag = 1; // TRUE
      break;
    }
    icnt+=1;
  } while (eleCfg[rj+t*Nsite] != -1 || LocSpn[rj]==1);

  *mi_ = mi;
  *ri_ = ri;
  *rj_ = rj;
  *s_  = s;
  *t_  = t;
  *rejectFlag_ = flag;

  return;
}


/* The mi-th electron with spin s exchanges with the electron on site rj with spin 1-s */
void makeCandidate_exchange_fsz(int *mi_, int *ri_, int *rj_, int *s_, int *rejectFlag_,
                           const int *eleIdx, const int *eleCfg, const int *eleNum,const int *eleSpn) {
  int mi, mj, ri, rj, s, t, flag,spn_0,spn_1;

// DEBUG!!!!!!!!!!!!!!!!!!!!!
/*
  for(mi=0;mi<Nsize;mi++){
    printf("XDEBUG: mi=%d spn=%d idx=%d\n",mi,eleSpn[mi],eleIdx[mi]);
  }
  for(ri=0;ri<Nsite;ri++){
    printf("XDEBUG: ri=%d up=%d down=%d\n",ri,eleNum[ri],eleNum[ri+Nsite]);
  }
*/
// DEBUG!!!!!!!!!!!!!!!!!!!!!

  flag = 1; // TRUE
  spn_0 = 0;//
  spn_1 = 0;//
  for(ri=0;ri<Nsite;ri++){
    if((eleNum[ri]+eleNum[ri+Nsite]) == 1  ){// up or down exists
      if(spn_0==0){
        spn_0  = 2*eleNum[ri]-1;// 0 (up)-> 1, 1(down)-> -1
      }else{
        spn_1 =  2*eleNum[ri]-1;// 0 (up)-> 1, 1(down)-> -1
      }
      //printf("ri =%d %d %d : spn0 %d spn1 %d\n",ri,eleNum[ri],eleNum[ri+Nsite],spn_0,spn_1);
      if(spn_0*spn_1<0){
        flag = 0; // FALSE
        break;
      }
    }
  }
  //printf("flag= %d spn_0=%d spn_1=%d \n",flag,spn_0,spn_1);
  if(flag) {
    *rejectFlag_ = flag;
    return;
  }

  do {
    mi = gen_rand32()%Nsize;//fsz
    s  = eleSpn[mi];// fsz //s = (genrand_real2()<0.5) ? 0 : 1;
    ri = eleIdx[mi]; //fsz
  } while (eleCfg[ri+(1-s)*Nsite] != -1);
  t = 1-s;
  do {
    mj = gen_rand32()%Nsize; //fsz
    rj = eleIdx[mj]; //fsz
  } while (eleCfg[rj+(1-t)*Nsite] != -1 || eleSpn[mj]!=t); // is it OK ?

  *mi_ = mi;
  *ri_ = ri;
  *rj_ = rj;
  *s_ = s;
  *rejectFlag_ = flag;
  return;
}

/* The mi-th electron with spin s hops to site rj and t */
void updateEleConfig_fsz(int mi, int org_r, int dst_r, int org_spn,int dst_spn,
                     int *eleIdx, int *eleCfg, int *eleNum, int *eleSpn) {
  eleIdx[mi]         = dst_r; 
  eleSpn[mi]         = dst_spn;  //fsz 
//
  eleCfg[org_r+org_spn*Nsite] = -1;
  eleCfg[dst_r+dst_spn*Nsite] = mi;
//
  eleNum[org_r+org_spn*Nsite] = 0;
  eleNum[dst_r+dst_spn*Nsite] = 1;
  return;
}

void revertEleConfig_fsz(int mi, int org_r, int dst_r, int org_spn,int dst_spn,
                     int *eleIdx, int *eleCfg, int *eleNum,int *eleSpn) {
  eleIdx[mi]         = org_r; 
  eleSpn[mi]         = org_spn; //fsz 
//
  eleCfg[org_r+org_spn*Nsite] = mi;
  eleCfg[dst_r+dst_spn*Nsite] = -1;
//
  eleNum[org_r+org_spn*Nsite] = 1;
  eleNum[dst_r+dst_spn*Nsite] = 0;
  return;
}


void CheckEleConfig_fsz(int *eleIdx, int *eleCfg, int *eleNum,int *eleSpn,MPI_Comm comm){
   int mi,ri,si;
   int check_ri,check_si;
   int rank;
   MPI_Comm_rank(comm,&rank);
   
   for(ri=0;ri<Nsite;ri++){
     for(si=0;si<2;si++){
       mi = eleCfg[ri+si*Nsite];
       if(mi>=0){
         check_ri = eleIdx[mi];
         check_si = eleSpn[mi];
         if(ri!=check_ri || si!=check_si){
           if(rank==0) fprintf(stderr, "error: vmcmakesample: fatal error in making sample: mi %d :ri %d %d: si %d %d\n",mi,ri,check_ri,si,check_si);
           MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE);
         }
       }
     }
   }
}
