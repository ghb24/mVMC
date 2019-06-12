/*
mVMC - A numerical solver package for a wide range of quantum lattice models based on many-variable Variational Monte Carlo method
Copyright (C) 2016 The University of Tokyo, All rights reserved.

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
 * calculate physical quantities
 *-------------------------------------------------------------
 * by Satoshi Morita
 *-------------------------------------------------------------*/

void VMCMainCal_fsz(MPI_Comm comm, MPI_Comm commSampler);

void VMCMainCal_fsz(MPI_Comm comm, MPI_Comm commSampler) {
  int *eleIdx,*eleCfg,*eleNum,*eleProjCnt,*eleSpn; //fsz
  double *eleGPWKern;
  double complex innerSum, differential;
  double complex e,ip, amp;
  double w;
  double sqrtw;
  double complex we;
  double Sz;

  const int qpStart=0;
  const int qpEnd=NQPFull;
  int sample,sampleStart,sampleEnd,sampleSize;
  int i,info,j,offset,idx,f,targetBasis;

  /* optimazation for Kei */
  const int nProj=NProj;
  const int nGPWIdx=NGPWIdx;
  const int nRBMVis=RBMNVisibleIdx;
  const int nRBMHidden=RBMNHiddenIdx;
  const int nsite2=Nsite2;
  double complex *srOptO = SROptO;
//  double         *srOptO_real = SROptO_real;

  int rank,size,int_i,rankSampler;
  char fileNameSamples[D_FileNameMax];
  FILE *fp;
  unsigned long cfgUp, cfgDown;
  MPI_Comm_size(comm,&size);
  MPI_Comm_rank(comm,&rank);
  MPI_Comm_rank(commSampler, &rankSampler);
#ifdef __DEBUG_DETAILDETAIL
  printf("  Debug: SplitLoop\n");
#endif
  SplitLoop(&sampleStart,&sampleEnd,NVMCSample,rank,size);

  /* initialization */
  StartTimer(24);
  clearPhysQuantity();
  StopTimer(24);

  // set up samples file
  sprintf(fileNameSamples, "%s_Samples_%d.dat", CDataFileHead, rankSampler);
  fp = fopen(fileNameSamples, "w");

  for(sample=sampleStart;sample<sampleEnd;sample++) {
    eleIdx = EleIdx + sample*Nsize;
    eleCfg = EleCfg + sample*Nsite2;
    eleNum = EleNum + sample*Nsite2;
    eleProjCnt = EleProjCnt + sample*NProj;
    eleGPWKern = EleGPWKern + sample*NGPWIdx;
    eleSpn     = EleSpn + sample*Nsize; //fsz

    StartTimer(40);
    if (UseOrbital) {
#ifdef _DEBUG_DETAIL
      printf("  Debug: sample=%d: CalculateMAll \n",sample);
#endif
      info = CalculateMAll_fsz(eleIdx,eleSpn,qpStart,qpEnd);//info = CalculateMAll_fcmp(eleIdx,qpStart,qpEnd); // InvM,PfM will change
      StopTimer(40);

      if(info!=0) {
        fprintf(stderr,"warning: VMCMainCal rank:%d sample:%d info:%d (CalculateMAll)\n",rank,sample,info);
        continue;
      }
#ifdef _DEBUG_DETAIL
      printf("  Debug: sample=%d: CalculateIP \n",sample);
#endif
      ip = CalculateIP_fcmp(PfM,qpStart,qpEnd,MPI_COMM_SELF);
    }
    else {
      ip = 1.0;
    }
    ip *= RBMVal(eleNum);

#ifdef _DEBUG_DETAIL
    printf("  Debug: sample=%d: LogProjVal \n",sample);
#endif
    //LogProjVal(eleProjCnt);
    /* calculate reweight */
    w =1.0;
#ifdef _DEBUG_DETAIL
    printf("  Debug: sample=%d: isfinite \n",sample);
#endif
    if( !isfinite(w) ) {
      fprintf(stderr,"warning: VMCMainCal rank:%d sample:%d w=%e\n",rank,sample,w);
      continue;
    }

    StartTimer(41);
    /* calculate energy */
#ifdef _DEBUG_DETAIL
    printf("  Debug: sample=%d: calculateHam \n",sample);
#endif
#ifdef _DEBUG_DETAIL
    printf("  Debug: sample=%d: calculateHam_cmp \n",sample);
#endif
    e  = CalculateHamiltonian_fsz(ip,eleIdx,eleCfg,eleNum,eleProjCnt,eleGPWKern,eleSpn);//fsz
    Sz = CalculateSz_fsz(ip,eleIdx,eleCfg,eleNum,eleProjCnt,eleGPWKern,eleSpn);//fsz
    //printf("MDEBUG: Sz=%lf \n",Sz);
    //printf("MDEBUG: e= %lf %lf ip= %lf %lf \n",creal(e),cimag(e),creal(ip),cimag(ip));
    StopTimer(41);
    if( !isfinite(creal(e) + cimag(e)) ) {
      fprintf(stderr,"warning: VMCMainCal rank:%d sample:%d e=%e\n",rank,sample,creal(e)); //TBC
      continue;
    }

    // print out samples into file
    cfgUp = 0;
    cfgDown = 0;

    for (i = 0; i < Nsite; i++) {
      if (eleNum[i]) {
        cfgUp |= (((unsigned long) 1) << i);
      }

      if (eleNum[i+Nsite]) {
        cfgDown |= (((unsigned long) 1) << i);
      }
    }
    amp = cexp(LogProjVal(eleProjCnt)+LogGPWVal(eleGPWKern))*ip;

    fprintf(fp,"%lu  %lu  %f  %f  %f  %f  %f  %f\n", cfgUp, cfgDown,
            creal(amp), cimag(amp), creal(ip/RBMVal(eleNum)),
            cimag(ip/RBMVal(eleNum)), creal(e), cimag(e));

    Wc    += w;
    Etot  += w * e;
    Sztot += w * Sz;
    Sztot2 += w * Sz*Sz;
    Etot2 += w * conj(e) * e;
#ifdef _DEBUG_DETAIL
    printf("  Debug: sample=%d: calculateOpt \n",sample);
#endif
    if(NVMCCalMode==0) {
      /* Calculate O for correlation fauctors */
      srOptO[0] = 1.0+0.0*I;//   real
      srOptO[1] = 0.0+0.0*I;//   real
      #pragma loop noalias
      for(i=0;i<nProj;i++){
        srOptO[(i+1)*2]     = (double)(eleProjCnt[i]); // even real
        srOptO[(i+1)*2+1]   = 0.0+0.0*I;               // odd  comp
      }
      offset = nProj+1;

      #pragma loop noalias
      for(i=0;i<nGPWIdx;i++){
        srOptO[(offset+i)*2]     = eleGPWKern[i];    // even real
        srOptO[(offset+i)*2+1]   = eleGPWKern[i]*I;  // odd  comp
      }
      offset += nGPWIdx;

      #pragma omp parallel for default(shared) private(f, i)
      for(f = 0; f < nRBMVis; f++) {
        srOptO[(offset+f)*2]     = 0.0;    // even real
        srOptO[(offset+f)*2+1]   = 0.0;  // odd
        for(i = 0; i < Nsite2; i++) {
          srOptO[(offset+f)*2]     += (eleNum[i]*2-1)*Nsite;    // even real
          srOptO[(offset+f)*2+1]   += (eleNum[i]*2-1)*Nsite*I;  // odd
        }
      }
      offset += nRBMVis;

      for(f = 0; f < nRBMHidden; f++) {
        differential = 0;
        #pragma omp parallel for default(shared) private(i, innerSum) reduction(+:differential)
        for(i = 0; i < Nsite; i++) {
          innerSum = RBMHiddenLayerSum(f, i, eleNum);
          differential += (cexp(innerSum) - cexp(-innerSum))/(cexp(innerSum) + cexp(-innerSum));
        }

        srOptO[(offset+f)*2]     = differential;    // even real
        srOptO[(offset+f)*2+1]   = differential*I;  // odd  comp



        for(j = 0; j < Nsite; j++) {
          differential = 0;

          #pragma omp parallel for default(shared) private(i, targetBasis, innerSum) reduction(+:differential)
          for(i = 0; i < Nsite; i++) {
            targetBasis = (j + i) % Nsite;
            innerSum = RBMHiddenLayerSum(f, i, eleNum);
            differential += (eleNum[targetBasis]*2-1)*(cexp(innerSum) - cexp(-innerSum))/(cexp(innerSum) + cexp(-innerSum));
          }
          idx = RBMWeightMatrIdx[j][f];
          srOptO[(offset+nRBMHidden+idx)*2]     = differential;    // even real
          srOptO[(offset+nRBMHidden+idx)*2+1]   = differential*I;  // odd  comp



          #pragma omp parallel for default(shared) private(i, targetBasis, innerSum) reduction(+:differential)
          for(i = 0; i < Nsite; i++) {
            targetBasis = (j + i) % Nsite + Nsite;
            innerSum = RBMHiddenLayerSum(f, i, eleNum);
            differential += (eleNum[targetBasis]*2-1)*(cexp(innerSum) - cexp(-innerSum))/(cexp(innerSum) + cexp(-innerSum));
          }
          idx = RBMWeightMatrIdx[j+Nsite][f];
          srOptO[(offset+nRBMHidden+idx)*2]     = differential;    // even real
          srOptO[(offset+nRBMHidden+idx)*2+1]   = differential*I;  // odd  comp
        }
      }
      offset += nsite2*nRBMHidden+nRBMHidden;

      StartTimer(42);
      /* SlaterElmDiff */
      SlaterElmDiff_fsz(SROptO+2*offset,ip,eleIdx,eleSpn) ;//SlaterElmDiff_fcmp(SROptO+2*NProj+2,ip,eleIdx); //TBC: using InvM not InvM_real
      StopTimer(42);

      offset += NSlater;

      if(FlagOptTrans>0) { // this part will be not used
        calculateOptTransDiff(SROptO+2*offset, ip); //TBC
      }
      StartTimer(43);
      /* Calculate OO and HO */
      if(NSRCG==0 && NStoreO==0){
        calculateOO(SROptOO,SROptHO,SROptO,w,e,SROptSize);
      }else{
        we    = w*e;
        sqrtw = sqrt(w);
        #pragma omp parallel for default(shared) private(int_i)
        for(int_i=0;int_i<SROptSize*2;int_i++){
        // SROptO_Store for fortran
          SROptO_Store[int_i+sample*(2*SROptSize)]  = sqrtw*SROptO[int_i];
          SROptHO[int_i]                           += we*SROptO[int_i];
        }
      }
      StopTimer(43);
    } else if(NVMCCalMode==1) {
      StartTimer(42);
      /* Calculate Green Function */
      CalculateGreenFunc_fsz(w,ip,eleIdx,eleCfg,eleNum,eleSpn,eleProjCnt,eleGPWKern);
      StopTimer(42);

      if(NLanczosMode>0){
        // for sz!=0, Lanczso is not supported
      }
    }
  } /* end of for(sample) */
  fclose(fp);

// calculate OO and HO at NVMCCalMode==0
  if(NVMCCalMode==0){
    if(NStoreO!=0 || NSRCG!=0){
      sampleSize=sampleEnd-sampleStart;
      StartTimer(45);
      calculateOO_Store(SROptOO,SROptHO,SROptO_Store,w,e,2*SROptSize,sampleSize);
      StopTimer(45);
    }
  }
  return;
}
