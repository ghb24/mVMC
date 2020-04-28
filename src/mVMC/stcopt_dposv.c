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
 * Stochastic Reconfiguration method by DPOSV
 *-------------------------------------------------------------
 * by Satoshi Morita
 *-------------------------------------------------------------*/
#include "stcopt_dposv.h"
#ifndef _SRC_STCOPT_DPOSV
#define _SRC_STCOPT_DPOSV

/* calculate the parameter change r[nSmat] from SOpt.*/
int stcOptMain(double *g, const int nSmat, const int * const smatToParaIdx, MPI_Comm comm, FILE *fp)
{
  /* for DPOSV */
  char uplo, jobz, trans;
  int n,lds,lwork,info,cutIdx,ldg,nrhs,i,j,k,inca,incy;
  double *S, sum, *eigVals, *eigVecs, workSize, *work, *alpha, *y, diagCutThreshold, tmp, prefactor, beta;
  int rank;

  MPI_Comm_rank(comm,&rank);

  StartTimer(53);

  S = (double*)calloc(nSmat*nSmat, sizeof(double));
  stcOptInit(S, g, nSmat, smatToParaIdx);

  if (RedCutMode == 1) {
    eigVals = (double*) malloc(sizeof(double) * nSmat * 2);
    alpha = eigVals + nSmat;
    memcpy(alpha, g, sizeof(double)*nSmat);

    jobz='V'; uplo='U'; n=nSmat; lds=n;

    // get workspace size
    lwork = -1;
    M_DSYEV(&jobz, &uplo, &n, S, &lds, eigVals, &workSize, &lwork, &info);

    // diagonalise S
    lwork = (int) workSize;
    work = (double*) malloc(sizeof(double) * lwork);
    M_DSYEV(&jobz, &uplo, &n, S, &lds, eigVals, work, &lwork, &info);

    diagCutThreshold = DSROptRedCut * eigVals[nSmat-1];

    cutIdx = 0;
    while (eigVals[cutIdx] < diagCutThreshold) {
      cutIdx++;
    }

    trans='T';
    n = nSmat;
    prefactor = 1.0;
    beta = 0.0;
    inca = incy = 1;

    //compute g = U^T * alpha
    M_DGEMV(&trans, &n, &n, &prefactor, S, &n, alpha, &inca, &beta, g, &incy);

    // cut non-relevant directions and set alpha = 1/lambda * (U^T * alpha)
    #pragma omp parallel for default(shared) private(i)
    for (i = 0; i < nSmat; i++) {
      if(i < cutIdx) {
        alpha[i] = 0.0;
      }
      else {
        alpha[i] = g[i]/eigVals[i];
      }
    }

    trans='N';
    M_DGEMV(&trans, &n, &n, &prefactor, S, &n, alpha, &inca, &beta, g, &incy);


    if (rank == 0) {
      fprintf(fp, "%e, %e, %d \n", eigVals[nSmat-1], eigVals[0], cutIdx);
    }

    free(work);
    free(eigVals);
  }
  else {
    uplo='U'; n=nSmat; nrhs=1; lds=n; ldg=n;
    M_DPOSV(&uplo, &n, &nrhs, S, &lds, g, &ldg, &info);
  }


  free(S);
  StopTimer(53);


  return info;

}

/* calculate and store S and g in the equation to be solved, Sx=g */
void stcOptInit(double *const S, double *const g, const int nSmat, const int *const smatToParaIdx) {
  const double ratioDiag = 1.0 + DSROptStaDel;
  int si,sj,pi,pj,idx,offset;
  double tmp;
  
  /* calculate the overlap matrix S */
  /* S[i][j] = OO[i+1][j+1] - OO[0][i+1] * OO[0][j+1]; */
  for(si=0;si<nSmat;++si) {
    pi = smatToParaIdx[si];
    //offset = (pi+1)*SROptSize;
    offset = (pi+2)*(2*SROptSize);
    tmp = creal(SROptOO[pi+2]);

    for(sj=0;sj<nSmat;++sj) {
      pj = smatToParaIdx[sj];
      idx = si + nSmat*sj; /* column major */
      S[idx] = creal(SROptOO[offset+(pj+2)]) - tmp * creal(SROptOO[pj+2]);
    }

    /* modify diagonal elements */
    idx = si + nSmat*si;
    S[idx] *= ratioDiag;
  }

  /* calculate the energy gradient * (-dt) */
  /* energy gradient = 2.0*( HO[i+1] - HO[0] * OO[i+1]) */
  for(si=0;si<nSmat;++si) {
    pi = smatToParaIdx[si];
    g[si] = -DSROptStepDt*2.0*(creal(SROptHO[pi+2]) - creal(SROptHO[0]) * creal(SROptOO[pi+2]));
  }

  return;
}

#endif
