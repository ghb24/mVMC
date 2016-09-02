#include <matrixlapack.h>
#include "diag.h"
#include "mfmemory.c"

void diag(struct BindStruct *X){

    int int_i,int_j,int_k,int_l;
    double *r;
    double complex **tmp_mat,**vec;
    int mfint[7],xMsize;

 
    xMsize = X->Def.Nsite;  
    c_malloc2(tmp_mat,2*xMsize,2*xMsize);
    c_malloc2(vec,2*xMsize,2*xMsize);
    d_malloc1(r,2*xMsize);

    for(int_i = 0;int_i < 2*xMsize; int_i++){
      for(int_j = 0;int_j < 2*xMsize; int_j++){
        tmp_mat[int_i][int_j] = X->Large.Ham[int_i][int_j];
        //          fprintf(stdout, "Debug: mat(%d, %d)=(%lf, %lf)\n", int_i, int_j, creal(X->Large.Ham[int_i][int_j]), cimag(X->Large.Ham[int_i][int_j]));
      }
    }
    ZHEEVall(2*xMsize,tmp_mat,r,vec);
    for(int_k = 0; int_k < 2*xMsize; int_k++){
      X->Large.EigenValues[int_k] = r[int_k];
    }

    //For zero-temperature to generate pair-orbitals
    for(int_k = 0; int_k < 2*X->Def.Ne; int_k++){
      for(int_l = 0; int_l < 2*xMsize; int_l++){
        X->Large.R_SLT[int_l][int_k] = conj(vec[int_k][int_l]);
        X->Large.L_SLT[int_k][int_l] = (vec[int_k][int_l]);
      }
    }

    c_free2(tmp_mat,2*xMsize,2*xMsize);
    c_free2(vec,2*xMsize,2*xMsize);
    d_free1(r,2*xMsize);
}