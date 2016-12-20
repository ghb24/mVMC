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
#include "StdFace_vals.h"
#include "StdFace_ModelUtil.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <string.h>

/**
 *
 * Setup a Hamiltonian for the generalized Heisenberg model on a square lattice
 *
 * @author Mitsuaki Kawamura (The University of Tokyo)
 */
void StdFace_Ladder(struct StdIntList *StdI, char *model)
{
  FILE *fp;
  int isite, jsite, ksite;
  int iL, iW;
  int ktrans, kintr;
  double complex phase;

  StdI->NsiteUC = 1;
  fprintf(stdout, "\n");
  fprintf(stdout, "#######  Parameter Summary  #######\n");
  fprintf(stdout, "\n");
  /**/
  StdI->NsiteUC = 1;
  fprintf(stdout, "  @ Lattice Size & Shape\n\n");
  StdFace_RequiredVal_i("L", StdI->L);
  StdFace_RequiredVal_i("W", StdI->W);
  StdFace_NotUsed_i("a0W", StdI->a0W);
  StdFace_NotUsed_i("a0L", StdI->a0L);
  StdFace_NotUsed_i("a1W", StdI->a1W);
  StdFace_NotUsed_i("a1L", StdI->a1L);
  /**/
  StdI->a0 = 1.0; StdI->a1 = 1.0;
  fp = fopen("/dev/null", "w");
  StdFace_InitSite2D(StdI, fp);
  fclose(fp);
  StdI->tau[0][0] = 0.0; StdI->tau[0][1] = 0.0;
  /**/
  StdFace_PrintVal_d("phase0", &StdI->phase0, 0.0);
  StdFace_NotUsed_c("phase1", StdI->phase1);
  StdI->phase1 = StdI->phase0;
  StdI->phase0 = 0.0;
  StdI->ExpPhase0 = cos(StdI->pi180 * StdI->phase0) + I*sin(StdI->pi180 * StdI->phase0);
  StdI->ExpPhase1 = cos(StdI->pi180 * StdI->phase1) + I*sin(StdI->pi180 * StdI->phase1);
  if (cabs(StdI->ExpPhase0 + 1.0) < 0.000001) StdI->AntiPeriod0 = 1;
  else StdI->AntiPeriod0 = 0;
  if (cabs(StdI->ExpPhase1 + 1.0) < 0.000001) StdI->AntiPeriod1 = 1;
  else StdI->AntiPeriod1 = 0;
  /**/
  fprintf(stdout, "\n  @ Hamiltonian \n\n");
  StdFace_NotUsed_J("J", StdI->JAll, StdI->J);
  StdFace_NotUsed_J("J'", StdI->JpAll, StdI->Jp);
  StdFace_NotUsed_c("t", StdI->t);
  StdFace_NotUsed_c("t'", StdI->tp);
  StdFace_NotUsed_d("V", StdI->V);
  StdFace_NotUsed_d("V'", StdI->Vp);
  StdFace_NotUsed_d("K", StdI->K);
  /**/
  if (strcmp(StdI->model, "spin") == 0 ) {
    StdFace_PrintVal_i("2S", &StdI->S2, 1);
    StdFace_PrintVal_d("h", &StdI->h, 0.0);
    StdFace_PrintVal_d("Gamma", &StdI->Gamma, 0.0);
    StdFace_PrintVal_d("D", &StdI->D[2][2], 0.0);
    StdFace_InputSpin(StdI, StdI->J0, StdI->J0All, "J0");
    StdFace_InputSpin(StdI, StdI->J1, StdI->J1All, "J1");
    StdFace_InputSpin(StdI, StdI->J2, StdI->J2All, "J2");
    StdFace_InputSpin(StdI, StdI->J1p, StdI->J1pAll, "J1'");
    StdFace_InputSpin(StdI, StdI->J2p, StdI->J2pAll, "J2'");
    /**/
    StdFace_NotUsed_d("mu", StdI->mu);
    StdFace_NotUsed_d("U", StdI->U);
    StdFace_NotUsed_c("t0", StdI->t0);
    StdFace_NotUsed_c("t1", StdI->t1);
    StdFace_NotUsed_c("t2", StdI->t2);
    StdFace_NotUsed_c("t1'", StdI->t1p);
    StdFace_NotUsed_c("t2'", StdI->t2p);
    StdFace_NotUsed_d("V0", StdI->V0);
    StdFace_NotUsed_d("V1", StdI->V1);
    StdFace_NotUsed_d("V2", StdI->V2);
    StdFace_NotUsed_d("V1'", StdI->V1p);
    StdFace_NotUsed_d("V2'", StdI->V2p);
  }/*if (strcmp(StdI->model, "spin") == 0 )*/
  else {
    StdFace_PrintVal_d("mu", &StdI->mu, 0.0);
    StdFace_PrintVal_d("U", &StdI->U, 0.0);
    StdFace_InputHopp(StdI, &StdI->t0, "t0");
    StdFace_InputHopp(StdI, &StdI->t1, "t1");
    StdFace_InputHopp(StdI, &StdI->t2, "t2");
    StdFace_InputHopp(StdI, &StdI->t1p, "t1'");
    StdFace_InputHopp(StdI, &StdI->t2p, "t2'");
    StdFace_InputCoulombV(StdI, &StdI->V0, "V0");
    StdFace_InputCoulombV(StdI, &StdI->V1, "V1");
    StdFace_InputCoulombV(StdI, &StdI->V2, "V2");
    StdFace_InputCoulombV(StdI, &StdI->V1p, "V1'");
    StdFace_InputCoulombV(StdI, &StdI->V2p, "V2'");
    /**/
    StdFace_NotUsed_J("J0", StdI->J0All, StdI->J0);
    StdFace_NotUsed_J("J1", StdI->J1All, StdI->J1);
    StdFace_NotUsed_J("J2", StdI->J2All, StdI->J2);
    StdFace_NotUsed_J("J1p", StdI->J1pAll, StdI->J1p);
    StdFace_NotUsed_J("J2p", StdI->J2pAll, StdI->J2p);
    StdFace_NotUsed_d("h", StdI->h);
    StdFace_NotUsed_d("Gamma", StdI->Gamma);
    StdFace_NotUsed_d("D", StdI->D[2][2]);

    if (strcmp(StdI->model, "hubbard") == 0 ) {
      StdFace_NotUsed_i("2S", StdI->S2);
      StdFace_NotUsed_J("J", StdI->JAll, StdI->J);
    }
    else {
      StdFace_PrintVal_i("2S", &StdI->S2, 1);
      StdFace_InputSpin(StdI, StdI->J, StdI->JAll, "J");
    }
  }/*if (model != "spin")*/
  fprintf(stdout, "\n  @ Numerical conditions\n\n");
  /*
  Local Spin
  */
  StdI->nsite = StdI->L * StdI->W;
  if (strcmp(StdI->model, "kondo") == 0 ) StdI->nsite *= 2;
  StdI->locspinflag = (int *)malloc(sizeof(int) * StdI->nsite);
  /**/
  if (strcmp(StdI->model, "spin") == 0 )
    for (isite = 0; isite < StdI->nsite; isite++)StdI->locspinflag[isite] = StdI->S2;
  else if (strcmp(StdI->model, "hubbard") == 0 )
    for (isite = 0; isite < StdI->nsite; isite++)StdI->locspinflag[isite] = 0;
  else if (strcmp(StdI->model, "kondo") == 0 )
    for (isite = 0; isite < StdI->nsite / 2; isite++) {
      StdI->locspinflag[isite] = StdI->S2;
      StdI->locspinflag[isite + StdI->nsite / 2] = 0;
    }
  /*
  The number of Transfer & Interaction
  */
  if (strcmp(StdI->model, "spin") == 0 ) {
    StdI->ntrans = StdI->L * StdI->W * (StdI->S2 + 1/*h*/ + 2 * StdI->S2/*Gamma*/);
    StdI->nintr = StdI->L * StdI->W * (1/*D*/ + 1/*J1*/ + 1/*J1'*/)
      * (3 * StdI->S2 + 1) * (3 * StdI->S2 + 1)
      + StdI->L * (StdI->W - 1) * (1/*J0*/ + 1/*J2*/ + 1/*J2'*/)
      * (3 * StdI->S2 + 1) * (3 * StdI->S2 + 1);
  }/*if (strcmp(StdI->model, "spin") == 0 )*/
  else {
    StdI->ntrans = StdI->L*StdI->W * (1/*mu*/ + 2/*t1*/ + 2/*t1'*/)
      + StdI->L*(StdI->W - 1) * (2/*t0*/ + 2/*t2*/ + 2/*t2'*/);
    StdI->nintr = StdI->L*StdI->W * 1/*U*/
      + StdI->L*StdI->W * 4 * (1/*V1*/ + 1/*V1'*/)
      + StdI->L*(StdI->W - 1) * 4 * (1/*V0*/ + 1/*V2*/ + 1/*V2'*/);

    if (strcmp(StdI->model, "kondo") == 0 ) StdI->nintr += StdI->nsite / 2 * (3 * 1 + 1) * (3 * StdI->S2 + 1);
  }
  /**/
  StdI->transindx = (int **)malloc(sizeof(int*) * StdI->ntrans);
  StdI->trans = (double complex *)malloc(sizeof(double complex) * StdI->ntrans);
  for (ktrans = 0; ktrans < StdI->ntrans; ktrans++){
    StdI->transindx[ktrans] = (int *)malloc(sizeof(int) * 4);
  }
  /**/
  StdI->intrindx = (int **)malloc(sizeof(int*) * StdI->nintr);
  StdI->intr = (double complex *)malloc(sizeof(double complex) * StdI->nintr);
  for (kintr = 0; kintr < StdI->nintr; kintr++) {
    StdI->intrindx[kintr] = (int *)malloc(sizeof(int) * 8);
  }
  StdFace_InterAllSeparate(StdI);/*debug*/
  /*
   Set Transfer & Interaction
  */
  StdI->ntrans = 0;
  StdI->nintr = 0;
  for (iL = 0; iL < StdI->L; iL++) {
    for (iW = 0; iW < StdI->W; iW++) {

      isite = iW + iL * StdI->W;
      if (strcmp(StdI->model, "kondo") == 0 ) isite += StdI->L * StdI->W;
      /*
       Local term
      */
      if (strcmp(StdI->model, "spin") == 0 ) {
        StdFace_MagField(StdI, StdI->S2, -StdI->h, -StdI->Gamma, isite);
        StdFace_GeneralJ(StdI, StdI->D, StdI->S2, StdI->S2, isite, jsite);
      }/*if (strcmp(StdI->model, "spin") == 0 )*/
      else {
        StdFace_Hopping(StdI, StdI->mu, isite, isite, 0);
        StdFace_intr(StdI, StdI->U, isite, 0, isite, 0, isite, 1, isite, 1);
        StdI->Cintra[StdI->NCintra] = StdI->U; StdI->CintraIndx[StdI->NCintra][0] = isite; StdI->NCintra += 1;
        /**/
        if (strcmp(StdI->model, "kondo") == 0 ) {
          jsite = iW + iL * StdI->W;
          StdFace_GeneralJ(StdI, StdI->J, 1, StdI->S2, isite, jsite);
        }/*if (strcmp(StdI->model, "kondo") == 0 )*/
      }/*if (model != "spin")*/
      /*
       Nearest neighbor along the ladder
      */
      jsite = iW + ((iL + 1) % StdI->L) * StdI->W;
      phase = cpow(StdI->ExpPhase0, (double)((iL + 1) / StdI->L));
      if (strcmp(StdI->model, "kondo") == 0 ) jsite += StdI->L * StdI->W;
      /**/
      if (strcmp(StdI->model, "spin") == 0 ) {
        StdFace_GeneralJ(StdI, StdI->J1, StdI->S2, StdI->S2, isite, jsite);
      }/*if (strcmp(StdI->model, "spin") == 0 )*/
      else {
        StdFace_Hopping(StdI, phase * StdI->t1, isite, jsite, 1);
        StdFace_Coulomb(StdI, StdI->V1, isite, jsite);
      }/*if (model != "spin")*/
      /*
       Second nearest neighbor along the ladder
      */
      jsite = iW + ((iL + 2) % StdI->L) * StdI->W;
      phase = cpow(StdI->ExpPhase0, (double)((iL + 2) / StdI->L));
      if (strcmp(StdI->model, "kondo") == 0 ) jsite += StdI->L * StdI->W;
      if (strcmp(StdI->model, "spin") == 0 ) {
        StdFace_GeneralJ(StdI, StdI->J1p, StdI->S2, StdI->S2, isite, jsite);
      }/*if (strcmp(StdI->model, "spin") == 0 )*/
      else {
        StdFace_Hopping(StdI, phase * StdI->t1p, isite, jsite, 1);
        StdFace_Coulomb(StdI, StdI->V1p, isite, jsite);
      }/*if (model != "spin")*/
      /*
      Across rung
      */
      if (iW < StdI->W - 1) {
        /*
         Vertical
        */
        jsite = (iW + 1) + iL * StdI->W;
        if (strcmp(StdI->model, "kondo") == 0 ) jsite += StdI->L * StdI->W;
        /**/
        if (strcmp(StdI->model, "spin") == 0 ) {
          StdFace_GeneralJ(StdI, StdI->J0, StdI->S2, StdI->S2, isite, jsite);
        }/*if (strcmp(StdI->model, "spin") == 0 )*/
        else {
          StdFace_Hopping(StdI, StdI->t0, isite, jsite, 1);
          StdFace_Coulomb(StdI, StdI->V0, isite, jsite);
        }/*if (model != "spin")*/
        /*
         Diagonal 1
        */
        jsite = (iW + 1) + ((iL + 1) % StdI->L) * StdI->W;
        if (strcmp(StdI->model, "kondo") == 0 ) jsite += StdI->L * StdI->W;
        /**/
        if (strcmp(StdI->model, "spin") == 0 ) {
          StdFace_GeneralJ(StdI, StdI->J2, StdI->S2, StdI->S2, isite, jsite);
        }/*if (strcmp(StdI->model, "spin") == 0 )*/
        else {
          StdFace_Hopping(StdI, StdI->t2, isite, jsite, 1);
          StdFace_Coulomb(StdI, StdI->V2, isite, jsite);
        }/*if (model != "spin")*/
        /*
         Diagonal 2
        */
        /**/
        if (strcmp(StdI->model, "spin") == 0 ) {
        }
        ksite = (iW + 1) + iL * StdI->W;
        if (strcmp(StdI->model, "kondo") == 0 ) ksite += StdI->L * StdI->W;
        jsite = iW + ((iL + 1) % StdI->L) * StdI->W;
        if (strcmp(StdI->model, "kondo") == 0 ) jsite += StdI->L * StdI->W;
        if (strcmp(StdI->model, "spin") == 0 ) {
          StdFace_GeneralJ(StdI, StdI->J2p, StdI->S2, StdI->S2, ksite, jsite);
        }/*if (strcmp(StdI->model, "spin") == 0 )*/
        else {
          StdFace_Hopping(StdI, StdI->t2p, ksite, jsite, 1);
          StdFace_Coulomb(StdI, StdI->V2p, ksite, jsite);
        }/*if (model != "spin")*/

      }/*if (iW < StdI->W - 1)*/

    }/*for (iW = 0; iW < StdI->W; iW++)*/
  }/*for (iL = 0; iL < StdI->L; iL++)*/
}
