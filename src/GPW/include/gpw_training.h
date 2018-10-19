// TODO include license and description

#ifndef _GPW_TRN_INCLUDE_FILES
#define _GPW_TRN_INCLUDE_FILES
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <gpw_kernel.h>

#ifdef _mpi_use
  #include <mpi.h>
#else
typedef int MPI_Comm;
MPI_Comm MPI_COMM_WORLD=0;
inline void MPI_Init(int argc, char* argv[]) {return;}
inline void MPI_Finalize() {return;}
inline void MPI_Abort(MPI_Comm comm, int errorcode) {exit(errorcode); return;}
inline void MPI_Barrier(MPI_Comm comm) {return;}
inline void MPI_Comm_size(MPI_Comm comm, int *size) {*size = 1; return;}
inline void MPI_Comm_rank(MPI_Comm comm, int *rank) {*rank = 0; return;}
#endif // _mpi_use


// global variables
int NTrn = 0; // total number of training points
int *TrnSize; // lattice sizes of the training sets
int *TrnCfgStrUp; // string representation of the up configurations
int *TrnCfgStrDown; // string representation of the down configurations
int **TrnCfg; // training configurations

double *CVec; // vector of the configuration factors for the training data (C_n) TODO complex values

#endif // _GPW_TRN_INCLUDE_FILES
