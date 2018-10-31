// TODO include license and description

#include "gpw_training.h"
#include <complex.h>
#include "blas_externs.h" // TODO what is the best way for this include?

int ReadTrnConfigs(const char *directory);

int InvertKernel(double* inv, double **kern, const double theta);

void SaveConfigFiles(const double complex *alpha);

int main(int argc, char* argv[]) {
  int i,j;
  double complex *alpha;
  double **kernelMatr, *invKernel;
  const double theta = 1.0; // TODO read in hyperparameter

  if (argc <= 1) {
    fprintf(stderr, "Error: Specify input directory of the training data.\n");
    return -1;
  }

  fprintf(stdout,"Start: Read in the training configurations.\n");
  if(ReadTrnConfigs(argv[1]) != 0) return -1;
  fprintf(stdout,"End: Read in the training configurations.\n");


  // store configurations in mVMC representation
  fprintf(stdout,"Start: Convert configurations into mVMC representation.\n");
  for(i=0;i<NTrn;i++) {
    int *up, *down, j;
    TrnCfg[i] = (int*)malloc(2*sizeof(int)*TrnSize[i]);
    up = TrnCfg[i];
    down = TrnCfg[i]+TrnSize[i];

    for(j=0;j<TrnSize[i];j++) {
      up[j] = (TrnCfgStrUp[i] >> j) & 1;
      down[j] = (TrnCfgStrDown[i] >> j) & 1;
    }
  }
  fprintf(stdout,"End: Convert configurations into mVMC representation.\n");

  // compute kernel matrix; TODO parallelise!! TODO include hyperparameter
  fprintf(stdout,"Start: Compute the kernel matrix.\n");
  kernelMatr = (double**)malloc(sizeof(double*)*NTrn);
  for(i=0;i<NTrn;i++) kernelMatr[i] = (double*)malloc(sizeof(double)*NTrn);
  for(i=0;i<NTrn;i++) {
    for(j=i;j<NTrn;j++) {
      kernelMatr[i][j] = GPWKernel1(TrnCfg[i], TrnSize[i], TrnCfg[j], TrnSize[j]);
    }
  }
  for(i=0;i<NTrn;i++) {
    for(j=i+1;j<NTrn;j++) {
      kernelMatr[j][i] = kernelMatr[i][j];
    }
  }
  fprintf(stdout,"End: Compute the kernel matrix. Size: %d x %d\n", NTrn, NTrn);

  // invert matrix
  fprintf(stdout,"Start: Invert the kernel matrix.\n");
  invKernel = (double*)malloc(sizeof(double)*NTrn*NTrn);
  if (InvertKernel(invKernel, kernelMatr, theta) != 0) {
    fprintf (stderr, "Matrix inversion failed. \n");
  }
  fprintf(stdout,"End: Invert the kernel matrix.\n");


  // compute vector of alpha values TODO optimise!
  fprintf(stdout,"Start: Create array of alpha values.\n");
  alpha = (double complex*)malloc(sizeof(double complex)*NTrn);
  for(i=0;i<NTrn;i++) {
    double complex an = 0 + 0*I;
    for(j=0;j<NTrn;j++) {
      an += invKernel[i+NTrn*j] * CVec[j];
    }
    alpha[i] = an;
  }
  fprintf(stdout,"End: Create array of alpha values.\n");

  // write out config files for mVMC
  fprintf(stdout,"Start: Write out training data for mVMC.\n");
  SaveConfigFiles(alpha);
  fprintf(stdout,"End: Write out training data for mVMC.\n");



  //free memories
  free(alpha);
  free(invKernel);
  for(i=0;i<NTrn;i++) free(kernelMatr[i]);
  free(kernelMatr);
  for(i=0;i<NTrn;i++) free(TrnCfg[i]);
  free(CVec);
  free(TrnCfg);
  free(TrnCfgStrDown);
  free(TrnCfgStrUp);
  free(TrnSize);

  return 0;
}


int ReadTrnConfigs(const char *directory) {
  char fileName[500], line[500];
  FILE *fp;
  int i,x,y;
  double *trnAmp, *trnSD;
  double amp;

  // read no of training configs
  strcat(strcpy(fileName, directory), "/config_strings");
  fp = fopen(fileName, "r");
  if (fp == NULL) {
    fprintf(stderr, "Error reading file %s\n", fileName);
    return -1;
  }
  while (fgets(line, sizeof(line)/sizeof(char), fp) != NULL) NTrn++;
  fclose(fp);

  // set memories
  TrnSize = (int*)malloc(sizeof(int)*NTrn);
  TrnCfgStrUp = (int*)malloc(sizeof(int)*NTrn);
  TrnCfgStrDown = (int*)malloc(sizeof(int)*NTrn);
  TrnCfg = (int**)malloc(sizeof(int*)*NTrn);
  CVec = (double complex*)malloc(sizeof(double complex)*NTrn);

  trnAmp = (double*)malloc(sizeof(double*)*NTrn);
  trnSD = (double*)malloc(sizeof(double)*NTrn);

  // read size of training set
  strcat(strcpy(fileName,directory), "/config_sizes");
  fp = fopen(fileName, "r");
  if (fp == NULL) {
    fprintf(stderr, "Error reading file %s\n", fileName);
    return -1;
  }
  i = 0;
  while (fscanf(fp, "%d\n", &TrnSize[i]) != EOF) i++;
  fclose(fp);

  // read training configs
  strcat(strcpy(fileName,directory), "/config_strings");
  fp = fopen(fileName, "r");
  if (fp == NULL) {
    fprintf(stderr, "Error reading file %s\n", fileName);
    return -1;
  }
  i = 0;
  while (fscanf(fp, "%d %d\n", &TrnCfgStrUp[i], &TrnCfgStrDown[i]) != EOF) i++;
  fclose(fp);

  // read training amplitudes
  strcat(strcpy(fileName,directory), "/config_amps");
  fp = fopen(fileName, "r");
  if (fp == NULL) {
    fprintf(stderr, "Error reading file %s\n", fileName);
    return -1;
  }
  i = 0;
  while (fscanf(fp, "%d %d %lf\n", &x, &y, &amp) != EOF) {
    trnAmp[i] = amp; // TODO support for complex amplitudes
    i++;
  }
  fclose(fp);

  // read training slater determinants
  strcat(strcpy(fileName,directory), "/config_SD");
  fp = fopen(fileName, "r");
  if (fp == NULL) {
    fprintf(stderr, "Error reading file %s\n", fileName);
    return -1;
  }
  i = 0;
  while (fscanf(fp, "%lf\n", &trnSD[i]) != EOF) i++;
  fclose(fp);

  // compute CVec
  for(i=0;i<NTrn;i++) {
    CVec[i] = clog(trnAmp[i]/trnSD[i]);
  }
  free(trnSD);
  free(trnAmp);


  return 0;
}

int InvertKernel(double* inv, double **kern, const double theta) {
  int i,j;
  int errorHandler;
  int nn = NTrn*NTrn;
  double *lapackWorkspace = (double*)malloc(sizeof(double)*nn);
  int *pivotArray = (int*)malloc(sizeof(int)*(NTrn));

  for(i=0;i<NTrn;i++) {
    for(j=0;j<NTrn;j++) {
      inv[i + NTrn*j] = kern[i][j];
      if(i == j) inv[i + NTrn*j] += theta;
    }
  }

  // TODO exploit symmetry + positivity of the matrix! more stable method: solve linear system...TODO implement!
  M_DGETRF(&NTrn, &NTrn, inv, &NTrn, pivotArray, &errorHandler);
  M_DGETRI(&NTrn, inv, &NTrn, pivotArray, lapackWorkspace, &nn, &errorHandler);
  if (errorHandler != 0) {
    return -1;
  }

  free(pivotArray);
  free(lapackWorkspace);

  return 0;
}

void SaveConfigFiles(const double complex *alpha) {
  FILE *fp;
  int i;
  const int opt = 0; //sets optimisation flag for the alpha values ...TODO implement support for this
  fp = fopen("gpwidx.def", "w");
  fprintf(fp, "====================== \n");
  fprintf(fp, "NGPWIdx %d  \n", NTrn);
  fprintf(fp, "ComplexType %d  \n", 1);
  fprintf(fp, "====================== \n");
  fprintf(fp, "====================== \n");
  for(i=0;i<NTrn;i++) fprintf(fp, "%d %d %d %d\n", TrnSize[i], TrnCfgStrUp[i], TrnCfgStrDown[i], i);
  for(i=0;i<NTrn;i++) fprintf(fp, "%d %d\n", i, opt);
  fflush(fp);
  fclose(fp);
  fprintf(stdout, "gpwidx.def written.\n");

  fp = fopen("ingpw.def", "w");
  fprintf(fp, "====================== \n");
  fprintf(fp, "NGPWIdx %d  \n", NTrn);
  fprintf(fp, "====================== \n");
  fprintf(fp, "========i_GPWCfg===== \n");
  fprintf(fp, "====================== \n");
  for(i=0;i<NTrn;i++) fprintf(fp, "%d %f %f\n", i, creal(alpha[i]), cimag(alpha[i]));
  fflush(fp);
  fclose(fp);
  fprintf(stdout, "ingpw.def written.\n");
}
