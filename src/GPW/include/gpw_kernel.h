// TODO include license and description

#ifndef _GPW_KERN_INCLUDE_FILES
#define _GPW_KERN_INCLUDE_FILES

// Computes the simple k(1) kernel
double GPWKernel1(const int *configA, const int sizeA, const int *configB, const int sizeB);

// Computes the simple k(2) kernel
double GPWKernel2(const int *configA, const int sizeA, const int *configB, const int sizeB);

// Computes the simple k(n) kernel
double GPWKernelN(const int *configA, const int sizeA, const int *configB, const int sizeB, const int n);

// Computes the delta matrix
void CalculatePairDelta(int *delta, const int *sysCfg, const int sysSize, const int *trnCfg, const int trnSize);

//computes the sum needed for the computation of the full kernel
double CalculateRangeSum(const int i, const int a, const int *delta, const int sysSize,
                         const int trnSize, const int d, const int *sysNeighbours,
                         const int *trnNeighbours, const int rC, const double theta0,
                         const double thetaC, int *workspace);

// Updates the full kernel with respect to the update matrix
double GPWKernel(const double kernelOld, const double *termMatr, const int updateCount);

// Computes the full kernel without help matrices
double GPWKernelInPlace(const int *sysCfg, const int *sysNeighbours, const int sysSize,
                        const int *trnCfg, const int *trnNeighbours, const int trnSize,
                        const int d, const int rC, const double theta0, const double thetaC);

#endif // _GPW_KERN_INCLUDE_FILES
