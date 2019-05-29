// TODO include license and description

#ifndef _GPW_KERN_INCLUDE_FILES
#define _GPW_KERN_INCLUDE_FILES

// Computes the simple k(1) kernel
double GPWKernel1(const int *configA, const int sizeA, const int *configB,
                  const int sizeB, const int tRSym, const int shift);

/* computes the kernel matrix (k(1) kernel) for two lists of configurations
(in bitstring representation) */
void GPWKernel1Mat(const int *configsAUp, const int *configsADown,
                   const int sizeA, const int numA, const int *configsBUp,
                   const int *configsBDown, const int sizeB, const int numB,
                   const int tRSym, const int shift, const int symmetric,
                   double *kernelMatr);

/* computes the kernel vector (k(1) kernel) for one reference configuration
and a list of confgurations in bitstring representation) */
void GPWKernel1Vec(const int *configsAUp, const int *configsADown,
                   const int sizeA, const int numA, const int *configRef,
                   const int sizeRef, const int tRSym, const int shift,
                   double *kernelVec);

// computes the plaquette starting from index i and a
int CalculatePlaquette(const int i, const int a, const int *delta, const int sysSize,
                       const int trnSize, const int dim, const int *sysNeighbours,
                       const int *trnNeighbours, const int n, int *workspace);

// Computes the simple k(n) kernel
double GPWKernelN(const int *configA, const int *neighboursA, const int sizeA,
                  const int *configB, const int *neighboursB, const int sizeB,
                  const int dim, const int n, const int tRSym, const int shift);

/* computes the kernel matrix (k(n) kernel) for two lists of configurations
(in bitstring representation) */
void GPWKernelNMat(const int *configsAUp, const int *configsADown,
                   const int *neighboursA, const int sizeA, const int numA,
                   const int *configsBUp, const int *configsBDown,
                   const int *neighboursB, const int sizeB, const int numB,
                   const int dim, const int n, const int tRSym, const int shift,
                   const int symmetric, double *kernelMatr);

/* computes the kernel vector (k(n) kernel) for one reference configuration
and a list of confgurations in bitstring representation) */
void GPWKernelNVec(const int *configsAUp, const int *configsADown,
                   const int *neighboursA, const int sizeA, const int numA,
                   const int *configRef, const int *neighboursRef,
                   const int sizeRef, const int dim, const int n,
                   const int tRSym, const int shift, double *kernelVec);

// Computes the delta matrix
void CalculatePairDelta(int *delta, const int *sysCfg, const int sysSize, const int *trnCfg, const int trnSize);

//computes the inner sum needed for the computation of the full kernel
double CalculateInnerSum(const int i, const int a, const int *delta, const int sysSize,
                         const int trnSize, const int dim, const int *sysNeighbours,
                         const int *trnNeighbours, const int rC, const double thetaC,
                         int *workspace);

// Computes the full kernel
double GPWKernel(const int *sysCfg, const int *sysNeighbours, const int sysSize,
                 const int *trnCfg, const int *trnNeighbours, const int trnSize,
                 const int dim, const int power, const int rC,
                 const double theta0, const double thetaC, const int tRSym,
                 const int shift);

/* computes the kernel matrix (complete kernel) for two lists of configurations
(in bitstring representation) */
void GPWKernelMat(const int *configsAUp, const int *configsADown,
                  const int *neighboursA, const int sizeA, const int numA,
                  const int *configsBUp, const int *configsBDown,
                  const int *neighboursB, const int sizeB, const int numB,
                  const int dim, const int power, const int rC,
                  const double theta0, const double thetaC,
                  const int tRSym, const int shift, const int symmetric,
                  double *kernelMatr);

/* computes the kernel vector (complete kernel) for one reference configuration
and a list of confgurations in bitstring representation) */
void GPWKernelVec(const int *configsAUp, const int *configsADown,
                  const int *neighboursA, const int sizeA, const int numA,
                  const int *configRef, const int *neighboursRef,
                  const int sizeRef, const int dim, const int power,
                  const int rC, const double theta0, const double thetaC,
                  const int tRSym, const int shift,
                  double *kernelVec);

#endif // _GPW_KERN_INCLUDE_FILES
