// TODO include license and description

#ifndef _GPW_KERN_INCLUDE_FILES
#define _GPW_KERN_INCLUDE_FILES

// Computes the simple k(1) kernel
double GPWKernel1(const int *configA, const int sizeA, const int *configB,
                  const int sizeB, const int tRSym, const int shift,
                  const int startIdA, const int startIdB);

/* computes the kernel matrix (k(1) kernel) for two lists of configurations
(in bitstring representation) */
void GPWKernel1Mat(const unsigned long *configsAUp, const unsigned long *configsADown,
                   const int sizeA, const int numA, const unsigned long *configsBUp,
                   const unsigned long *configsBDown, const int sizeB, const int numB,
                   const int tRSym, const int shift, const int startIdA,
                   const int startIdB, const int symmetric, double *kernelMatr);

/* computes the kernel vector (k(1) kernel) for one reference configuration
and a list of confgurations in bitstring representation) */
void GPWKernel1Vec(const unsigned long *configsAUp, const unsigned long *configsADown,
                   const int sizeA, const int numA, const int *configRef,
                   const int sizeRef, const int tRSym, const int shift,
                   const int startIdA, const int startIdB, double *kernelVec);

// computes the delta function
int delta(const int *cfgA, const int sizeA,  const int *cfgB, const int sizeB, const int i, const int a, const int flip);

// Sets up the index list for the two lattices
int SetupPlaquetteIdx(const int rC, const int *neighboursA,
                      const int sizeA, const int *neighboursB,
                      const int sizeB, const int dim, int **plaquetteAIdx,
                      int **plaquetteBIdx, int **distList, int includeCentre);

// Frees the memory allocated in SetupPlaquetteIdx
void FreeMemPlaquetteIdx(int *plaquetteAIdx, int *plaquetteBIdx,
                         int *distList);

// Sets up the hash tables for the fast update of the inSum matrix
void SetupPlaquetteHash(const int sysSize, const int plaqSize,
                        const int *plaqIdx, int ***plaqHash,
                        int **plaqHashSz);

// Frees the memory allocated in SetupPlaquetteHash
void FreeMemPlaquetteHash(const int sysSize, int **plaqHash, int *plaqHashSz);

// Computes the inner sum for the full kernel
void ComputeInSum(double complex *inSum, const int *plaquetteAIdx,
                  const int *cfgA, const int sizeA,
                  const int *plaquetteBIdx, const int *cfgB, const int sizeB,
                  const int plaquetteSize, const int *distList,
                  const int shift, const int startIdA, const int startIdB,
                  const double distWeightPower, const int tRSym);

// Updates the inner sum for the full kernel
void UpdateInSum(double complex *inSumNew, const int *cfgAOldReduced, const int *cfgANew,
                 const int *plaquetteAIdx, const int sizeA,
                 const int *plaquetteBIdx, const int *cfgB, const int sizeB,
                 const int plaquetteSize, const int *distList,
                 const int shift, const int startIdA, const int startIdB,
                 const double distWeightPower, int **plaqHash,
                 int *plaqHashSz, const int siteA, const int siteB,
                 const int tRSym);

// Computes the full kernel
double ComputeKernel(const int *cfgA, const int sizeA, const int *cfgB,
                     const int sizeB, const double power,
                     const double theta, const double norm, const int tRSym,
                     const int shift, const int startIdA, const int startIdB,
                     const double complex *inSum);

/* Computes the derivative of the kernel (divided by the wave function amplitude)
with respect to theta */
double ComputeKernDeriv(const int *cfgA, const int sizeA, const int *cfgB,
                        const int sizeB, const double power,
                        const double theta, const double norm, const int tRSym,
                        const int shift, const int startIdA, const int startIdB,
                        const double complex *inSum);

// Computes the kn kernel
double ComputeKernelN(const int *cfgA, const int *plaquetteAIdx, const int sizeA,
                      const int *cfgB, const int *plaquetteBIdx, const int sizeB,
                      const int n, const int tRSym, const int shift,
                      const int startIdA, const int startIdB, const double complex *inSum);

// Computes the kn kernel in place
double GPWKernelN(const int *cfgA, const int *plaquetteAIdx, const int sizeA,
                  const int *cfgB, const int *plaquetteBIdx, const int sizeB,
                  const int dim, const int n, const int tRSym, const int shift,
                  const int startIdA, const int startIdB, double complex *workspace);

/* computes the kernel matrix (k(n) kernel) for two lists of configurations
(in bitstring representation) */
void GPWKernelNMat(const unsigned long *configsAUp,
                   const unsigned long *configsADown, const int *neighboursA,
                   const int sizeA, const int numA,
                   const unsigned long *configsBUp,
                   const unsigned long *configsBDown, const int *neighboursB,
                   const int sizeB, const int numB, const int dim, const int n,
                   const int tRSym, const int shift, const int startIdA,
                   const int startIdB, const int symmetric,
                   double *kernelMatr);

/* computes the kernel vector (k(n) kernel) for one reference configuration
and a list of confgurations in bitstring representation) */
void GPWKernelNVec(const unsigned long *configsAUp, const unsigned long *configsADown,
                   const int *neighboursA, const int sizeA, const int numA,
                   const int *configRef, const int *neighboursRef,
                   const int sizeRef, const int dim, const int n,
                   const int tRSym, const int shift, const int startIdA,
                   const int startIdB, double *kernelVec);

// Computes the full kernel in place
double GPWKernel(const int *cfgA, const int *plaquetteAIdx, const int sizeA,
                 const int *cfgB, const int *plaquetteBIdx, const int sizeB,
                 const double power, const double theta, const double distWeightPower,
                 const int tRSym, const int shift, const int startIdA,
                 const int startIdB, const int plaquetteSize, const int *distList,
                 double complex *workspace);

/* computes the kernel matrix (complete kernel) for two lists of configurations
(in bitstring representation) */
void GPWKernelMat(const unsigned long *configsAUp,
                  const unsigned long *configsADown, const int *neighboursA,
                  const int sizeA, const int numA,
                  const unsigned long *configsBUp,
                  const unsigned long *configsBDown,
                  const int *neighboursB, const int sizeB, const int numB,
                  const int dim, const double power, const int rC,
                  const double theta, const double distWeightPower,
                  const int tRSym, const int shift, const int startIdA,
                  const int startIdB, const int symmetric,
                  double *kernelMatr);

/* computes the kernel vector (complete kernel) for one reference configuration
and a list of confgurations in bitstring representation) */
void GPWKernelVec(const unsigned long *configsAUp,
                  const unsigned long *configsADown, const int *neighboursA,
                  const int sizeA, const int numA, const int *configRef,
                  const int *neighboursRef, const int sizeRef, const int dim,
                  const double power, const int rC, const double theta,
                  const double distWeightPower, const int tRSym, const int shift,
                  const int startIdA, const int startIdB, double *kernelVec);
#endif // _GPW_KERN_INCLUDE_FILES
