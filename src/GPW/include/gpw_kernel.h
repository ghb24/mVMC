// TODO include license and description

#ifndef _GPW_KERN_INCLUDE_FILES
#define _GPW_KERN_INCLUDE_FILES

// Computes the simple k(1) kernel
double GPWKernel1(const int *configA, const int sizeA, const int *configB,
                  const int sizeB, const int tRSym, const int shift);

/* computes the kernel matrix (k(1) kernel) for two lists of configurations
(in bitstring representation) */
void GPWKernel1Mat(const unsigned long *configsAUp, const unsigned long *configsADown,
                   const int sizeA, const int numA, const unsigned long *configsBUp,
                   const unsigned long *configsBDown, const int sizeB, const int numB,
                   const int tRSym, const int shift, const int symmetric,
                   double *kernelMatr);

/* computes the kernel vector (k(1) kernel) for one reference configuration
and a list of confgurations in bitstring representation) */
void GPWKernel1Vec(const unsigned long *configsAUp, const unsigned long *configsADown,
                   const int sizeA, const int numA, const int *configRef,
                   const int sizeRef, const int tRSym, const int shift,
                   double *kernelVec);

// Computes the delta matrix
void CalculatePairDelta(int *delta, const int *cfgA, const int sizeA,
                        const int *cfgB, const int sizeB);

// Sets up the index list for the two lattices
int SetupPlaquetteIdx(const int rC, const int *neighboursA,
                      const int sizeA, const int *neighboursB,
                      const int sizeB, const int dim, int **plaquetteAIdx,
                      int **plaquetteBIdx, int **distList);

// Frees the memory allocated in SetupPlaquetteIdx
void FreeMemPlaquetteIdx(int *plaquetteAIdx, int *plaquetteBIdx,
                         int *distList);


double GPWKernelN(const int *cfgA, const int *plaquetteAIdx,
                  const int sizeA, const int *cfgB, const int *plaquetteBIdx,
                  const int sizeB, const int dim, const int n, const int tRSym,
                  const int shift, int *workspace);

/* computes the kernel matrix (k(n) kernel) for two lists of configurations
(in bitstring representation) */
void GPWKernelNMat(const unsigned long *configsAUp,
                   const unsigned long *configsADown, const int *neighboursA,
                   const int sizeA, const int numA,
                   const unsigned long *configsBUp,
                   const unsigned long *configsBDown,
                   const int *neighboursB, const int sizeB, const int numB,
                   const int dim, const int n, const int tRSym,
                   const int shift, const int symmetric, double *kernelMatr);

/* computes the kernel vector (k(n) kernel) for one reference configuration
and a list of confgurations in bitstring representation) */
void GPWKernelNVec(const unsigned long *configsAUp,
                   const unsigned long *configsADown, const int *neighboursA,
                   const int sizeA, const int numA,
                   const int *configRef, const int *neighboursRef,
                   const int sizeRef, const int dim, const int n,
                   const int tRSym, const int shift, double *kernelVec);

// Computes the full kernel
double GPWKernel(const int *cfgA, const int *plaquetteAIdx, const int sizeA,
                 const int *cfgB, const int *plaquetteBIdx, const int sizeB,
                 const int dim, const int power, const double theta0,
                 const double thetaC, const int tRSym, const int shift,
                 const int plaquetteSize, const int *distList, int *workspace);

/* computes the kernel matrix (complete kernel) for two lists of configurations
(in bitstring representation) */
void GPWKernelMat(const unsigned long *configsAUp,
                  const unsigned long *configsADown,
                  const int *neighboursA, const int sizeA, const int numA,
                  const unsigned long *configsBUp,
                  const unsigned long *configsBDown,
                  const int *neighboursB, const int sizeB, const int numB,
                  const int dim, const int power, const int rC,
                  const double theta0, const double thetaC,
                  const int tRSym, const int shift, const int symmetric,
                  double *kernelMatr);

/* computes the kernel vector (complete kernel) for one reference configuration
and a list of confgurations in bitstring representation) */
void GPWKernelVec(const unsigned long *configsAUp,
                  const unsigned long *configsADown, const int *neighboursA,
                  const int sizeA, const int numA, const int *configRef,
                  const int *neighboursRef, const int sizeRef, const int dim,
                  const int power, const int rC, const double theta0,
                  const double thetaC, const int tRSym, const int shift,
                  double *kernelVec);

#endif // _GPW_KERN_INCLUDE_FILES
