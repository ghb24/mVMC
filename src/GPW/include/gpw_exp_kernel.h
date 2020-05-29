// TODO include license and description

#ifndef _GPW_EXP_KERN_INCLUDE_FILES
#define _GPW_EXP_KERN_INCLUDE_FILES

#include "complex.h"

// Computes the inner sum for the exponential kernel
void ComputeInSumExp(double *inSum, const int *plaquetteAIdx, const int *cfgA,
                     const int sizeA, const int *plaquetteBIdx, const int *cfgB,
                     const int sizeB, const int plaquetteSize, const double complex *distWeights,
                     const int *distWeightIdx, const int shift, const int startIdA, const int startIdB,
                     const int tRSym);

// Updates the inner sum for the exponential kernel
void UpdateInSumExp(double *inSumNew, const int *cfgAOldReduced, const int *cfgANew,
                    const int *plaquetteAIdx, const int sizeA,
                    const int *plaquetteBIdx, const int *cfgB, const int sizeB,
                    const int plaquetteSize,
                    const double complex *distWeights,
                    const int *distWeightIdx, const int shift, const int startIdA,
                    const int startIdB,int **plaqHash, int *plaqHashSz, const int siteA,
                    const int siteB, const int tRSym);


// Computes the exponential kernel
double ComputeExpKernel(const int *cfgA, const int sizeA, const int *cfgB,
                        const int sizeB, const int tRSym,
                        const int shift, const int startIdA, const int startIdB,
                        const double *inSum, const int centralDelta);

// Computes the exponential kernel in place
double GPWExpKernel(const int *cfgA, const int *plaquetteAIdx, const int sizeA,
                    const int *cfgB, const int *plaquetteBIdx, const int sizeB,
                    const int tRSym, const int shift, const int startIdA,
                    const int startIdB, const int plaquetteSize,
                    const double *distWeights, const int numDistWeights,
                    const int *distWeightIdx, const int centralDelta,
                    double *workspace);

/* computes the kernel matrix (exponential kernel) for two lists of configurations
(in bitstring representation) */
void GPWExpKernelMat(const unsigned long *configsAUp,
                     const unsigned long *configsADown, const int *neighboursA,
                     const int sizeA, const int numA,
                     const unsigned long *configsBUp,
                     const unsigned long *configsBDown,
                     const int *neighboursB, const int sizeB, const int numB,
                     const double *distWeights, const int numDistWeights,
                     int **distWeightIdx, const int dim, const int rC,
                     const int tRSym, const int shift, const int startIdA,
                     const int startIdB, const int centralDelta,
                     const int symmetric, double *kernelMatr);

/* computes the kernel vector (exponential kernel) for one reference configuration
and a list of confgurations in bitstring representation) */
void GPWExpKernelVec(const unsigned long *configsAUp,
                     const unsigned long *configsADown, const int *neighboursA,
                     const int sizeA, const int numA, const int *configRef,
                     const int *neighboursRef, const int sizeRef,
                     const double *distWeights, const int numDistWeights,
                     const int *distWeightIdx, const int dim, const int rC,
                     const int tRSym, const int shift, const int startIdA,
                     const int startIdB, const int centralDelta,
                     double *kernelVec);

// Computes the inner sum for the exponential kernel with optimised basis
void ComputeInSumExpBasisOpt(double *inSum, const int *plaquetteAIdx, const int sizeA,
                             const int plaquetteSize, const double complex *distWeights,
                             const int *eleNum, const int tRSym);

// Updates the inner sum for the exponential kernel with optimised basis
void UpdateInSumExpBasisOpt(double *inSumNew, const int *cfgOldReduced, const int *eleNum,
                            const int *plaquetteAIdx, const int sizeA, const int plaquetteSize,
                            const double complex *distWeights, int **plaqHash, const int ri,
                            const int rj, const int tRSym);

// Computes the exponential kernel with optimised basis
double ComputeExpKernelBasisOpt(const int size, const int tRSym,
                                const double *inSum);
#endif // _GPW_EXP_KERN_INCLUDE_FILES
