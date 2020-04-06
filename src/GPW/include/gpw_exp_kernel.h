// TODO include license and description

#ifndef _GPW_EXP_KERN_INCLUDE_FILES
#define _GPW_EXP_KERN_INCLUDE_FILES

#include "complex.h"

// Computes the inner sum for the exponential kernel
void ComputeInSumExp(double *inSum, const int *plaquetteAIdx,
                     const int sizeA, const int *plaquetteBIdx, const int sizeB,
                     const int plaquetteSize, const double complex *distWeights,
                     const int *distWeightIdx);

// Updates the inner sum for the exponential kernel
void UpdateInSumExp(double *inSumNew, const double *inSumOld,
                    const int *plaquetteAIdx, const int sizeA,
                    const int *plaquetteBIdx, const int sizeB,
                    const int plaquetteSize, const double complex *distWeights,
                    const int *distWeightIdx, int **plaqHash,
                    int *plaqHashSz, const int siteA,
                    const int siteB);


// Computes the exponential kernel
double ComputeExpKernel(const int sizeA, const int sizeB, const int tRSym,
                        const int shift, const int startIdA, const int startIdB,
                        const double *inSum, const double *inSumFlipped);

// Computes the exponential kernel in place
double GPWExpKernel(const int *cfgA, const int *plaquetteAIdx, const int sizeA,
                    const int *cfgB, const int *plaquetteBIdx, const int sizeB,
                    const int tRSym, const int shift, const int startIdA,
                    const int startIdB, const int plaquetteSize,
                    const double *distWeights, const int numDistWeights,
                    const int *distWeightIdx, double *workspace);

/* computes the kernel matrix (exponential kernel) for two lists of configurations
(in bitstring representation) */
void GPWExpKernelMat(const unsigned long *configsAUp,
                     const unsigned long *configsADown, const int *neighboursA,
                     const int sizeA, const int numA,
                     const unsigned long *configsBUp,
                     const unsigned long *configsBDown,
                     const int *neighboursB, const int sizeB, const int numB,
                     const double *distWeights, const int numDistWeights,
                     const int **distWeightIdx, const int dim, const int rC,
                     const int tRSym, const int shift, const int startIdA,
                     const int startIdB, const int symmetric, double *kernelMatr);

/* computes the kernel vector (exponential kernel) for one reference configuration
and a list of confgurations in bitstring representation) */
void GPWExpKernelVec(const unsigned long *configsAUp,
                     const unsigned long *configsADown, const int *neighboursA,
                     const int sizeA, const int numA, const int *configRef,
                     const int *neighboursRef, const int sizeRef,
                     const double *distWeights, const int numDistWeights,
                     const int *distWeightIdx, const int dim, const int rC,
                     const int tRSym, const int shift, const int startIdA,
                     const int startIdB, double *kernelVec);
#endif // _GPW_EXP_KERN_INCLUDE_FILES