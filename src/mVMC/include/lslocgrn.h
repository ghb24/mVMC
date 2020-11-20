#ifndef _LSLOCGRN_CMP
#define _LSLOCGRN_CMP
#include <complex.h>
#include "global.h"

void LSLocalQ(const double complex h1, const double complex ip, int *eleIdx, int *eleCfg, int *eleNum, int *eleProjCnt,
              double complex *eleGPWKern, double complex *eleGPWInSum, double complex *_LSLQ);

void LSLocalCisAjs(const double complex h1, const double complex ip, int *eleIdx, int *eleCfg, int *eleNum, int *eleProjCnt,
                   double complex *eleGPWKern, double complex *eleGPWInSum);


#endif
