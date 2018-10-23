#ifndef _LSLOCGRN_REAL
#define _LSLOCGRN_REAL

void LSLocalQ_real(const double h1, const double ip, int *eleIdx, int *eleCfg, int *eleNum, int *eleProjCnt, double *eleGPWKern, double *_LSLQ_real);

double calculateHK_real(const double h1, const double ip, int *eleIdx, int *eleCfg, int *eleNum, int *eleProjCnt, double *eleGPWKern);

double calHCA_real(const int ri, const int rj, const int s,
                   const double h1, const double ip, int *eleIdx, int *eleCfg,
                   int *eleNum, int *eleProjCnt, double *eleGPWKern);

double checkGF1_real(const int ri, const int rj, const int s, const double ip,
                int *eleIdx, const int *eleCfg, int *eleNum);

double calHCA1_real(const int ri, const int rj, const int s,
               const double ip, int *eleIdx, int *eleCfg, int *eleNum, int *eleProjCnt, double *eleGPWKern);
double calHCA2_real(const int ri, const int rj, const int s,
                       const double ip, int *eleIdx, int *eleCfg, int *eleNum, int *eleProjCnt, double *eleGPWKern);

double calculateHW_real(const double h1, const double ip, int *eleIdx, int *eleCfg,
                           int *eleNum, int *eleProjCnt, double *eleGPWKern);

double calHCACA_real(const int ri, const int rj, const int rk, const int rl,
                        const int si,const int sk,
                        const double h1, const double ip, int *eleIdx, int *eleCfg,
                        int *eleNum, int *eleProjCnt, double *eleGPWKern);

double checkGF2_real(const int ri, const int rj, const int rk, const int rl,
                     const int s, const int t, const double ip,
                     int *eleIdx, const int *eleCfg, int *eleNum);

double calHCACA1_real(const int ri, const int rj, const int rk, const int rl,
                 const int si,const int sk,
                 const double ip, int *eleIdx, int *eleCfg,
                 int *eleNum, int *eleProjCnt, double *eleGPWKern);

double calHCACA2_real(const int ri, const int rj, const int rk, const int rl,
                      const int si,const int sk,
                      const double ip, int *eleIdx, int *eleCfg, int *eleNum, int *eleProjCnt, double *eleGPWKern);

void LSLocalCisAjs_real(const double h1, const double ip, int *eleIdx, int *eleCfg, int *eleNum, int *eleProjCnt, double *eleGPWKern);

void copyMAll_real(double *invM_from, double *pfM_from, double *invM_to, double *pfM_to);

#endif
