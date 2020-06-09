#ifndef _REFSTATE
#define _REFSTATE
int ComputeRefState(const int *eleIdx, const int *eleNum, int *workspace);

int UpdateRefState(const int *eleIdx, const int *eleNum, int *workspace);

int ComputeRefState_fsz(const int *eleIdx, const int *eleNum, const int *eleSpn, int *workspace);

int UpdateRefState_fsz(const int *eleIdx, const int *eleNum, const int *eleSpn, int *workspace);

#endif
