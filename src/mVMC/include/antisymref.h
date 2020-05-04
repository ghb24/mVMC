#ifndef _REFSTATE
#define _REFSTATE
int ComputeRefState(const int *eleIdx, const int *eleNum, int *workspace);

int UpdateRefState(const int *eleIdx, const int *eleNum, int *workspace);

#endif
