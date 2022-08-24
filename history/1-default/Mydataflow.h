/*
Modified from BlockMMEntry.
Use float as interface and pass data one by one.
Has a long initation and writeout for every function.
*/

#include "hls_stream.h"
#include "hls_math.h"
#include <iostream>
#include <iomanip>
#include <vector>

extern "C"{

using namespace std;

#define BSIZE 16
#define L0 2
#define L1 128
#define L2 1
#define P0 2        // UNROLL factor = L0/P0, P0 in parallel
#define P1 16
#define P2 1

typedef struct {
	float a[L1];
} VecL1;  // Feature Vector

typedef struct{
	float a[L2];
} VecL2;


// typedef int D_INT;
// typedef float D_FLOAT;

void loadIn(float *input, hls::stream<float> &outStream);
void matmul1(hls::stream<float> &inStream, VecL1 *weight1, hls::stream<float> &outStream);
void matmul2(hls::stream<float> &inStream, VecL2 *weight2, hls::stream<float> &outStream);
void activation(hls::stream<float> &inStream, const float* bias, const int L, hls::stream<float> &outStream);
void storeDDR(hls::stream<float> &inStream, float* output);
void top(float* input, float* output);

}
