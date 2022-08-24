/*
Modified from default. Modified the interface units and fixed the initiation and writeout.
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
#define P1 8
#define P2 1

typedef struct{
	float a[L0];
} VecL0;

typedef struct {
	float a[L1];
} VecL1;  // Feature Vector

typedef struct{
	float a[L2];
} VecL2;


// typedef int D_INT;
// typedef float D_FLOAT;

void loadIn(float* input, hls::stream<VecL0> &outStream);
void matmul1(hls::stream<VecL0> &inStream, VecL1* weight1, hls::stream<VecL1> &outStream);
void matmul2(hls::stream<VecL1> &inStream, VecL2* weight2, hls::stream<VecL2> &outStream);
void act1(hls::stream<VecL1> &inStream, const float* bias1, hls::stream<VecL1> &outStream);
void act2(hls::stream<VecL2> &inStream, const float* bias2, hls::stream<VecL2> &outStream);
void storeDDR(hls::stream<VecL2> &inStream, float* output);
void top(float* input, float* output);

}
