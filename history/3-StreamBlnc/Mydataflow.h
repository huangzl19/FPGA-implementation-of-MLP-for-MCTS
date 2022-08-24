/*
Use more harware to accelerate the cumputation.
Tiny optimizations.
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
#define P2 4
#define UNIT_SIZE 4
#define UNIT_NUM BSIZE/UNIT_SIZE    // UNIT_NUM = 4

/*
typedef struct{
    float a[L0];
} VecL0;
*/

typedef struct {
    float a[L1];
} VecL1;  // Feature Vector

typedef struct{
    float a[L2];
} VecL2;


typedef struct{
    float a[L0];
} Block1in;

typedef struct{
    float a[UNIT_SIZE][L1];
} Block2in;


typedef struct{
    float a[UNIT_SIZE][L2];
} Block3in;


// typedef int D_INT;
// typedef float D_FLOAT;

void loadIn(float* input, hls::stream<Block1in> &outStream);
void matmul1(hls::stream<Block1in> &inStream, VecL1* weight1, hls::stream<VecL1> &outStream);
void act1(hls::stream<VecL1> &inStream, const float* bias1, hls::stream<Block2in> &outStream);   //act and aggregate
void matmul2(hls::stream<Block2in> &inStream, VecL2* weight2, hls::stream<Block3in> &outStream);
void act2(hls::stream<Block3in> &inStream, const float* bias2, hls::stream<Block3in> &outStream);
void storeDDR(hls::stream<Block3in> &inStream, float* output);
void top(float* input, float* output);

}