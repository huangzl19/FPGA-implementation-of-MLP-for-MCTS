#include "hls_stream.h"
#include "hls_math.h"
#include <iostream>
#include <iomanip>
#include <vector>

extern "C"{

using namespace std;

// Network: 4 - 128 - 2
#define BSIZE 1024
#define L0 4                // dimension of input layer
#define L1 32               // dimension of hidden layer
#define L2 2                // dimension of output layer

// Performance: assigned by users
#define TP_K 4              // deal with TP_K samples in every outer_loop
#define LOOP_N BSIZE / TP_K // number of outer loop

// Hardware: can be calculate
#define P0 4                // UNROLL factor of input layer(L0)
#define P1 16               // UNROLL factor of hidden layer(L1)
#define P2 8                // UNROLL factor of output layer(L2)
#define U1 1                // U is not necessary for blockMul
#define U1_LOOP (TP_K / U1)
#define U2 (P2 / L2)        // batch size of mm2
#define U2_LOOP (TP_K / U2)


// STREAM IN/OUT PIPE
typedef struct{
    float a[L0];
} block1in;             // blockMul input, 1-D vector with dimension L0

typedef struct {
    float a[L1];
} block1out;            // blockMul output, 1-D vector with dimension L1

typedef struct{
    float a[U2][L1];
} unroll2in;            // unrollMul input, 2-D vector with U2 samples L1 dimensions

typedef struct{
    float a[U2][L2];
} unroll2out;           // unrollMul input, 2-D vector with U2 samples L2 dimensions

typedef unroll2out outPipe;        
// Blockout is the same as final layer's output


// WEIGHT
typedef struct{
    float a[L1];
} VecL1;

typedef struct{
    float a[L2];
} VecL2;

//                                  [L0] = [4]
void loadIn(float* input, hls::stream<block1in> &outStream);
//void loadIn(float* input, hls::stream<unroll1in> &outStream);
//                  [L0] = [4]                                      [L1] = [128]
void blockMul1(hls::stream<block1in> &inStream, VecL1* weight1, hls::stream<block1out> &outStream);
//                  [L1] = [128]                                 [U2][L1] = [8][128]
void act1(hls::stream<block1out> &inStream, const float* bias1, hls::stream<unroll2in> &outStream);
//              [U2][L1] = [8][128]                             [U2][L2] = [8][2]
void unrollMul2(hls::stream<unroll2in> &inStream, VecL2* weight2, hls::stream<unroll2out> &outStream);
//              [U2][L2] = [8][2]                               [U2][L2] = [8][2]                      
void act2(hls::stream<unroll2out> &inStream, const float* bias2, hls::stream<outPipe> &outStream);
//              [U2][L2] = [8][2] 
void storeDDR(hls::stream<outPipe> &inStream, float* output);
// top function utilizing all the funs above
void top(float* input, float* output);

}
