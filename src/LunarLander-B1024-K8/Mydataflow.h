#include "hls_stream.h"
#include "hls_math.h"
#include <iostream>
#include <iomanip>
#include <vector>

extern "C"{

using namespace std;
// Network architecture
#define BSIZE 1024
#define L0 64               
#define L1 64               
#define L2 32               
#define L3 4

// Performance: assigned by users
#define TP_K 8             // deal with TP_K samples in every outer_loop
#define LOOP_N BSIZE / TP_K // number of outer loop

// Hardware: can be calculate
#define P0 64                // UNROLL factor of input layer(L0)
#define P1 512               // UNROLL factor of hidden layer(L1)
#define P2 256               // UNROLL factor of output layer(L2)
#define P3 16
#define U1 (P1 / L1)        // U1 = 2
#define U1_LOOP (TP_K / U1) // looptime = 1
#define U2 (P2 / L2)        // U2 = 2
#define U2_LOOP (TP_K / U2) // looptime = 1
#define U3 (P3 / L3)        // U1 = 1
#define U3_LOOP (TP_K / U3) // looptime = 2

// STREAM IN/OUT PIPE
typedef struct{
    float a[U1][L0];
} unroll1in;

typedef struct{
    float a[U1][L1];
} unroll1out;

typedef struct{
    float a[U2][L1];
} unroll2in;            

typedef struct{
    float a[U2][L2];
} unroll2out;           

typedef struct{
    float a[U3][L2];
} unroll3in;            

typedef struct{
    float a[U3][L3];
} unroll3out;  

typedef unroll3out outPipe;        
// Blockout is the same as final layer's output


// WEIGHT
typedef struct{
    float a[L1];
} VecL1;

typedef struct{
    float a[L2];
} VecL2;

typedef struct{
    float a[L3];
} VecL3;


//                                [U1][L0] = [2][64]
void loadIn(float* input, hls::stream<unroll1in> &outStream);
//              [U1][L0] = [2][64]                               [U1][L1] = [2][64]
void unrollMul1(hls::stream<unroll1in> &inStream, VecL1* weight1, hls::stream<unroll1out> &outStream);
//              [U1][L1] = [2][64]                               [U2][L1] = [2][64]
void act1(hls::stream<unroll1out> &inStream, const float* bias1, hls::stream<unroll2in> &outStream);
//              [U2][L1] = [2][64]                             [U2][L2] = [2][32]
void unrollMul2(hls::stream<unroll2in> &inStream, VecL2* weight2, hls::stream<unroll2out> &outStream);
//              [U2][L2] = [2][32]                               [U3][L2] = [1][32]                      
void act2(hls::stream<unroll2out> &inStream, const float* bias2, hls::stream<unroll3in> &outStream);
//              [U3][L2] = [1][32]                               [U3][L3] = [1][4]
void unrollMul3(hls::stream<unroll3in> &inStream, VecL3* weight3, hls::stream<unroll3out> &outStream);
//              [U3][L3] = [1][4]                                 [U3][L3] = [1][4]                      
void act3(hls::stream<unroll3out> &inStream, const float* bias3, hls::stream<outPipe> &outStream);
//              [U3][L3] = [1][4] 
void storeDDR(hls::stream<outPipe> &inStream, float* output);
// top function utilizing all the funs above
void top(float* input, float* output);

}
