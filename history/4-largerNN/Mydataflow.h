/*
Tested a larger network. The codes can be easily changed to suit different cases.
Most comprehensive: all the structs are 2-D array. (Actually, unnecessary according to analysis).
*/
#include "hls_stream.h"
#include "hls_math.h"
#include <iostream>
#include <iomanip>
#include <vector>

extern "C"{

using namespace std;

// Network: 4 - 128 - 2

#define BSIZE 16
#define L0 4        // dimension of input layer
#define L1 128      // dimension of hidden layer
#define L2 2        // dimension of output layer
#define P0 2        // UNROLL factor of input layer(L0)
#define P1 64       // UNROLL factor of hidden layer(L1)
#define P2 16       // UNROLL factor of output layer(L2)
#define U1 1        // batch size of the hidden layer(usually = 1, except 
                    // the resources are many enough to support block loadin)
#define U2 8        // batch size of output layer(depend on Pi and Li)
#define UMAX 8      // UMAX = max{Ui} = max{U1, U2}
#define U1LOOP UMAX / U1
#define U2LOOP UMAX / U2
const int iteration_time = BSIZE / UMAX;


// STREAM IN/OUT PIPE
typedef struct{
    float a[U1][L0];
} MM1in;

typedef struct {
    float a[U1][L1];
} MM1out;

typedef struct{
    float a[U2][L1];
} MM2in;

typedef struct{
    float a[U2][L2];
} MM2out;

typedef MM2out Blockout;        
// Blockout is the same as final layer's output


// WEIGHT
typedef struct{
    float a[L1];
} VecL1;

typedef struct{
    float a[L2];
} VecL2;

//                          [U1][L0] = [1][4]
void loadIn(float* input, hls::stream<MM1in> &outStream);
//              [U1][L0] = [1][4]                               [U1][L1] = [1][128]
void matmul1(hls::stream<MM1in> &inStream, VecL1* weight1, hls::stream<MM1out> &outStream);
//              [U1][L1] = [1][128]                             [U2][L1] = [8][128]
void act1(hls::stream<MM1out> &inStream, const float* bias1, hls::stream<MM2in> &outStream);
//              [U2][L1] = [8][128]                             [U2][L2] = [8][2]
void matmul2(hls::stream<MM2in> &inStream, VecL2* weight2, hls::stream<MM2out> &outStream);
//              [U2][L2] = [8][2]                               [U2][L2] = [8][2]                      
void act2(hls::stream<MM2out> &inStream, const float* bias2, hls::stream<Blockout> &outStream);
//              [U2][L2] = [8][2] 
void storeDDR(hls::stream<Blockout> &inStream, float* output);
// top function utilizing all the funs above
void top(float* input, float* output);

}
