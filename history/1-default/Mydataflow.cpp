#include "./Mydataflow.h"
#include "./params.h"
extern "C"{
/*
 * @brief: load off-chip data(1,L1) to inStream
 * @input: 2-D input -> 1-D input, batch first.
 */
void loadIn(float *input, hls::stream<float> &outStream)
{
    loop_load:
    for (int i = 0; i < L0; i++){ // using macro L1
    	#pragma HLS PIPELINE
        outStream.write(input[i]);
    }
}

void matmul1(hls::stream<float> &inStream, VecL1 *weight1, hls::stream<float> &outStream){
#pragma HLS aggregate variable=weight1

    float tmp[L1 / P1][P1] = {0};
    #pragma HLS ARRAY_PARTITION variable=tmp dim=2 complete

    stream_loop:
    for (int k = 0; k < L0; k++){
        
		float tmpA = inStream.read();
        VecL1 tmpB = weight1[k];
        #pragma HLS aggregate variable=tmpB
		
		block_loop:
        for (int i = 0; i < L1 / P1; i++){
		#pragma HLS PIPELINE
        #pragma HLS dependence variable=tmp inter false
            ele_loop:
            for (int ii = 0; ii < P1; ii++){
	    #pragma HLS UNROLL
                tmp[i][ii] += tmpA * tmpB.a[i * P1 + ii];
            }
        }
    }

    write_out:
    for(int i = 0; i < L1 / P1; i++){
	for(int ii = 0; ii < P1; ii++){
	#pragma HLS PIPELINE
            outStream.write(tmp[i][ii]);
        }
    }
}

void matmul2(hls::stream<float> &inStream, VecL2 *weight2, hls::stream<float> &outStream){
#pragma HLS aggregate variable=weight2
    float tmp[L2 / P2][P2] = {0};   //DEFINE
    #pragma HLS ARRAY_PARTITION variable=tmp dim=2 complete

    stream_loop:
    for (int k = 0; k < L1; k++){   // DEFINE
        float tmpA = inStream.read();
        VecL2 tmpB = weight2[k];
        #pragma HLS aggregate variable=tmpB
	block_loop:
        for (int i = 0; i < L2 / P2; i++){  //DEFINE
	#pragma HLS PIPELINE
        #pragma HLS dependence variable=tmp inter false            
	    ele_loop:
            for (int ii = 0; ii < P2; ii++){
	    #pragma HLS UNROLL
                tmp[i][ii] += tmpA * tmpB.a[i * P2 + ii];
            }
        }
    }

    write_out:  //DEFINE
    for(int i = 0; i < L2 / P2; i++){
        for(int ii = 0; ii < P2; ii++){
	#pragma HLS PIPELINE
            outStream.write(tmp[i][ii]);
        }
    }
}

void activation(hls::stream<float> &inStream, const float* bias, const int L, hls::stream<float> &outStream){
	for (int i = 0; i < L; i++){
	#pragma HLS PIPELINE
		float tmp = inStream.read()+bias[i];
		outStream.write(tmp > 0 ? tmp : 0);
	}
}

void storeDDR(hls::stream<float> &inStream, float* output){
    store_loop:
    for(int i = 0; i < L2; i++){
        output[i] = inStream.read();
    }
}

void top(float* input, float* output){
	#pragma HLS INTERFACE m_axi port=input bundle=gmem0 offset=slave
	#pragma HLS INTERFACE m_axi port=output bundle=gmem1 offset=slave
	#pragma HLS INTERFACE s_axilite port=input bundle=control
	#pragma HLS INTERFACE s_axilite port=output bundle=control
	#pragma HLS INTERFACE s_axilite port=return bundle=control

    hls::stream<float> inpipe;
    hls::stream<float> outpipe[4];

    VecL1 weight1[L0];
    VecL2 weight2[L1];

    for(int i = 0; i < L0; i++){
        for(int j = 0; j < L1; j++){
            weight1[i].a[j] = w1list[i][j];
        }
    }
    for(int i = 0; i < L1; i++){
        for(int j = 0; j < L2; j++){
            weight2[i].a[j] = w2list[i][j];
        }
    }
	
    for(int i = 0; i < BSIZE; i++){
	#pragma HLS DATAFLOW
    	//cout << "## index = " << i << endl;
    	loadIn(&input[i * L0], inpipe);
    	matmul1(inpipe, weight1, outpipe[0]);
    	activation(outpipe[0], bias1, L1, outpipe[1]);
    	matmul2(outpipe[1], weight2, outpipe[2]);
    	activation(outpipe[2], bias2, L2, outpipe[3]);
    	storeDDR(outpipe[3], &output[i * L2]);
    }

}
}
