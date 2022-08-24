#include "./Mydataflow.h"
#include "./params.h"

extern "C"{

void loadIn(float* input, hls::stream<block1in> &outStream){
	block1in outStreamEle;
	#pragma HLS ARRAY_PARTITION variable=outStreamEle.a cyclic factor=4	// P0

    outer_loop:	
    for(int loopTime = 0; loopTime < TP_K; loopTime++){
        load_loop:
        for (int i = 0; i < L0 / P0; i++){
	    #pragma HLS PIPELINE
	    	for(int ii = 0; ii < P0; ii++){
	    		outStreamEle.a[i * P0 + ii] = input[i * P0 + ii + loopTime * L0];	
	    	}
	    }
	    outStream.write(outStreamEle);
    }
}

void blockMul1(hls::stream<block1in> &inStream, VecL1* weight1, hls::stream<block1out> &outStream){

	block1in inStreamEle;
	#pragma HLS aggregate variable=inStreamEle
	block1out outStreamEle;
	#pragma HLS ARRAY_PARTITION variable=outStreamEle.a cyclic factor=16   // P1

	for(int loopTime = 0; loopTime < U1_LOOP; loopTime++){
		
		inStreamEle = inStream.read();

		init:
		for(int i = 0; i < L1 / P1;i++){
		#pragma HLS PIPELINE
			for(int ii = 0; ii < P1; ii++)
				outStreamEle.a[i * P1 + ii] = 0;
		}
    
		load_weight:
		for (int k = 0; k < L0; k++){  

			VecL1 weightVec = weight1[k];
			#pragma HLS aggregate variable=weightVec
    
			block_loop:
			for (int i = 0; i < L1 / P1; i++){
			#pragma HLS PIPELINE
			#pragma HLS dependence variable=outStreamEle.a inter false
    
		    		ele_loop:
	    			for (int ii = 0; ii < P1; ii++){
	    			#pragma HLS UNROLL
	    				outStreamEle.a[i * P1 + ii] += inStreamEle.a[k] * weightVec.a[i * P1 + ii];
				}
	    	}
		}
		outStream.write(outStreamEle);
	}
}

void act1(hls::stream<block1out> &inStream, const float* bias1, hls::stream<unroll2in> &outStream){

	block1out inStreamEle;
	#pragma HLS ARRAY_PARTITION variable=inStreamEle.a cyclic factor=16		  // P1, less resources
	unroll2in outStreamEle;
	#pragma HLS ARRAY_PARTITION variable=outStreamEle.a dim=2 cyclic factor=16   // P1, directly compact

	outer_loop:
	for(int k = 0; k < U1_LOOP; k++){	
		inStreamEle = inStream.read();
		
		inner_loop:
		for(int i = 0; i < L1 / P1; i++){
	    	#pragma HLS PIPELINE
	    		for(int ii = 0; ii < P1; ii++){	// special
	    			float tmp = inStreamEle.a[i * P1 + ii] + bias1[i * P1 + ii];
	    			outStreamEle.a[k][i * P1 + ii] = (tmp > 0 ? tmp : 0);
	    		}
        	}
	}
	
	outStream.write(outStreamEle);	// must write out in one pack, thus mm2 could deal with it in parallel
}

void unrollMul2(hls::stream<unroll2in> &inStream, VecL2* weight2, hls::stream<unroll2out> &outStream){

	unroll2in inStreamEle = inStream.read();
	#pragma HLS ARRAY_PARTITION variable=inStreamEle.a dim=0 complete
	unroll2out outStreamEle;
	#pragma HLS ARRAY_PARTITION variable=outStreamEle.a dim=0 complete

	init:
	for(int i = 0; i < P2; i++){
	#pragma HLS UNROLL
		int batch_idx = i / L2;
		int dimension = i % L2;
		outStreamEle.a[batch_idx][dimension] = 0;
	}

	load_weight:
	for (int k = 0; k < L1; k++){
	#pragma HLS PIPELINE
		VecL2 weightVec = weight2[k];
		#pragma HLS aggregate variable=weightVec

		cal_loop:
		for(int i = 0; i < P2; i++){
		#pragma HLS UNROLL
			int batch_idx = i / L2;
			int dimension = i % L2;
			outStreamEle.a[batch_idx][dimension] += inStreamEle.a[batch_idx][k] * weightVec.a[dimension];
		}
	}
	outStream.write(outStreamEle);
}

void act2(hls::stream<unroll2out> &inStream, const float* bias2, hls::stream<outPipe> &outStream){
	
	unroll2out inStreamEle = inStream.read();
	#pragma HLS ARRAY_PARTITION variable=inStreamEle.a dim=0 complete
	outPipe outStreamEle;
	#pragma HLS ARRAY_PARTITION variable=outStreamEle.a dim=0 complete

	for(int i = 0; i < P2; i++ ){
	#pragma HLS UNROLL
		int batch_idx = i / L2;
		int dimension = i % L2;
		float tmp = inStreamEle.a[batch_idx][dimension] + bias2[dimension];
		outStreamEle.a[batch_idx][dimension] = (tmp > 0 ? tmp : 0);
	}
	outStream.write(outStreamEle);
}

void storeDDR(hls::stream<outPipe> &inStream, float* output){
	
	outPipe inStreamEle = inStream.read();

	store_loop:
	for(int i = 0; i < P2; i++){
	#pragma HLS UNROLL //factor = x
        int batch_idx = i / L2;
        int dimension = i % L2;
		output[i] = inStreamEle.a[batch_idx][dimension];
	}
}

void top(float* input, float* output){
	#pragma HLS INTERFACE m_axi port=input bundle=gmem0 offset=slave
	#pragma HLS INTERFACE m_axi port=output bundle=gmem1 offset=slave
	#pragma HLS INTERFACE s_axilite port=input bundle=control
	#pragma HLS INTERFACE s_axilite port=output bundle=control
	#pragma HLS INTERFACE s_axilite port=return bundle=control
	
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

    hls::stream<block1in> pipeIn;
    hls::stream<block1out> pipeL1;
    hls::stream<unroll2in> pipeIn2;
    hls::stream<unroll2out> pipeL2;
	hls::stream<outPipe> pipeOut;
	
	for(int i = 0; i < LOOP_N; i++){
	#pragma HLS DATAFLOW
		loadIn(&input[i * L0 * TP_K], pipeIn);
		blockMul1(pipeIn, weight1, pipeL1);
		act1(pipeL1, bias1, pipeIn2);
		unrollMul2(pipeIn2, weight2, pipeL2);
		act2(pipeL2, bias2, pipeOut);
		storeDDR(pipeOut, &output[i * L2 * TP_K]);
	}
	
}

}