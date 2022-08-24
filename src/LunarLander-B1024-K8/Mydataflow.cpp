#include "./Mydataflow.h"
#include "./params.h"

extern "C"{

void loadIn(float* input, hls::stream<unroll1in> &outStream){

	unroll1in outStreamEle;
	#pragma HLS ARRAY_PARTITION variable=outStreamEle.a dim=0 complete

    outer_loop:	
    for(int loopTime = 0; loopTime < U1_LOOP; loopTime++){
        load_loop:
		for(int k = 0; k < U1; k++){
		#pragma HLS PIPELINE
			for (int i = 0; i < L0; i++){
			#pragma HLS UNROLL
				outStreamEle.a[k][i] = input[i + k * L0 + loopTime * U1 * L0];
			}
		}
		outStream.write(outStreamEle);
	}

}

void unrollMul1(hls::stream<unroll1in> &inStream, VecL1* weight1, hls::stream<unroll1out> &outStream){

	unroll1in inStreamEle;
	// #pragma HLS ARRAY_PARTITION variable=inStreamEle.a dim=1 complete
	#pragma HLS aggregate variable=inStreamEle
	unroll1out outStreamEle;
	#pragma HLS ARRAY_PARTITION variable=outStreamEle.a dim=0 complete

	outer_loop:
	for(int loopTime = 0; loopTime < U1_LOOP; loopTime++){

		inStreamEle = inStream.read();

		init:
		for(int i = 0; i < P1; i++){
		#pragma HLS UNROLL
			int batch_idx = i / L1;
			int dimension = i % L1;
			outStreamEle.a[batch_idx][dimension] = 0;
		}

		load_weight:
		for (int k = 0; k < L0; k++){
		#pragma HLS PIPELINE
			VecL1 weightVec = weight1[k];
			#pragma HLS aggregate variable=weightVec

			cal_loop:
			for(int i = 0; i < P1; i++){
			#pragma HLS UNROLL
				int batch_idx = i / L1;
				int dimension = i % L1;
				outStreamEle.a[batch_idx][dimension] += inStreamEle.a[batch_idx][k] * weightVec.a[dimension];
			}
		}
		outStream.write(outStreamEle);
	}


}

void act1(hls::stream<unroll1out> &inStream, const float* bias1, hls::stream<unroll2in> &outStream){
	
	unroll1out inStreamEle;
	#pragma HLS ARRAY_PARTITION variable=inStreamEle.a dim=0 complete
	unroll2in outStreamEle;
	#pragma HLS ARRAY_PARTITION variable=outStreamEle.a dim=2 complete
	#pragma HLS ARRAY_PARTITION variable=outStreamEle.a dim=1 cyclic factor=8 // factor=U1

	for(int k = 0; k < U1_LOOP; k++){	// changed
	#pragma HLS PIPELINE
		inStreamEle = inStream.read();
		for(int i = 0; i < P1; i++ ){	// changed
		#pragma HLS UNROLL
			int batch_idx = i / L1;
			int dimension = i % L1;
			float tmp = inStreamEle.a[batch_idx][dimension] + bias1[dimension];
			outStreamEle.a[batch_idx + k * U1][dimension] = (tmp > 0 ? tmp : 0);
		}
	}
	outStream.write(outStreamEle);
}

void unrollMul2(hls::stream<unroll2in> &inStream, VecL2* weight2, hls::stream<unroll2out> &outStream){

	unroll2in inStreamEle;
	// #pragma HLS ARRAY_PARTITION variable=inStreamEle.a dim=1 complete
	#pragma HLS aggregate variable=inStreamEle
	unroll2out outStreamEle;
	#pragma HLS ARRAY_PARTITION variable=outStreamEle.a dim=0 complete

	outer_loop:
	for(int loopTime = 0; loopTime < U2_LOOP; loopTime++){
		inStreamEle = inStream.read();
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

}

void act2(hls::stream<unroll2out> &inStream, const float* bias2, hls::stream<unroll3in> &outStream){
	
	unroll2out inStreamEle = inStream.read();
	#pragma HLS ARRAY_PARTITION variable=inStreamEle.a dim=2 complete
	#pragma HLS ARRAY_PARTITION variable=inStreamEle.a dim=1 cyclic factor=4 // U3
	unroll3in outStreamEle;
	#pragma HLS ARRAY_PARTITION variable=outStreamEle.a dim=0 complete
	

	outer_loop:
	for(int k = 0; k < U2 / U3; k++){
		cal:
		for(int i = 0; i < L2 * U3; i++ ){
		#pragma HLS UNROLL
			int batch_idx = i / L2;
			int dimension = i % L2;
			float tmp = inStreamEle.a[batch_idx + k * U3][dimension] + bias2[dimension];
			outStreamEle.a[batch_idx][dimension] = (tmp > 0 ? tmp : 0);
		}
		outStream.write(outStreamEle);
	}
	
}

void unrollMul3(hls::stream<unroll3in> &inStream, VecL3* weight3, hls::stream<unroll3out> &outStream){

	unroll3in inStreamEle;
	// #pragma HLS ARRAY_PARTITION variable=inStreamEle.a dim=1 complete
	#pragma HLS aggregate variable=inStreamEle
	unroll3out outStreamEle;
	#pragma HLS ARRAY_PARTITION variable=outStreamEle.a dim=0 complete

	outer_loop:
	for(int loopTime = 0; loopTime < U3_LOOP; loopTime++){	// 2 times
		inStreamEle = inStream.read();

		init:
		for(int i = 0; i < P3; i++){
		#pragma HLS UNROLL
			int batch_idx = i / L3;
			int dimension = i % L3;
			outStreamEle.a[batch_idx][dimension] = 0;
		}

		load_weight:
		for (int k = 0; k < L2; k++){
		#pragma HLS PIPELINE
			VecL3 weightVec = weight3[k];
			#pragma HLS aggregate variable=weightVec

			cal_loop:
			for(int i = 0; i < P3; i++){
			#pragma HLS UNROLL
				int batch_idx = i / L3;
				int dimension = i % L3;
				outStreamEle.a[batch_idx][dimension] += inStreamEle.a[batch_idx][k] * weightVec.a[dimension];
			}
		}
		outStream.write(outStreamEle);
	}

}

void act3(hls::stream<unroll3out> &inStream, const float* bias3, hls::stream<outPipe> &outStream){
	
	unroll3out inStreamEle;
	#pragma HLS ARRAY_PARTITION variable=inStreamEle.a dim=0 complete
	outPipe outStreamEle;
	#pragma HLS ARRAY_PARTITION variable=outStreamEle.a dim=0 complete
	

	outer_loop:
	for(int k = 0; k < U3_LOOP; k++){
		inStreamEle = inStream.read();
		cal:
		for(int i = 0; i < P3; i++ ){
		#pragma HLS UNROLL
			int batch_idx = i / L3;
			int dimension = i % L3;
			float tmp = inStreamEle.a[batch_idx][dimension] + bias3[dimension];
			outStreamEle.a[batch_idx][dimension] = (tmp > 0 ? tmp : 0);
		}
		outStream.write(outStreamEle);
	}
	
}

void storeDDR(hls::stream<outPipe> &inStream, float* output){
	
	outPipe inStreamEle;

	for(int loopTime = 0; loopTime < U3_LOOP; loopTime++){
		inStreamEle = inStream.read();
		store_loop:
		for(int i = 0; i < P3; i++){
		#pragma HLS UNROLL //factor = x
    	    int batch_idx = i / L3;
    	    int dimension = i % L3;
			output[i + loopTime * P3] = inStreamEle.a[batch_idx][dimension];
		}
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
	VecL3 weight3[L2];
	
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
	for(int i = 0; i < L2; i++){
		for(int j = 0; j < L3; j++){
			weight3[i].a[j] = w3list[i][j];
		}
	}

    hls::stream<unroll1in> pipeIn;
    hls::stream<unroll1out> pipeL1;
    hls::stream<unroll2in> pipeIn2;
    hls::stream<unroll2out> pipeL2;
	hls::stream<unroll3in> pipeIn3;
	hls::stream<unroll3out> pipeL3;
	hls::stream<outPipe> pipeOut;
	

	main_loop:
	for(int i = 0; i < LOOP_N; i++){
	#pragma HLS DATAFLOW
		loadIn(&input[i * L0 * TP_K], pipeIn);
		unrollMul1(pipeIn, weight1, pipeL1);
		act1(pipeL1, bias1, pipeIn2);
		unrollMul2(pipeIn2, weight2, pipeL2);
		act2(pipeL2, bias2, pipeIn3);
		unrollMul3(pipeIn3, weight3, pipeL3);
		act3(pipeL3, bias3, pipeOut);
		storeDDR(pipeOut, &output[i * L3 * TP_K]);
	}

}
}
