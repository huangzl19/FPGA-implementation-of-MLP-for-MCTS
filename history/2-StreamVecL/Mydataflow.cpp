#include "./Mydataflow.h"
#include "./params.h"

extern "C"{

void loadIn(float *input, hls::stream<VecL0> &outStream){
	VecL0 tmp;
	#pragma HLS ARRAY_PARTITION variable=tmp.a cyclic factor=2	// p0
	// #pragma HLS aggregate variable=tmp
	
	loop_load:
	for (int i = 0; i < L0/P0; i++){
	#pragma HLS PIPELINE
		for(int ii = 0; ii < P0; ii++){
			tmp.a[i * P0 + ii] = input[i * P0 + ii];
		}
	}
	outStream.write(tmp);
}

void matmul1(hls::stream<VecL0> &inStream, VecL1* weight1, hls::stream<VecL1> &outStream){
//#pragma HLS ARRAY_PARTITION variable=weight1

	float tmp[L1 / P1][P1];
	#pragma HLS ARRAY_PARTITION variable=tmp dim=2 complete
	VecL0 tmpA = inStream.read();
	#pragma HLS aggregate variable=tmpA

	init:
	for(int i = 0; i < L1 / P1;i++){
	#pragma HLS PIPELINE
		for(int ii = 0; ii < P1; ii++)
			tmp[i][ii] = 0;
	}
    
	load_weight:
	for (int k = 0; k < L0; k++){  

		VecL1 tmpB = weight1[k];
		#pragma HLS aggregate variable=tmpB
		
		block_loop:
		for (int i = 0; i < L1 / P1; i++){
		#pragma HLS PIPELINE
		#pragma HLS dependence variable=tmp inter false

			ele_loop:
			for (int ii = 0; ii < P1; ii++){
			#pragma HLS UNROLL
				tmp[i][ii] += tmpA.a[k] * tmpB.a[i * P1 + ii];
			}
		}
	}

	VecL1 tmpC;
	#pragma HLS ARRAY_PARTITION variable=tmpC.a cyclic factor=8 //P1
    
	write_out:
	for(int i = 0; i < L1 / P1; i++){
	#pragma HLS PIPELINE
		for(int ii = 0; ii < P1; ii++){
			tmpC.a[i * P1 + ii] = tmp[i][ii];
		}
	}
	outStream.write(tmpC);
}

void matmul2(hls::stream<VecL1> &inStream, VecL2* weight2, hls::stream<VecL2> &outStream){
#pragma HLS aggregate variable=weight2

	float tmp[L2 / P2][P2];   
	#pragma HLS ARRAY_PARTITION variable=tmp dim=2 complete
	VecL1 tmpA = inStream.read();
	#pragma HLS aggregate variable=tmpA

	init:
	for(int i = 0; i < L2 / P2;i++){
	#pragma HLS PIPELINE
		for(int ii = 0; ii < P2; ii++)
			tmp[i][ii] = 0;
	}

	load_weight:
	for (int k = 0; k < L1; k++){
	
		VecL2 tmpB = weight2[k];
		#pragma HLS aggregate variable=tmpB

		block_loop:
		for (int i = 0; i < L2 / P2; i++){
		#pragma HLS PIPELINE
		#pragma HLS dependence variable=tmp inter false

			ele_loop:
			for (int ii = 0; ii < P2; ii++){
			#pragma HLS UNROLL
				tmp[i][ii] += tmpA.a[k] * tmpB.a[i * P2 + ii];
			}
		}
	}

	VecL2 tmpC;
	#pragma HLS aggregate variable=tmpC
	
	write_out: 
	for(int i = 0; i < L2 / P2; i++){
	#pragma HLS PIPELINE
		for(int ii = 0; ii < P2; ii++){
			tmpC.a[i * P2 + ii] = tmp[i][ii];
		}
	}
	outStream.write(tmpC);
}

void act1(hls::stream<VecL1> &inStream, const float* bias1, hls::stream<VecL1> &outStream){
#pragma HLS aggregate variable=bias1

	VecL1 tmpA = inStream.read();
	VecL1 tmpB;
	#pragma HLS aggregate variable=tmpA
	#pragma HLS ARRAY_PARTITION variable=tmpB.a cyclic factor=8    
	
	for(int i = 0; i < L1 / P1; i++){
	#pragma HLS PIPELINE
		for(int ii = 0; ii < P1; ii++){
			float tmp = tmpA.a[i * P1 + ii] + bias1[i * P1 + ii];
			tmpB.a[i * P1 + ii] = (tmp > 0 ? tmp : 0);
		}
	}
	outStream.write(tmpB);
}

void act2(hls::stream<VecL2> &inStream, const float* bias2, hls::stream<VecL2> &outStream){
#pragma HLS aggregate variable=bias2
	
	VecL2 tmpA = inStream.read();
	VecL2 tmpB;
	#pragma HLS aggregate variable=tmpA
	#pragma HLS aggregate variable=tmpB
	
	for(int i = 0; i < L2 / P2; i++){
	#pragma HLS PIPELINE
		for(int ii = 0; ii < P2; ii++){
			float tmp = tmpA.a[i * P2 + ii] + bias2[i * P2 + ii];
			tmpB.a[i * P2 + ii] = (tmp > 0 ? tmp : 0);
		}
	}
	outStream.write(tmpB);
}

void storeDDR(hls::stream<VecL2> &inStream, float* output){
	
	VecL2 tmpA = inStream.read();
	#pragma HLS aggregate variable=tmpA
	
	store_loop:
	for(int i = 0; i < L2; i++){
	#pragma HLS UNROLL //factor = x
		output[i] = tmpA.a[i];
	}
}

void top(float* input, float* output){
	#pragma HLS INTERFACE m_axi port=input bundle=gmem0 offset=slave
	#pragma HLS INTERFACE m_axi port=output bundle=gmem1 offset=slave
	#pragma HLS INTERFACE s_axilite port=input bundle=control
	#pragma HLS INTERFACE s_axilite port=output bundle=control
	#pragma HLS INTERFACE s_axilite port=return bundle=control
	
	hls::stream<VecL0> pipeL0;
	hls::stream<VecL1> pipeL1[2];
	hls::stream<VecL2> pipeL2[2];
	
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
		loadIn(&input[i * L0], pipeL0);
		matmul1(pipeL0, weight1, pipeL1[0]);
		act1(pipeL1[0], bias1, pipeL1[1]);
		matmul2(pipeL1[1], weight2, pipeL2[0]);
		act2(pipeL2[0], bias2, pipeL2[1]);
		storeDDR(pipeL2[1], &output[i * L2]);
	}
	
}
}
