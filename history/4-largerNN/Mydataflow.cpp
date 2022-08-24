#include "./Mydataflow.h"
#include "./params.h"

// 4-128-2

extern "C"{

void loadIn(float* input, hls::stream<MM1in> &outStream){
	
	MM1in outStreamEle;
	#pragma HLS ARRAY_PARTITION variable=outStreamEle.a dim=2 cyclic factor=4	// P0
	
	outer_loop:	
	for(int loopTime = 0; loopTime < UMAX; loopTime++){
		load_loop:
		for (int i = 0; i < L0 / P0; i++){
		#pragma HLS PIPELINE
			for(int ii = 0; ii < P0; ii++){
				// float batch_indx = ;
				// float dimension = 
				// first easy part, 1st dimension of outStreamEle = 0
				// then hard, 1st dimension of outStreamEle > 0
				outStreamEle.a[0][i * P0 + ii] = input[i * P0 + ii + loopTime * L0];	
			}
		}
		outStream.write(outStreamEle);
	}
	
}

void matmul1(hls::stream<MM1in> &inStream, VecL1* weight1, hls::stream<MM1out> &outStream){
	
	float tmp[L1 / P1][P1];
	#pragma HLS ARRAY_PARTITION variable=tmp dim=2 complete
	
	MM1in inStreamEle;
	#pragma HLS aggregate variable=inStreamEle
	MM1out outStreamEle;
	#pragma HLS ARRAY_PARTITION variable=outStreamEle.a dim=2 cyclic factor=64   // P1=8
	//#pragma HLS aggregate variable=outStreamEle
	
	for(int loopTime = 0; loopTime < UMAX; loopTime++){
		inStreamEle = inStream.read();
	
		init:
		for(int i = 0; i < L1 / P1;i++){
		#pragma HLS PIPELINE
			for(int ii = 0; ii < P1; ii++)
			tmp[i][ii] = 0;
		}
	
		load_weight:
		for (int k = 0; k < L0; k++){  
	
			VecL1 weightVec = weight1[k];
			#pragma HLS aggregate variable=weightVec
	
			block_loop:
			for (int i = 0; i < L1 / P1; i++){
			#pragma HLS PIPELINE
			#pragma HLS dependence variable=tmp inter false
	
				ele_loop:
				for (int ii = 0; ii < P1; ii++){
				#pragma HLS UNROLL
					tmp[i][ii] += inStreamEle.a[0][k] * weightVec.a[i * P1 + ii];
				}
			}
		}
	
		write_out:
		for(int i = 0; i < L1 / P1; i++){
		#pragma HLS PIPELINE
			for(int ii = 0; ii < P1; ii++){
				outStreamEle.a[0][i * P1 + ii] = tmp[i][ii];
			}
		}
		outStream.write(outStreamEle);
	}
}

void act1(hls::stream<MM1out> &inStream, const float* bias1, hls::stream<MM2in> &outStream){
	
	MM1out inStreamEle; 
	//#pragma HLS aggregate variable=inStreamEle
	#pragma HLS ARRAY_PARTITION variable=inStreamEle.a dim=2 cyclic factor=64	// P1, less resources
	MM2in outStreamEle;
	#pragma HLS ARRAY_PARTITION variable=outStreamEle.a dim=2 cyclic factor=64   // P1, directly compact
	// calculate the act one vector by one, reason: stream input
	
	outer_loop:
	for(int k = 0; k < UMAX; k++){	
		inStreamEle = inStream.read();
		
		inner_loop:
		for(int i = 0; i < L1 / P1; i++){
			#pragma HLS PIPELINE
				for(int ii = 0; ii < P1; ii++){	// special
					float tmp = inStreamEle.a[0][i * P1 + ii] + bias1[i * P1 + ii];
					outStreamEle.a[k][i * P1 + ii] = (tmp > 0 ? tmp : 0);
				}
			}
	}
	
	// must write out in one pack, thus mm2 could deal with it in parallel
	outStream.write(outStreamEle);
}

void matmul2(hls::stream<MM2in> &inStream, VecL2* weight2, hls::stream<MM2out> &outStream){
	
	float tmp[L2 * U2 / P2][P2];   
	#pragma HLS ARRAY_PARTITION variable=tmp dim=2 complete
	
	MM2in inStreamEle = inStream.read();
	// #pragma HLS aggregate variable=inStreamEle
	
	MM2out outStreamEle;	// always complete
	#pragma HLS ARRAY_PARTITION variable=outStreamEle.a dim=0 complete
	// can be improved: when unroll factor \neq batchsize
	
	init:
	for(int i = 0; i < L2 * U2 / P2; i++){
	#pragma HLS PIPELINE
		for(int ii = 0; ii < P2; ii++){
		#pragma HLS UNROLL
			tmp[i][ii] = 0;}
	}
	
	load_weight:
	for (int k = 0; k < L1; k++){
		VecL2 weightVec = weight2[k];
		//#pragma HLS aggregate variable=weightVec
		//#pragma HLS PIPELINE
		//#pragma HLS dependence variable=tmp inter false
		block_loop:	// if block_loop is bottleneck, then the outer loop always = 0
		for (int i = 0; i < L2 * U2 / P2; i++){
		#pragma HLS PIPELINE
		#pragma HLS dependence variable=tmp inter false
	
			ele_loop:   // completely unrolled
			for (int ii = 0; ii < P2; ii++){
				tmp[i][ii] += inStreamEle.a[ii / L2][k] * weightVec.a[ii % L2];
			}
		}
	}
	
	write_out: 
	//for(int i = 0; i < L2 * U2 / P2; i++){
	//#pragma HLS PIPELINE
		for(int ii = 0; ii < P2; ii++){
		#pragma HLS UNROLL
			outStreamEle.a[(ii) / L2][(ii) % L2] = tmp[0][ii];    // tmp[i][ii]->i*P+ii
		}
	//}
	outStream.write(outStreamEle);
	// batch_indx = (i * P2 + ii) / L2
	// dimension = (i * P2 + ii) % L2
}

void act2(hls::stream<MM2out> &inStream, const float* bias2, hls::stream<Blockout> &outStream){
	
	MM2out inStreamEle = inStream.read();
	#pragma HLS ARRAY_PARTITION variable=inStreamEle.a dim=0 complete
	// #pragma HLS aggregate variable=inStreamEle
	Blockout outStreamEle;
	#pragma HLS ARRAY_PARTITION variable=outStreamEle.a dim=0 complete
	
	for(int i = 0; i < L2 * U2 / P2; i++){	// no loop at all
	#pragma HLS PIPELINE
		for(int ii = 0; ii < P2; ii++){
			int batch_indx = (i * P2 + ii) / L2;
			int dimension = (i * P2 + ii) % L2;
			float tmp = inStreamEle.a[batch_indx][dimension] + bias2[dimension];
			outStreamEle.a[batch_indx][dimension] = (tmp > 0 ? tmp : 0);
		}
	}
	outStream.write(outStreamEle);
}

void storeDDR(hls::stream<Blockout> &inStream, float* output){
	
	Blockout inStreamEle = inStream.read();
	#pragma HLS aggregate variable=inStreamEle
	
	store_loop:
	for(int i = 0; i < L2 * U2; i++){
	#pragma HLS UNROLL //factor = x
		int batch_indx = i / L2;
		int dimension = i % L2;
		output[i] = inStreamEle.a[batch_indx][dimension];
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
	
	hls::stream<MM1in> pipeIn;
	hls::stream<MM1out> pipeL1;
	hls::stream<MM2in> pipeIn2;
	hls::stream<MM2out> pipeL2;
	hls::stream<Blockout> pipeOut;
	
	for(int i = 0; i < iteration_time; i++){
	#pragma HLS DATAFLOW
	cout << "top Index #" << i << endl;
		loadIn(&input[i * L0 * UMAX], pipeIn);
		matmul1(pipeIn, weight1, pipeL1);
		act1(pipeL1, bias1, pipeIn2);
		matmul2(pipeIn2, weight2, pipeL2);
		act2(pipeL2, bias2, pipeOut);
		storeDDR(pipeOut, &output[i * L2 * UMAX]);
	}
	
}

}
