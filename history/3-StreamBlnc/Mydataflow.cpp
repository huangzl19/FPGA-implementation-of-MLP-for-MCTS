#include "./Mydataflow.h"
#include "./params.h"

extern "C"{

void loadIn(float* input, hls::stream<Block1in> &outStream){
	Block1in tmp;
	// #pragma HLS aggregate variable=tmp // no influence here, since the L0=2 is too small
	
    outer_loop:	// to do 4 batch_size in a roll
    for(int loopTime = 0; loopTime < UNIT_SIZE; loopTime++){
        loop_load:
        for (int i = 0; i < L0 / P0; i++){
	    #pragma HLS PIPELINE
	    	for(int ii = 0; ii < P0; ii++){
	    		tmp.a[i * P0 + ii] = input[i * P0 + ii + loopTime * L0];
	    		//cout << loopTime << endl;
	    		//cout << "loadIn #" << i * P0 + ii << " = " << tmp.a[i * P0 + ii] << endl;
	    	}
	    }
	    outStream.write(tmp);
    }
	
}

// must use hls::stream
void matmul1(hls::stream<Block1in> &inStream, VecL1* weight1, hls::stream<VecL1> &outStream){
//#pragma HLS aggregate variable=inStream
//#pragma HLS aggregate variable=weight1
//#pragma HLS aggregate variable=outStream

	float tmp[L1 / P1][P1];
	#pragma HLS ARRAY_PARTITION variable=tmp dim=2 complete

	Block1in inStreamEle;
	#pragma HLS aggregate variable=inStreamEle

	for(int loopTime = 0; loopTime < UNIT_SIZE; loopTime++){
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
	    				tmp[i][ii] += inStreamEle.a[k] * weightVec.a[i * P1 + ii];
				}
	    		}
		}
	}
    
        VecL1 outStreamEle;
	//#pragma HLS aggregate variable=outStreamEle
	#pragma HLS ARRAY_PARTITION variable=outStreamEle.a cyclic factor=8   // P1=8
	
	write_out:
	for(int i = 0; i < L1 / P1; i++){
	#pragma HLS PIPELINE
		for(int ii = 0; ii < P1; ii++){
			outStreamEle.a[i * P1 + ii] = tmp[i][ii];
		}
	}
	outStream.write(outStreamEle);
}

void act1(hls::stream<VecL1> &inStream, const float* bias1, hls::stream<Block2in> &outStream){
//#pragma HLS aggregate variable=bias1
//#pragma HLS ARRAY_PARTITION variable=bias1 cyclic factor=8

	VecL1 inStreamEle; 
	//#pragma HLS aggregate variable=inStreamEle
	#pragma HLS ARRAY_PARTITION variable=inStreamEle cyclic factor=8
	Block2in outStreamEle;
	#pragma HLS ARRAY_PARTITION variable=outStreamEle.a dim=2 cyclic factor=8   // P1

	outer_loop:
	for(int k = 0; k < UNIT_SIZE; k++){	
		inStreamEle = inStream.read();
		
		inner_loop:
		for(int i = 0; i < L1 / P1; i++){
	    	#pragma HLS PIPELINE
	    		for(int ii = 0; ii < P1; ii++){
	    			float tmp = inStreamEle.a[i * P1 + ii] + bias1[i * P1 + ii];
	    			outStreamEle.a[k][i * P1 + ii] = (tmp > 0 ? tmp : 0);
	    		}
        	}
	}
	
	// must write out in one pack, thus mm2 could deal with it in parallel
	outStream.write(outStreamEle);
}

void matmul2(hls::stream<Block2in> &inStream, VecL2* weight2, hls::stream<Block3in> &outStream){
// // #pragma HLS aggregate variable=weight2
// another idea: eliminate tmp, directly write into outStreamEle
	float tmp[L2 * UNIT_SIZE / P2][P2];     // for every layer, should hold one UNIT_SIZE: mm2->4, mm1->1
	#pragma HLS ARRAY_PARTITION variable=tmp dim=2 complete
	Block2in inStreamEle = inStream.read();
	// #pragma HLS aggregate variable=inStreamEle.a dim=1 complete

	init:
	for(int i = 0; i < L2 * UNIT_SIZE / P2; i++){
	#pragma HLS PIPELINE
		for(int ii = 0; ii < P2; ii++)
			tmp[i][ii] = 0;
	}

	load_weight:
	for (int k = 0; k < L1; k++){
		VecL2 weightVec = weight2[k];
		// #pragma HLS aggregate variable=weightVec

		block_loop:
		for (int i = 0; i < L2 * UNIT_SIZE / P2; i++){
		#pragma HLS PIPELINE
		#pragma HLS dependence variable=tmp inter false

			ele_loop:   // completely unrolled
			for (int ii = 0; ii < P2; ii++){
				tmp[i][ii] += inStreamEle.a[i * P2 + ii][k] * weightVec.a[(i * P2 + ii) % L2];
                //tmp[ii] += tmpA.a[ii][k] * weightVec.a[0]
				//cout << "Batch #" << i*P2+ii << " in " << k << "th accumulation: input = " << inStreamEle.a[i * P2 + ii][k] << ", weight = " << weightVec.a[(i * P2 + ii) % L2] << ", tmp now = "<< tmp[i][ii] << endl;
			}
		}
		//cout << endl;
	}

	Block3in outStreamEle;
	#pragma HLS ARRAY_PARTITION variable=outStreamEle.a dim=1 complete
    // can be improved: when unroll factor \neq batchsize
	
	write_out: 
	for(int i = 0; i < L2 * UNIT_SIZE / P2; i++){
	#pragma HLS PIPELINE
		for(int ii = 0; ii < P2; ii++){
			outStreamEle.a[i * P2 + ii][(i * P2 + ii) % L2] = tmp[i][ii];    // tmp[i][ii]->i*P+ii
			//cout << "mm2 #(" << i * P2 + ii << ", " << (i * P2 + ii) % L2 << ") = " << tmp[i][ii] << endl;
		}
	}
	outStream.write(outStreamEle);
    // batch_indx = (i * P2 + ii) / L2
    // dimension = (i * P2 + ii) % L2
}

void act2(hls::stream<Block3in> &inStream, const float* bias2, hls::stream<Block3in> &outStream){
// #pragma HLS aggregate variable=bias2
	
	Block3in inStreamEle = inStream.read();
	Block3in outStreamEle;
    // difference between aggregate and array_partition
	// #pragma HLS aggregate variable=inStreamEle
	// #pragma HLS aggregate variable=outStreamEle
	
	for(int i = 0; i < L2 * UNIT_SIZE/ P2; i++){
	#pragma HLS PIPELINE
		for(int ii = 0; ii < P2; ii++){
            int batch_indx = (i * P2 + ii) / L2;
            int dimension = (i * P2 + ii) % L2;
			float tmp = inStreamEle.a[batch_indx][dimension] + bias2[dimension];
			outStreamEle.a[batch_indx][dimension] = (tmp > 0 ? tmp : 0);
			//cout << "mm2 #(" << batch_indx << ", " << dimension << ") = " << outStreamEle.a[batch_indx][dimension] << endl;
		}
	}
	outStream.write(outStreamEle);
}

void storeDDR(hls::stream<Block3in> &inStream, float* output){
	
	Block3in inStreamEle = inStream.read();
	// #pragma HLS aggregate variable=inStreamEle

	store_loop:
	for(int i = 0; i < L2 * UNIT_SIZE; i++){
	#pragma HLS UNROLL //factor = x
        int batch_indx = i / L2;
        int dimension = i % L2;
		output[i] = inStreamEle.a[batch_indx][dimension];
		//cout << "output #(" << batch_indx << ", " << dimension << ") = " << output[i] << endl;
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

    hls::stream<Block1in> pipeIn;
    hls::stream<VecL1> pipeL1;
    hls::stream<Block2in> pipeIn2;
    hls::stream<Block3in> pipeIn3[2];
	
	for(int i = 0; i < UNIT_NUM; i++){
		cout << "top Index #" << i << endl;
	#pragma HLS DATAFLOW
		loadIn(&input[i * L0 * UNIT_SIZE], pipeIn);
		matmul1(pipeIn, weight1, pipeL1);
		act1(pipeL1, bias1, pipeIn2);
		matmul2(pipeIn2, weight2, pipeIn3[0]);
		act2(pipeIn3[0], bias2, pipeIn3[1]);
		storeDDR(pipeIn3[1], &output[i * L2 * UNIT_SIZE]);
	}
	
}
}
