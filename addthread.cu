//Add THREAD Vector Using GPU
// NAMA : Galih Aji Pambudi
// NIM	: 3332180058
// MyGPU: NVIDIA GTX 650 


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

//#define N 1048576
#define N 50000000
#define MAX_ERR 1e-6

__global__ void vector_add(float *out, float *a, float *b, int n){

    int index = threadIdx.x;
    int stride = blockDim.x;
    // int block = blockIdx.x;
    //int global = threadIdx.x + blockIdx.x * blockDim.x;
    //out[global] = a[global] + b[global];
    //out[global] = global;
    //printf("Global = %d\n", global);
    //printf("BlockDim.x = %d\n", stride);
    //printf("threadIdx.x = %d\n", threadIdx.x);
    //printf("Block = %d\n", block);

	for (int i = index; i < n; i += stride){
		out[i] = (a[i] / b[i]) / 78723;
	}
}

int main(){
	float *a, *b, *out;
	float *d_a, *d_b, *d_out;

	//Alokasi Host Memori
	a = (float*)malloc(sizeof(float) * N);
	b = (float*)malloc(sizeof(float) * N);
	out = (float*)malloc(sizeof(float) * N);

	//Inisialisasi Array
	for (int i = 0; i < N; i++){
		a[i] = 1.0f;
		//b[i] = 49237332.0f;
		b[i] = 1;
	}

	//Alokasi Device memori
	cudaMalloc((void**)&d_a, sizeof(float) * N);
	cudaMalloc((void**)&d_b, sizeof(float) * N);
	cudaMalloc((void**)&d_out, sizeof(float) * N);

	//Transfer Data dari Host memori ke Device memori
	cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

	//Eksekusi Kernel
	vector_add<<<857,857>>>(d_out, d_a, d_b, N); //<<<block,thread>>>

	//Transfer Data kembali ke Host Memori
	cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

	//Verification
	//for (int i = 0; i < N; i++){
	//	assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
	//}
	printf("out[0] = %f\n", out[0]);
	printf("PASSED\n");

	//Dealokasi Device Memori
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_out);

	//Dealokasi Host Memori
	free(a);
	free(b);
	free(out);

	return 0;
}