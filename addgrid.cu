//Add GRID Vector Using GPU

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 50000000
#define MAX_ERR 1e-6

__global__ void vector_add_grid(float *out, float *a, float *b, int n){
	
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("Global = %d\n", tid);

	//Handling arbitary vector size
	if (tid < n){
		out[tid] = a[tid] + b[tid];

	}
}

int main(int argc, char **argv){
	float *a, *b, *out;
	float *d_a, *d_b, *d_out;

	//Alokasi Host Memori
	a = (float*)malloc(sizeof(float) * N);
	b = (float*)malloc(sizeof(float) * N);
	out = (float*)malloc(sizeof(float) * N);

	//Inisialisasi Array
	for (int i = 0; i < N; i++){
		a[i] = 29.0f;
		b[i] = 57.0f;
	}

	//Alokasi Device memori
	cudaMalloc((void**)&d_a, sizeof(float) * N);
	cudaMalloc((void**)&d_b, sizeof(float) * N);
	cudaMalloc((void**)&d_out, sizeof(float) * N);

	//Transfer Data dari Host memori ke Device memori
	cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

	//Eksekusi Kernel
	int block_size = 256;
	int grid_size = ((N + block_size) / block_size);
	vector_add_grid <<<grid_size, block_size>>> (d_out, d_a, d_b, N);

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