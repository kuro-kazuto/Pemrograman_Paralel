//Add Vector Using GPU

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 10000000
#define MAX_ERR 1e-6

__global__ void vector_add(float *out, float *a, float *b, int n){
	for (int i = 0; i < n; i++){
		out[i] = a[i] + b[i];
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
		b[i] = 2.0f;
	}

	//Alokasi Device memori
	cudaMalloc((void**)&d_a, sizeof(float) * N);
	cudaMalloc((void**)&d_b, sizeof(float) * N);
	cudaMalloc((void**)&d_out, sizeof(float) * N);

	//Transfer Data dari Host memori ke Device memori
	cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

	//Eksekusi Kernel
	vector_add<<<1,1>>>(d_out, d_a, d_b, N);

	//Transfer Data kembali ke Host Memori
	cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

	//Verification
	for (int i = 0; i < N; i++){
		assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
	}
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