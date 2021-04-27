//Add Vector Using CPU


#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

#define N 10000000

void vector_add_cpu(float *out, float *a, float *b, int n) {
	for (int i = 0; i < n; i++){
		out[i] = a[i] + b[i];
	}
	printf("out[0] = %f\n", out[0]);
}

int main(){
	float *a, *b, *out;

	//Alokasi Memori
	a = (float*)malloc(sizeof(float) * N);
	b = (float*)malloc(sizeof(float) * N);
	out = (float*)malloc(sizeof(float) * N);

	//Inisialisasi Array
	for (int i = 0; i < N; i++){
		a[i] = 1.0f; 
		b[i] = 2.0f;
	}

	//Main Function
	vector_add_cpu(out, a, b, N);
	return 0;
}
