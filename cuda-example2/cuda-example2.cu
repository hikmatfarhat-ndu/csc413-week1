#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


__global__ void kernel(int* a, int* b, int* c) {
	int idx = threadIdx.x;
	c[idx] = a[idx] + b[idx];
}

int main() {
	int N = 1024;
	int* a, * b, * c;
	int* da, * db, * dc;
	a = (int*)malloc(N * sizeof(int));
	b = (int*)malloc(N * sizeof(int));
	c = (int*)malloc(N * sizeof(int));

	cudaMalloc(&da, N * sizeof(int));
	cudaMalloc(&db, N * sizeof(int));
	cudaMalloc(&dc, N * sizeof(int));
	for (int i = 0; i < N; ++i) {
		a[i] = i;
		b[i] = 2 * i;
	}
	cudaMemcpy(da, a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(db, b, N * sizeof(int), cudaMemcpyHostToDevice);

	kernel << <1, N >> > (da, db, dc);
	cudaMemcpy(c, dc, N * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < 10; ++i)
		std::cout << c[i] << ' ';
	std::cout << std::endl;
	free(a);
	free(b);
	free(c);
	cudaFree(db);
	cudaFree(dc);
	cudaFree(da);

}