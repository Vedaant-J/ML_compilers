//
// Created by damitha on 1/29/25.
//
#include "utils.h"
#include <cuda_runtime.h>

__global__ void gemm_gpu_kernel(float* A, float* B, float *C, int M, int N, int K) {
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			C[i * N + j] = 0;
			for (int k = 0; k < K; k++) {
				C[i * N + j]  += A[i * K + k]  * B[j * K + k];
			}
		}
	}
}

void gemm_gpu(float* A, float* B, float* C, int M, int N, int K)
{
	// Init block and grid size
	dim3 blockSize(1, 1);
	dim3 gridSize(1, 1);
	gemm_gpu_kernel<<<gridSize, blockSize>>>(A, B, C, M, N, K);
}

// The scafolding for optimized GEMM implementations
__global__ void gemm_gpu_opt_kernel(float* A, float* B, float *C, int M, int N, int K) {

}


int main(int argc, char* argv[]) {
	if (argc < 3) {
		std::cout << "Usage: mp1 <M> <N> <K>" << std::endl;
		return 1;
	}

	int M = atoi(argv[1]);
	int N = atoi(argv[2]);
	int K = atoi(argv[3]);

	// int runs = atoi(argv[3]);

	float* A = new float[M * K]();
	float* B = new float[K * N]();
	float* C = new float[M * N]();

	fillRandom(A, M * K);
	fillRandom(B, K * N);

	/// GPU Implementation
	float *d_A, *d_B, *d_C;
	// Device memory allocation
	cudaMalloc(&d_A, M * K * sizeof(float));
	cudaMalloc(&d_B, K * N * sizeof(float));
	cudaMalloc(&d_C, M * N * sizeof(float));
	// Copy host memory to device
	cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);
	// For timing
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Discard the first 5 runs
	for (int i = 0; i < 5; i++)
	{
		gemm_gpu(A, B, C, M, N, K);
	}
	cudaEventRecord(start);
	for (int i = 0; i < 100; i++)
	{
		gemm_gpu(A, B, C, M, N, K);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	// Copy output back to host
	cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
	std::cout << "Time taken for GEMM (GPU): " << milliseconds << "ms" << std::endl;

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cudaFreeHost(A);
	cudaFreeHost(B);
	cudaFreeHost(C);

	delete[] A;
	delete[] B;
	delete[] C;

	return 0;
}