//
// Created by damitha on 1/29/25.
//
#include "utils.h"
#include <cuda_runtime.h>

#define CUDA_CHECK(func)                                                     	   \
	do {                                                                           \
		cudaError_t status = (func);                                               \
		if (status != cudaSuccess) {                                               \
			printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__,   \
				cudaGetErrorString(status), status);                               \
			exit(EXIT_FAILURE);                                                    \
		}                                                                          \
	} while (0)

__global__ void gemm_gpu_kernel(float* A, float* B, float *C, int M, int N, int K) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				C[i * N + j] = 0;
				for (int k = 0; k < K; k++) {
					C[i * N + j]  += A[i * K + k]  * B[k * N + j];
				}
			}
		}
    }
}

void gemm_gpu(float* A, float* B, float* C, int M, int N, int K)
{
	// Init block and grid size
	dim3 blockSize(1);
	dim3 gridSize(1);
	gemm_gpu_kernel<<<gridSize, blockSize>>>(A, B, C, M, N, K);
}

// The scafolding for optimized GEMM implementations
__global__ void gemm_gpu_opt_kernel(float* A, float* B, float *C, int M, int N, int K) {
}
void gemm_gpu_opt(float* A, float* B, float* C, int M, int N, int K)
{
	// Init block and grid size
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
    // Check if implementation is correct
	float *d_Aref, *d_Bref, *d_Cref;
	CUDA_CHECK(cudaMalloc(&d_Aref, Ref::M * Ref::K * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_Bref, Ref::K * Ref::N * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_Cref, Ref::M * Ref::N * sizeof(float)));
	auto ref = Ref();
	CUDA_CHECK(cudaMemcpy(d_Aref, ref.A, Ref::M * Ref::K * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_Bref, ref.B, Ref::K * Ref::N * sizeof(float), cudaMemcpyHostToDevice));
	gemm_gpu(d_Aref, d_Bref, d_Cref, Ref::M, Ref::N, Ref::K);
    // Print errors if there are any
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
	}
	float* refC = new float[Ref::M * Ref::N]();
	CUDA_CHECK(cudaMemcpy(refC, d_Cref, Ref::M * Ref::N * sizeof(float), cudaMemcpyDeviceToHost));
	if (!ref.checkRef(refC)){
		std::cerr << "check ref failed!" << std::endl;
	};
    // Actual run
	float *d_A, *d_B, *d_C;
	// Device memory allocation
	CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
	// Copy host memory to device
	CUDA_CHECK(cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice));
	// For timing
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Discard the first 5 runs
	for (int i = 0; i < 5; i++)
	{
		gemm_gpu(d_A, d_B, d_C, M, N, K);
	}
	cudaError_t err_base = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cerr << "CUDA Error: " << cudaGetErrorString(err_base) << std::endl;
	}
	cudaDeviceSynchronize();
	cudaEventRecord(start);
	for (int i = 0; i < 100; i++)
	{
		gemm_gpu(d_A, d_B, d_C, M, N, K);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	// Copy output back to host
	cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
	std::cout << "Time taken for GEMM (GPU, unoptimized): " << milliseconds << "ms" << std::endl;
    // Optimized implementation
	// For timing
	cudaEvent_t start_opt, stop_opt;
	cudaEventCreate(&start_opt);
	cudaEventCreate(&stop_opt);
	// Discard the first 5 runs
	for (int i = 0; i < 5; i++)
	{
		gemm_gpu_opt(A, B, C, M, N, K);
	}
	cudaError_t err_opt = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cerr << "CUDA Error: " << cudaGetErrorString(err_opt) << std::endl;
	}
	cudaDeviceSynchronize();
	cudaEventRecord(start);
	for (int i = 0; i < 100; i++)
	{
		gemm_gpu_opt(A, B, C, M, N, K);
	}
	cudaEventRecord(stop_opt);
	cudaEventSynchronize(stop_opt);
	float milliseconds_opt = 0;
	cudaEventElapsedTime(&milliseconds_opt, start_opt, stop_opt);
	// Copy output back to host
	cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
	std::cout << "Time taken for GEMM (GPU, optimized): " << milliseconds_opt << "ms" << std::endl;


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