#include "../include/utils.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define NUM_RUNS 10
#define BLOCK_SIZE 32

#define CUDA_CHECK(func)                                                     	   \
	do {                                                                           \
		cudaError_t status = (func);                                               \
		if (status != cudaSuccess) {                                               \
			printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__,   \
				cudaGetErrorString(status), status);                               \
			exit(EXIT_FAILURE);                                                    \
		}                                                                          \
	} while (0)

#define CHECK(name) \
	float *d_Aref_ ## name, *d_Bref_ ## name, *d_Cref_ ## name; \
	std::cerr << "checking " << #name << std::endl; \
	CUDA_CHECK(cudaMalloc(&d_Aref_ ## name, Ref::M * Ref::K * sizeof(float))); \
	CUDA_CHECK(cudaMalloc(&d_Bref_ ## name, Ref::K * Ref::N * sizeof(float))); \
	CUDA_CHECK(cudaMalloc(&d_Cref_ ## name, Ref::M * Ref::N * sizeof(float))); \
	CUDA_CHECK(cudaMemcpy(d_Aref_ ## name, ref.A, Ref::M * Ref::K * sizeof(float), cudaMemcpyHostToDevice)); \
	CUDA_CHECK(cudaMemcpy(d_Bref_ ## name, ref.B, Ref::K * Ref::N * sizeof(float), cudaMemcpyHostToDevice)); \
	float* d_Cref_INI_ ## name = new float[M * N](); \
	for (int i = 0; i < Ref::M; i++) { \
		for (int j = 0; j < Ref::N; j++) { \
			d_Cref_INI_ ## name[i * Ref::N + j] = 0; \
		} \
	} \
	CUDA_CHECK(cudaMemcpy(d_Cref_ ## name, d_Cref_INI_ ## name, Ref::M * Ref::N * sizeof(float), cudaMemcpyHostToDevice)); \
	name(d_Aref_ ## name, d_Bref_ ## name, d_Cref_ ## name, Ref::M, Ref::N, Ref::K); \
	cudaError_t err_c_ ## name = cudaGetLastError(); \
	if (err_c_ ## name != cudaSuccess) { \
		std::cerr << "CUDA Error: " << cudaGetErrorString(err_c_ ## name) << std::endl; \
	} \
	CUDA_CHECK(cudaMemcpy(refC, d_Cref_ ## name, Ref::M * Ref::N * sizeof(float), cudaMemcpyDeviceToHost)); \
	if (!ref.checkRef(refC)){ \
		std::cerr << "check ref failed!" << std::endl; \
	};

#define TIME(name) \
	float *d_A_ ## name, *d_B_ ## name, *d_C_ ## name; \
	CUDA_CHECK(cudaMalloc(&d_A_ ## name, M * K * sizeof(float))); \
	CUDA_CHECK(cudaMalloc(&d_B_ ## name, K * N * sizeof(float))); \
	CUDA_CHECK(cudaMalloc(&d_C_ ## name, M * N * sizeof(float))); \
	CUDA_CHECK(cudaMemcpy(d_A_ ## name, A, M * K * sizeof(float), cudaMemcpyHostToDevice)); \
	CUDA_CHECK(cudaMemcpy(d_B_ ## name, B, K * N * sizeof(float), cudaMemcpyHostToDevice)); \
	cudaEvent_t start_ ## name, end_ ## name; \
	cudaEventCreate(&start_ ## name); \
	cudaEventCreate(&end_ ## name); \
	float* d_C_INI_ ## name = new float[M * N](); \
	for (int i = 0; i < M; i++) { \
		for (int j = 0; j < N; j++) { \
			d_C_INI_ ## name[i * N + j] = 0; \
		} \
	} \
	for (int i = 0; i < 2; i++) \
	{ \
		CUDA_CHECK(cudaMemcpy(d_C_ ## name, d_C_INI_ ## name, M * N * sizeof(float), cudaMemcpyHostToDevice)); \
		name(d_A_ ## name, d_B_ ## name, d_C_ ## name, M, N, K); \
	} \
	cudaError_t err_t_ ## name = cudaGetLastError(); \
	if (err_t_ ## name != cudaSuccess) { \
		std::cerr << "CUDA Error: " << cudaGetErrorString(err_t_ ## name) << std::endl; \
	} \
	float milliseconds_ ## name = 0; \
	for (int i = 0; i < NUM_RUNS; i++) \
	{ \
		CUDA_CHECK(cudaMemcpy(d_C_ ## name, d_C_INI_ ## name, M * N * sizeof(float), cudaMemcpyHostToDevice)); \
		cudaDeviceSynchronize(); \
		cudaEventRecord(start_ ## name); \
		name(d_A_ ## name, d_B_ ## name, d_C_ ## name, M, N, K); \
		cudaEventRecord(end_ ## name); \
		cudaEventSynchronize(end_ ## name); \
		float milliseconds_ ## i = 0; \
		cudaEventElapsedTime(&milliseconds_ ## i, start_ ## name, end_ ## name); \
		milliseconds_ ## name += milliseconds_ ## i; \
	} \
	cudaMemcpy(C, d_C_ ## name, M * N * sizeof(float), cudaMemcpyDeviceToHost); \
	std::cout << "Time taken for GEMM (GPU, " << #name <<"): " << milliseconds_ ## name / (float)NUM_RUNS << "ms" << std::endl; \
	cudaFree(d_A_ ## name); \
	cudaFree(d_B_ ## name); \
	cudaFree(d_C_ ## name);

__global__ void gemm_gpu_o0_kernel(float* A, float* B, float *C, int M, int N, int K) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				for (int k = 0; k < K; k++) {
					C[i * N + j]  += A[i * K + k]  * B[k * N + j];
				}
			}
		}
    }
}

void gemm_gpu_o0(float* A, float* B, float* C, int M, int N, int K)
{
	// Init block and grid size
	dim3 blockSize(1);
	dim3 gridSize(1);
	gemm_gpu_o0_kernel<<<gridSize, blockSize>>>(A, B, C, M, N, K);
}

// The scafolding for optimized GEMM implementations
__global__ void gemm_gpu_o1_kernel(float* A, float* B, float *C, int M, int N, int K) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
	
}
void gemm_gpu_o1(float* A, float* B, float* C, int M, int N, int K)
{

	dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + 15) / 16, (M + 15) / 16);

    // Launch the kernel with the full grid.
    gemm_gpu_o1_kernel<<<numBlocks, threadsPerBlock>>>(A, B, C, M, N, K);
}

__global__ void gemm_gpu_o2_kernel(float* A, float* B, float *C, int M, int N, int K) {
	__shared__ float A_tile[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float B_tile[BLOCK_SIZE][BLOCK_SIZE];

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	float sum = 0.0f; // accumluate and assign beccause memset takes long

	for (int k0=0; k0 < K; k0 += BLOCK_SIZE) {
		if (row < M && (k0+threadIdx.x) < K)
			A_tile[threadIdx.y][threadIdx.x] = A[row * K + k0 + threadIdx.x];
		else
			A_tile[threadIdx.y][threadIdx.x] = 0.0f;
		if (col < N && (k0+threadIdx.y) < K)
			B_tile[threadIdx.y][threadIdx.x] = B[(k0 + threadIdx.y) * N + col];
		else
			B_tile[threadIdx.y][threadIdx.x] = 0.0f;

		// the 32x32 threads should have loaded the needed numbers in tiles by now I think
		__syncthreads();

		for (int k1 = 0; k1 < BLOCK_SIZE; k1++) {
			sum += A_tile[threadIdx.y][k1] * B_tile[k1][threadIdx.x];
		}
		__syncthreads();
	}
	if (row < M && col < N)
		C[row * N + col] = sum;

}
void gemm_gpu_o2(float* A, float* B, float* C, int M, int N, int K)
{
	// Init block and grid size
	dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
	dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

	gemm_gpu_o2_kernel<<<numBlocks, blockSize>>>(A, B, C, M, N, K);
}

__global__ void gemm_gpu_o3_kernel(float* A, float* B, float *C, int M, int N, int K) {
	__shared__ float A_tile[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float B_tile[BLOCK_SIZE][BLOCK_SIZE];

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	float sum = 0.0f; // accumluate and assign beccause memset takes long

	for (int k0=0; k0 < K; k0 += BLOCK_SIZE) {
		if (row < M && (k0+threadIdx.x) < K)
			A_tile[threadIdx.y][threadIdx.x] = A[row * K + k0 + threadIdx.x];
		else
			A_tile[threadIdx.y][threadIdx.x] = 0.0f;
		if (col < N && (k0+threadIdx.y) < K)
			B_tile[threadIdx.y][threadIdx.x] = B[(k0 + threadIdx.y) * N + col];
		else
			B_tile[threadIdx.y][threadIdx.x] = 0.0f;

		// the 32x32 threads should have loaded the needed numbers in tiles by now I think
		__syncthreads();

		for (int k1 = 0; k1 < BLOCK_SIZE; k1++) {
			sum += A_tile[threadIdx.y][k1] * B_tile[k1][threadIdx.x];
		}
		__syncthreads();
	}
	if (row < M && col < N)
		C[row * N + col] = sum;
}
void gemm_gpu_o3(float* A, float* B, float* C, int M, int N, int K)
{
	// Init block and grid size
	dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
	dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

	gemm_gpu_o2_kernel<<<numBlocks, blockSize>>>(A, B, C, M, N, K);
}

void gemm_cublas(float* d_A, float* d_B, float* d_C, int M, int N, int K) {
    // 1. Create a handle to the cuBLAS library context
    cublasHandle_t handle;
    cublasCreate(&handle);

    // 2. Set scalar multipliers for C = alpha * A*B + beta*C
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // 3. Call the cuBLAS SGEMM function
    // Note the trick for row-major matrices: we compute C = B * A
    // and pass the dimensions as (N, M, K) instead of (M, N, K).
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,     // Don't transpose A or B
                N, M, K,                     // Dimensions: M_B, N_A, K_A/B
                &alpha,                      // alpha
                d_B, N,                      // Pointer to B (as A), and its leading dimension
                d_A, K,                      // Pointer to A (as B), and its leading dimension
                &beta,                       // beta
                d_C, N                       // Pointer to C, and its leading dimension
               );

    // 4. Destroy the cuBLAS handle
    cublasDestroy(handle);
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
	auto ref = Ref();
	float* refC = new float[Ref::M * Ref::N]();
 	// CHECK(gemm_gpu_o0)
	// CHECK(gemm_gpu_o1)
	// CHECK(gemm_gpu_o2)
	CHECK(gemm_gpu_o3)
	CHECK(gemm_cublas)

	// Actual run
 	// TIME(gemm_gpu_o0)
	// TIME(gemm_gpu_o1)
	// TIME(gemm_gpu_o2)
	TIME(gemm_gpu_o3)
	TIME(gemm_cublas)

	cudaFreeHost(A);
	cudaFreeHost(B);
	cudaFreeHost(C);

	delete[] A;
	delete[] B;
	delete[] C;

	return 0;
}