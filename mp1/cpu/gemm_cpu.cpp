#include <chrono>
#include "../include/utils.h"
#include <cmath>
#include <stdio.h>

#define NUM_RUNS 2

#define CHECK(name) \
  std::cout << "checking " << #name << std::endl;		\
  initialize(refC, Ref::M * Ref::N);				\
  name(ref.A, ref.B, refC, Ref::M, Ref::N, Ref::K);		\
  if (!ref.checkRef(refC)){					\
    std::cerr << #name << ": check ref failed!" << std::endl;	\
  };								
  
#define TIME(name) \
  for (int i = 0; i < 1; i++)						\
    {									\
      name(A, B, C, M, N, K);						\
    }									\
  std::chrono::duration<double, std::milli> time_##name(0);		\
  for (int i = 0; i < NUM_RUNS; i++)					\
    {									\
      initialize(C, M * N);						\
      auto start_time_ ## name = std::chrono::high_resolution_clock::now(); \
      name(A, B, C, M, N, K);						\
      auto end_time_ ## name = std::chrono::high_resolution_clock::now(); \
      time_ ## name += end_time_ ## name - start_time_ ## name;		\
    }									\
std::chrono::duration<double, std::milli> duration_ ## name = time_ ## name/float(NUM_RUNS); \
  std::cout << "Time taken for GEMM (CPU," << #name <<"): " << duration_ ## name.count() << "ms" << std::endl; 


// reference CPU implementation of the GEMM kernel
// note that this implementation is naive and will run for longer for larger
// graphs
void gemm_cpu_o0(float* A, float* B, float *C, int M, int N, int K) {
  for (int j = 0; j < N; j++) {
    for (int i = 0; i < M; i++) {
      for (int k = 0; k < K; k++) {
	C[i * N + j]  += A[i * K + k]  * B[k * N + j];
      }
    }
  }
}

// Your optimized implementations go here
// note that for o4 you don't have to change the code, but just the compiler flags. So, you can use o3's code for that part
void gemm_cpu_o1(float* A, float* B, float *C, int M, int N, int K) {
	for (int i = 0; i < M; i++) {
		for (int k = 0; k < K; k++) {
			for (int j = 0; j < N; j++) {
				C[i * N + j]  += A[i * K + k]  * B[k * N + j];
			}
		}
	}
}

void gemm_cpu_o2(float* A, float* B, float *C, int M, int N, int K) {
	// int L1D_CACHE_SIZE = 48000; // 48KiB
	// int FLOAT_SIZE = 4; // 4 bytes
	// int CACHE_SIZE_FLOAT = L1D_CACHE_SIZE / FLOAT_SIZE; // number of floats that can fit in cache
	// int BLOKE_SIZE = (int) std::sqrt(CACHE_SIZE_FLOAT / 3); // 3 matrices, so 3 blocks
	// BLOKE_SIZE = log2(BLOKE_SIZE);
	// BLOKE_SIZE = pow(2, BLOKE_SIZE); // make it power of 2
	int BLOKE_SIZE = 32; // coded to enable auto vectorization and also the above calculations take time

	for (int i0 = 0; i0 < M; i0 += BLOKE_SIZE) {
		for (int j0 = 0; j0 < N; j0 += BLOKE_SIZE) {
			for (int k0 = 0; k0 < K; k0 += BLOKE_SIZE) {
				for (int i = i0; i < std::min(i0 + BLOKE_SIZE, M); i++) {
					for (int k = k0; k < std::min(k0 + BLOKE_SIZE, K); k++) {
						for (int j = j0; j < std::min(j0 + BLOKE_SIZE, N); j++) {
							C[i * N + j] += A[i * K + k] * B[k * N + j];
						}
					}
				}
			}
		}
	}

}

void gemm_cpu_o3(float* __restrict__ A, float* __restrict__ B, float * __restrict__ C, int M, int N, int K) {
	// int L1D_CACHE_SIZE = 48000; // 48KiB
	// int FLOAT_SIZE = 4; // 4 bytes
	// int CACHE_SIZE_FLOAT = L1D_CACHE_SIZE / FLOAT_SIZE; // number of floats that can fit in cache
	// int BLOKE_SIZE = (int) std::sqrt(CACHE_SIZE_FLOAT / 3);
	// BLOKE_SIZE = log2(BLOKE_SIZE);
	// BLOKE_SIZE = pow(2, BLOKE_SIZE); // make it power of 2
	int BLOKE_SIZE = 32; // coded to enable auto vectorization

	#pragma omp parallel for shared(A, B, C) collapse(2)
	for (int i0 = 0; i0 < M; i0 += BLOKE_SIZE) {
		for (int j0 = 0; j0 < N; j0 += BLOKE_SIZE) {
			for (int k0 = 0; k0 < K; k0 += BLOKE_SIZE) {
				for (int i = i0; i < std::min(i0 + BLOKE_SIZE, M); i++) {
					for (int k = k0; k < std::min(k0 + BLOKE_SIZE, K); k++) {
						#pragma omp simd
						for (int j = j0; j < std::min(j0 + BLOKE_SIZE, N); j++) {
							C[i * N + j] += A[i * K + k] * B[k * N + j];
						}
					}
				}
			}
		}
	}
}


int main(int argc, char* argv[]) {
	if (argc < 3) {
	  std::cout << "Usage: mp1 <M> <N> <K>" << std::endl;
	  return 1;
	}

	int M = atoi(argv[1]);
	int N = atoi(argv[2]);
	int K = atoi(argv[3]);

	float* A = new float[M * K]();
	float* B = new float[K * N]();
	float* C = new float[M * N]();

	fillRandom(A, M * K);
	fillRandom(B, K * N);

	// Check if the kernel results are correct
	// note that even if the correctness check fails all optimized kernels will run.
	// We are not exiting the program at failure at this point.
	// It is a good idea to add more correctness checks to your code.
	// We may (at discretion) verify that your code is correct.
	float* refC = new float[Ref::M * Ref::N]();
	auto ref = Ref();
	// CHECK(gemm_cpu_o0)
	// CHECK(gemm_cpu_o1)
	// CHECK(gemm_cpu_o2)
	CHECK(gemm_cpu_o3)
	delete[] refC;
	
	// TIME(gemm_cpu_o0)
	// TIME(gemm_cpu_o1)
	// TIME(gemm_cpu_o2)
	TIME(gemm_cpu_o3)

	delete[] A;
	delete[] B;
	delete[] C;

	return 0;
}