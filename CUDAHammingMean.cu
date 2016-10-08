/*******************************************************************
*   CUDAHammingMean.cu
*   CUDAHammingMean
*
*	Author: Kareem Omar
*	kareem.omar@uah.edu
*	https://github.com/komrad36
*
*	Last updated Oct 8, 2016
*******************************************************************/
//
// Fastest GPU implementation of a brute-force
// Hamming-weight matrix for 512-bit binary descriptors.
//
// Yes, that means the DIFFERENCE in popcounts is used
// for thresholding, NOT the ratio. This is the CORRECT
// approach for binary descriptors.
//
// This laboriously crafted kernel is EXTREMELY fast.
// 43 BILLION comparisons per second on a stock GTX1080,
// enough to match nearly 38,000 descriptors per frame at 30 fps (!)
//
// A key insight responsible for much of the performance of
// this insanely fast CUDA kernel is due to
// Christopher Parker (https://github.com/csp256), to whom
// I am extremely grateful.
//
// CUDA CC 3.0 or higher is required.
//
// All functionality is contained in the files CUDAK2NN.h
// and CUDAK2NN.cu. 'main.cpp' is simply a sample test harness
// with example usage and performance testing.
//

#include "CUDAHammingMean.h"

__global__ void
#ifndef __INTELLISENSE__
__launch_bounds__(256, 0)
#endif
CUDAK2NN_kernel(const cudaTextureObject_t tex_q, const int num_q, const uint64_t* const __restrict__ g_training, const int num_t, uint64_t* const __restrict__ g_sums) {
	uint64_t q[8];
#pragma unroll
	for (int i = 0, offset = ((threadIdx.x & 24) << 3) + (threadIdx.x & 7) + (blockIdx.x << 11) + (threadIdx.y << 8); i < 8; ++i, offset += 8) {
		const uint2 buf = tex1Dfetch<uint2>(tex_q, offset);
		asm("mov.b64 %0, {%1,%2};" : "=l"(q[i]) : "r"(buf.x), "r"(buf.y)); // some assembly required
	}
	uint64_t total = 0ULL;
#pragma unroll 6
	for (int t = 0; t < num_t; ++t) {
		const uint64_t train = g_training[(t << 3) + (threadIdx.x & 7)];
		uint32_t dist[4];
		dist[0] = __byte_perm(__popcll(q[0] ^ train), __popcll(q[4] ^ train), 0x5410U);
		dist[0] += __shfl_xor(dist[0], 1);
		dist[1] = __byte_perm(__popcll(q[1] ^ train), __popcll(q[5] ^ train), 0x5410U);
		dist[1] += __shfl_xor(dist[1], 1);
		dist[2] = __byte_perm(__popcll(q[2] ^ train), __popcll(q[6] ^ train), 0x5410U);
		dist[2] += __shfl_xor(dist[2], 1);
		dist[3] = __byte_perm(__popcll(q[3] ^ train), __popcll(q[7] ^ train), 0x5410U);
		dist[3] += __shfl_xor(dist[3], 1);
		if (threadIdx.x & 1) dist[0] = dist[1];
		if (threadIdx.x & 1) dist[2] = dist[3];
		dist[0] += __shfl_xor(dist[0], 2);
		dist[2] += __shfl_xor(dist[2], 2);
		if (threadIdx.x & 2) dist[0] = dist[2];
		dist[0] += __shfl_xor(dist[0], 4);
		total += __byte_perm(dist[0], 0U, threadIdx.x & 4 ? 0x5432U : 0x5410U);
	}
	const int idx = (blockIdx.x << 8) + (threadIdx.y << 5) + threadIdx.x;
	if (idx < num_q) g_sums[idx] = total;
}

void CUDAK2NN(const void* const __restrict d_t, const int num_t, const cudaTextureObject_t tex_q, const int num_q, uint64_t* const __restrict d_sums) {
	CUDAK2NN_kernel<<<((num_q - 1) >> 8) + 1, { 32, 8 }>>>(tex_q, num_q, reinterpret_cast<const uint64_t*>(d_t), num_t, d_sums);
	cudaDeviceSynchronize();
}
