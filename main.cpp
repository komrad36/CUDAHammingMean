/*******************************************************************
*   main.cpp
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
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <chrono>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

using namespace std::chrono;

int main() {
	// ------------- Configuration ------------
	constexpr int warmups = 100;
	constexpr int runs = 300;
	constexpr uint64_t qsize = 10000;
	constexpr uint64_t tsize = 10000;
	// --------------------------------


	// ------------- Generation of Random Data ------------
	// obviously, this is not representative of real data;
	// it doesn't matter for brute-force matching
	// but the MIH methods will be much faster
	// on real data
	void *qvecs = malloc(64 * qsize), *tvecs = malloc(64 * tsize);
	srand(36);
	for (uint64_t i = 0; i < 64 * qsize; ++i) reinterpret_cast<uint8_t*>(qvecs)[i] = static_cast<uint8_t>(rand());
	for (uint64_t i = 0; i < 64 * tsize; ++i) reinterpret_cast<uint8_t*>(tvecs)[i] = static_cast<uint8_t>(rand());
	// --------------------------------


	// setting cache and shared modes
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

	// allocating and transferring query vecs and binding to texture object
	void* d_qvecs;
	cudaMalloc(&d_qvecs, 64 * qsize);
	cudaMemcpy(d_qvecs, qvecs, 64 * qsize, cudaMemcpyHostToDevice);
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeLinear;
	resDesc.res.linear.devPtr = d_qvecs;
	resDesc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
	resDesc.res.linear.desc.x = 32;
	resDesc.res.linear.desc.y = 32;
	resDesc.res.linear.sizeInBytes = 64 * qsize;
	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeBorder;
	texDesc.addressMode[1] = cudaAddressModeBorder;
	texDesc.filterMode = cudaFilterModePoint;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 0;
	cudaTextureObject_t tex_q = 0;
	cudaCreateTextureObject(&tex_q, &resDesc, &texDesc, nullptr);

	// allocating and transferring query vecs as simple global
	void* d_tvecs;
	cudaMalloc(&d_tvecs, 64 * tsize);
	cudaMemcpy(d_tvecs, tvecs, 64 * tsize, cudaMemcpyHostToDevice);

	// allocating space for sums (results)
	uint64_t* d_sums;
	cudaMalloc(&d_sums, sizeof(uint64_t) * qsize);

	std::cout << std::endl << "Warming up..." << std::endl;
	for (int i = 0; i < warmups; ++i) CUDAK2NN(d_tvecs, tsize, tex_q, qsize, d_sums);
	std::cout << "Testing..." << std::endl;
	high_resolution_clock::time_point start = high_resolution_clock::now();
	for (int i = 0; i < runs; ++i) CUDAK2NN(d_tvecs, tsize, tex_q, qsize, d_sums);
	high_resolution_clock::time_point end = high_resolution_clock::now();
	// --------------------------------


	// transferring sums back to host
	uint64_t* h_sums = reinterpret_cast<uint64_t*>(malloc(sizeof(uint64_t) * qsize));
	cudaMemcpy(h_sums, d_sums, sizeof(uint64_t) * qsize, cudaMemcpyDeviceToHost);
	cudaDeviceReset();

	std::cout << "CUDA reports " << cudaGetErrorString(cudaGetLastError()) << std::endl;

	// Careful with sizes. int will overflow if you accumulate-0 instead of accumulate-0ULL.
	const double total = static_cast<double>(std::accumulate(h_sums, h_sums + qsize, 0ULL)) / static_cast<double>(tsize * qsize);

	const double sec = static_cast<double>(duration_cast<nanoseconds>(end - start).count()) * 1e-9 / static_cast<double>(runs);
	std::cout.precision(15);
	std::cout << "CUDAK2NN found the mean to be " << total << ", in ";
	std::cout.precision(-1);
	std::cout << sec * 1e3 << " ms" << std::endl;
	std::cout << "Throughput: " << static_cast<double>(qsize)*static_cast<double>(tsize) / sec * 1e-9 << " billion comparisons/second." << std::endl << std::endl;
}
