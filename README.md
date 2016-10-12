Fastest GPU implementation of a brute-force
Hamming-weight matrix for 512-bit binary descriptors.

Yes, that means the DIFFERENCE in popcounts is used
for thresholding, NOT the ratio. This is the CORRECT
approach for binary descriptors.

A key insight responsible for much of the performance of
this insanely fast CUDA kernel is due to
Christopher Parker (https://github.com/csp256), to whom
I am extremely grateful.

CUDA CC 3.0 or higher is required.

All functionality is contained in the files CUDAHammingMean.h
and CUDAHammingMean.cu. 'main.cpp' is simply a sample test harness
with example usage and performance testing.
