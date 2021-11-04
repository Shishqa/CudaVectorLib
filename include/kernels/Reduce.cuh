#pragma once


__global__
void AccumulateToBlock(size_t numElements, float *vector, float *result);


__global__
void ReduceOnBlock(int numElements, float *arr, float *result);
