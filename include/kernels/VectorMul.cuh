#pragma once

__global__
void KernelMul(size_t numElements, float *x, float *y, float *result);
