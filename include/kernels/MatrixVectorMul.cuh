#pragma once

__global__
void MatrixVectorMul(size_t height, size_t width, cudaPitchedPtr matrix, float *vector, float *result);
