#pragma once


__global__
void KernelMatrixAdd(size_t height, size_t width, cudaPitchedPtr A, cudaPitchedPtr B, cudaPitchedPtr result);
