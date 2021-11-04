#include <kernels/VectorAdd.cuh>

__global__
void KernelAdd(size_t numElements, float *x, float *y, float *result)
{
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;

  for (size_t i = index; i < numElements; i += stride) {
    result[i] = x[i] + y[i];
  }
}
