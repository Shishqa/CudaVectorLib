#include <kernels/VectorScalarMul.cuh>

__global__
void ScalarMulBlock(int numElements, float *vector1, float *vector2, float *result)
{
  extern __shared__ int s[];
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < numElements) {
    s[index] = vector1[index] * vector2[index];
  }
  __syncthreads();

  for (size_t step = numElements >> 1; step > 0; step >>= 1) {
    if (index < step) {
      s[index] = s[index] + s[index + step];
    }
    __syncthreads();
  }

  if (index == 0) {
    result[blockIdx.x] = s[0];
  }
}

