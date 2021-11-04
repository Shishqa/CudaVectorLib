#include <kernels/Reduce.cuh>


__global__
void AccumulateToBlock(size_t numElements, float *vector, float *result)
{
  const size_t index = blockDim.x * blockIdx.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;

  float sum = 0;
  for (size_t i = index; i < numElements; i += stride) {
    sum += vector[i];
  }

  if (index < numElements) result[index] = sum;
}

__global__
void ReduceOnBlock(int numElements, float *arr, float *result)
{
  extern __shared__ int s[];
  const size_t tid = threadIdx.x;
  const size_t i = blockDim.x * blockIdx.x * 2 + threadIdx.x;

  if (i + blockDim.x < numElements) {
    s[tid] = arr[i] + arr[i + blockDim.x];
  } else if (i < numElements) {
    s[tid] = arr[i];
  } else {
    s[tid] = 0;
  }
  __syncthreads();

  for (size_t step = blockDim.x >> 1; step > 0; step >>= 1) {
    if (tid < step) {
      s[tid] += s[tid + step];
    }
    __syncthreads();
  }

  if (tid == 0) {
    result[blockIdx.x] = s[0];
  }
}
