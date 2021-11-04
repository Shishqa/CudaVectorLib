#include <kernels/MatrixMul.cuh>

__global__
void MatrixMul(size_t resultH, size_t resultW, size_t commonSide,
               cudaPitchedPtr A, cudaPitchedPtr B,
               cudaPitchedPtr result)
{
  const size_t i = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t j = blockIdx.x * blockDim.x + threadIdx.x;

  const size_t bi = threadIdx.y;
  const size_t bj = threadIdx.x;

  extern __shared__ float s[];
  float *windowA = s;
  float *windowB = s + blockDim.x * blockDim.y;

  float sum = 0;
  for (size_t k = 0; k < commonSide; k += blockDim.x) {
    float *A_row = (float *)((char *)A.ptr + i * A.pitch);
    float *B_row = (float *)((char *)B.ptr + (k + bi) * B.pitch);
    windowA[bj + bi * blockDim.x] = k + bj < commonSide && i < resultH ? A_row[k + bj] : 0;
    windowB[bi + bj * blockDim.x] = k + bi < commonSide && j < resultW ? B_row[j] : 0;
    __syncthreads();

    for (size_t bk = 0; bk < blockDim.x; ++bk) {
      sum += windowA[bk + bi * blockDim.x] * windowB[bk + bj * blockDim.x];
    }
    __syncthreads();
  }

  ((float *)((char *)result.ptr + i * result.pitch))[j] = sum;
}
