#include <kernels/MatrixAdd.cuh>

__global__
void KernelMatrixAdd(size_t height, size_t width, cudaPitchedPtr A, cudaPitchedPtr B, cudaPitchedPtr result)
{
  const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t x = blockIdx.x * blockDim.x + threadIdx.x;

  const size_t stride_y = gridDim.y * blockDim.y;
  const size_t stride_x = gridDim.x * blockDim.x;

  for (size_t i = y; i < height; i += stride_y) {
    const float *A_row = (float *)((char *)A.ptr + i * A.pitch);
    const float *B_row = (float *)((char *)B.ptr + i * B.pitch);
    float *R_row = (float *)((char *)result.ptr + i * result.pitch);
    for (size_t j = x; j < width; j += stride_x) {
      R_row[j] = A_row[j] + B_row[j];
    }
  }
}

