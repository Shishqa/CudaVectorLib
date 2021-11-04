cmake_minimum_required(VERSION 3.10)

cuda_add_library(algebra
  ./CommonUtils.cu
  ./Matrix.cu
  ./MatrixOps.cu
  ./Reduce.cu
  ./Vector.cu
  ./VectorOps.cu

  STATIC
)

target_link_libraries(algebra cuda_kernels)

add_subdirectory(kernels)
