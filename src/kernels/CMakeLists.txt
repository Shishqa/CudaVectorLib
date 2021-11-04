cmake_minimum_required(VERSION 3.10)

cuda_add_library(cuda_kernels
  ./MatrixAdd.cu
  ./MatrixMul.cu
  ./Reduce.cu
  ./VectorAdd.cu
  ./VectorMul.cu
  ./VectorScalarMul.cu

  STATIC
)
