#pragma once

#include <Matrix.cuh>


float MatrixAdd(dim3 block_size, const Matrix& a, const Matrix& b, Matrix& res);

float MatrixMul(dim3 block_size, const Matrix& a, const Matrix& b, Matrix& res);
