#pragma once


class Matrix {
 public:
  Matrix(dim3 size);

  Matrix(dim3 size, const float *host_data);

  ~Matrix();

  Matrix(const Matrix& other) = delete;

  Matrix& operator=(const Matrix& other) = delete;

  Matrix(Matrix&& other);

  Matrix& operator=(Matrix&& other);

  dim3 Fetch(dim3 size, float *host_data);

 private:
  friend float MatrixAdd(dim3 block_size, const Matrix& a, const Matrix& b, Matrix& res);

  friend float MatrixMul(dim3 block_size, const Matrix& a, const Matrix& b, Matrix& res);

 private:
  dim3 size_;
  cudaPitchedPtr device_data_;
};
