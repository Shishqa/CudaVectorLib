#include <Matrix.cuh>


Matrix::Matrix(dim3 size)
    : size_(size) {
  cudaExtent extent = make_cudaExtent(size.x * sizeof(float), size.y, 1);
  cudaMalloc3D(&device_data_, extent);
}


Matrix::Matrix(dim3 size, const float *host_data)
    : size_(size) {
  cudaExtent extent = make_cudaExtent(size.x * sizeof(float), size.y, 1);
  cudaMalloc3D(&device_data_, extent);
  cudaMemcpy2D(device_data_.ptr, device_data_.pitch, host_data, size.x * sizeof(*host_data), size.x * sizeof(*host_data), size.y, cudaMemcpyHostToDevice);
}


Matrix::~Matrix() {
  if (device_data_.ptr) {
    cudaFree(device_data_.ptr);
  }
}


Matrix::Matrix(Matrix&& other)
    : size_(other.size_), device_data_(other.device_data_) {
  other.device_data_.ptr = nullptr;
  other.size_ = dim3(0, 0, 0);
}


Matrix& Matrix::operator=(Matrix&& other) {
  if (&other == this) {
    return *this;
  }

  device_data_ = other.device_data_;
  size_ = other.size_;

  other.device_data_.ptr = nullptr;
  other.size_ = dim3(0, 0, 0);

  return *this;
}


dim3 Matrix::Fetch(dim3 size, float *host_data) {
  cudaMemcpy2D(host_data, size.x * sizeof(*host_data), device_data_.ptr, device_data_.pitch, size.x * sizeof(*host_data), size.y, cudaMemcpyDeviceToHost);
  return size;
}


