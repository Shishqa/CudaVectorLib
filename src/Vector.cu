#include <cassert>

#include <Vector.cuh>

#include <kernels/VectorAdd.cuh>
#include <kernels/VectorMul.cuh>
#include <Reduce.cuh>

#include <CommonUtils.cuh>


Vector::Vector(size_t size)
    : size_(size) {
  cudaMalloc(&device_data_, size * sizeof(*device_data_));
}


Vector::Vector(size_t size, const float *host_data)
    : size_(size) {
  cudaMalloc(&device_data_, size * sizeof(*device_data_));
  cudaMemcpy(device_data_, host_data, size * sizeof(*host_data),
             cudaMemcpyHostToDevice);
}


Vector::~Vector() {
  if (device_data_) {
    cudaFree(device_data_);
  }
}


Vector::Vector(Vector&& other)
    : size_(other.size_), device_data_(other.device_data_) {
  other.device_data_ = nullptr;
  other.size_ = 0;
}


Vector& Vector::operator=(Vector&& other) {
  if (&other == this) {
    return *this;
  }

  device_data_ = other.device_data_;
  size_ = other.size_;

  other.device_data_ = nullptr;
  other.size_ = 0;

  return *this;
}


size_t Vector::Fetch(size_t size, float *host_data) {
  size_t fetch_size = (size > size_) ? size_ : size;
  cudaMemcpy(host_data, device_data_, fetch_size * sizeof(*host_data), cudaMemcpyDeviceToHost);
  return fetch_size;
}


