#include <cassert>

#include <VectorOps.cuh>

#include <kernels/VectorAdd.cuh>
#include <kernels/VectorMul.cuh>
#include <Reduce.cuh>

#include <CommonUtils.cuh>


float VectorAdd(size_t block_size, const Vector& a, const Vector& b,
                Vector& res) {
  assert(a.size_ == b.size_);
  const size_t grid_size = (a.size_ + block_size - 1) / block_size;

  return MeasureTime([&]() {
    KernelAdd<<<grid_size, block_size>>>(a.size_, a.device_data_, b.device_data_, res.device_data_);
  });
}


float VectorMul(size_t block_size, const Vector& a, const Vector& b,
                Vector& res) {
  assert(a.size_ == b.size_);
  const size_t grid_size = (a.size_ + block_size - 1) / block_size;

  return MeasureTime([&]() {
    KernelMul<<<grid_size, block_size>>>(a.size_, a.device_data_, b.device_data_, res.device_data_);
  });
}


float VectorDot(size_t block_size, const Vector& a, const Vector& b,
                float& res, Reducer reducer) {
  assert(a.size_ == b.size_);
  Vector tmp(a.size_);

  size_t grid_size = (a.size_ + block_size - 1) / block_size;
  float elapsed = MeasureTime([&]() {
    KernelMul<<<grid_size, block_size>>>(a.size_, a.device_data_, b.device_data_, tmp.device_data_);
    reducer(block_size, tmp);
  });

  tmp.Fetch(1, &res);
  return elapsed;
}


float VectorCos(size_t block_size, const Vector& a, const Vector& b,
                float &res, Reducer reducer)
{
  float a_dot_b, a_dot_a, b_dot_b;
  float elapsed_1 = VectorDot(block_size, a, b, a_dot_b, reducer);
  float elapsed_2 = VectorDot(block_size, a, a, a_dot_a, reducer);
  float elapsed_3 = VectorDot(block_size, b, b, b_dot_b, reducer);

  res = a_dot_b / sqrt(a_dot_a * b_dot_b);
  return elapsed_1 + elapsed_2 + elapsed_3;
}

