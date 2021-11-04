#include <cassert>

#include <MatrixOps.cuh>

#include <kernels/MatrixAdd.cuh>
#include <kernels/MatrixMul.cuh>

#include <CommonUtils.cuh>


float MatrixAdd(dim3 block_size, const Matrix& a, const Matrix& b,
                Matrix& res) {
  assert(a.size_.x == b.size_.x);
  assert(a.size_.y == b.size_.y);
  const dim3 grid_size((a.size_.x + block_size.x - 1) / block_size.x,
                       (a.size_.y + block_size.y - 1) / block_size.y);

  return MeasureTime([&]() {
    KernelMatrixAdd<<<grid_size, block_size>>>(a.size_.y, a.size_.x, a.device_data_, b.device_data_, res.device_data_);
  });
}


float MatrixMul(dim3 block_size, const Matrix& a, const Matrix& b,
                Matrix& res) {
  assert(a.size_.x == b.size_.y);
  assert(res.size_.x == b.size_.x);
  assert(res.size_.y == a.size_.y);

  const dim3 grid_size((res.size_.x + block_size.x - 1) / block_size.x,
                       (res.size_.y + block_size.y - 1) / block_size.y);
  const size_t cache_size = 2 * block_size.x * block_size.y * sizeof(float);

  return MeasureTime([&]() {
    MatrixMul<<<grid_size, block_size, cache_size>>>(
      res.size_.y, res.size_.x, a.size_.x,
      a.device_data_, b.device_data_, res.device_data_
    );
  });
}
