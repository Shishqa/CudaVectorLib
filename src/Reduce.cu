#include <Reduce.cuh>

#include <kernels/Reduce.cuh>


void ReducePyramidal(size_t block_size, Vector& v) {
  size_t blk_ilp2 = block_size << 1;  // ILP: 2

  size_t vec_size = v.size_;
  size_t grid_size = (v.size_ + blk_ilp2 - 1) / blk_ilp2;
  while (vec_size > 1) {
    ReduceOnBlock<<<grid_size, block_size, block_size * sizeof(float)>>>(vec_size, v.device_data_, v.device_data_);
    vec_size = grid_size;
    grid_size = (grid_size + blk_ilp2 - 1) / blk_ilp2;
  }
}


void ReduceStride(size_t block_size, Vector& v) {
  size_t blk_ilp2 = block_size << 1;  // ILP: 2

  if (v.size_ > blk_ilp2) {
    AccumulateToBlock<<<2, block_size>>>(v.size_, v.device_data_, v.device_data_);
  }

  ReduceOnBlock<<<1, block_size, block_size * sizeof(float)>>>(blk_ilp2, v.device_data_, v.device_data_);
}
