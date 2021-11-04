#pragma once


class Vector;

using Reducer = void (*)(size_t size, Vector& vector);

class Vector {
 public:
  Vector(size_t size);

  Vector(size_t size, const float *host_data);

  ~Vector();

  Vector(const Vector& other) = delete;

  Vector& operator=(const Vector& other) = delete;

  Vector(Vector&& other);

  Vector& operator=(Vector&& other);

  size_t Fetch(size_t size, float *host_data);

 private:
  friend float VectorAdd(size_t block_size, const Vector& a,
                         const Vector& b, Vector& res);

  friend float VectorMul(size_t block_size, const Vector& a,
                         const Vector& b, Vector& res);

  friend void ReducePyramidal(size_t block_size, Vector& v);

  friend void ReduceStride(size_t block_size, Vector& v);

  friend float VectorDot(size_t block_size, const Vector& a, const Vector& b,
                         float& res, Reducer reducer);

 private:
  size_t size_;
  float *device_data_;
};
