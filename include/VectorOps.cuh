#pragma once

#include <Vector.cuh>


float VectorAdd(size_t block_size, const Vector& a, const Vector& b,
                Vector& res);

float VectorMul(size_t block_size, const Vector& a, const Vector& b,
                Vector& res);

float VectorDot(size_t block_size, const Vector& a, const Vector& b,
                float& res, Reducer reducer);

float VectorCos(size_t block_size, const Vector& a, const Vector& b,
                float &res, Reducer reducer);
