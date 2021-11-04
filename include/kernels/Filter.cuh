#pragma once

enum OperationFilterType {
  GT,
  LT
};

__global__
void Filter(size_t numElements, float *array, OperationFilterType type, float value, float *result);
