#include <cstdio>
#include <cstdlib>
#include <cassert>

#include <CommonUtils.cuh>
#include <VectorOps.cuh>

void PrintUsage(const char *programName)
{
  printf("usage: %s array_size block_size\n\n", programName);
}

int main(int argc, char *argv[])
{
  if (argc != 3) {
    PrintUsage(argc == 0 ? "kernel_mul" : argv[0]);
    return 0;
  }

  const size_t arrSize = strtoull(argv[1], NULL, 10);
  const size_t blockSize = strtoull(argv[2], NULL, 10);

  float *x = new float[arrSize];
  float *y = new float[arrSize];

  static const float X_VAL = 5.0f;
  static const float Y_VAL = 2.0f;
  FillMatrix(x, dim3(arrSize, 1, 1), X_VAL);
  FillMatrix(y, dim3(arrSize, 1, 1), Y_VAL);

  Vector a(arrSize, x);
  Vector b(arrSize, y);
  Vector c(arrSize);

  delete[] x;
  delete[] y;

  float elapsed = VectorMul(blockSize, a, b, c);
  Report(arrSize, blockSize, elapsed);

  float *r = new float[arrSize];
  c.Fetch(arrSize, r);

  float maxError = 0.0f;
  for (size_t i = 0; i < arrSize; i++) {
    maxError = fmax(maxError, fabs(r[i] - (X_VAL * Y_VAL)));
  }
  assert(maxError < 0.001);

  return 0;
}
