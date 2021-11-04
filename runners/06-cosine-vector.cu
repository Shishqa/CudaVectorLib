#include <cstdio>
#include <cassert>

#include <VectorOps.cuh>
#include <CommonUtils.cuh>
#include <Reduce.cuh>

void PrintUsage(const char *programName)
{
  printf("usage: %s array_size block_size\n\n", programName);
}

int main(int argc, const char *argv[])
{
  if (argc != 3) {
    PrintUsage(argc == 0 ? "scalar_mul" : argv[0]);
    return 0;
  }

  const size_t arrSize = strtoull(argv[1], NULL, 10);
  const size_t blockSize = strtoull(argv[2], NULL, 10);

  float *x = new float[arrSize];
  float *y = new float[arrSize];

  static const float X_VAL =  3.0f;
  static const float Y_VAL = -2.0f;
  FillMatrix(x, dim3(arrSize, 1, 1), X_VAL);
  FillMatrix(y, dim3(arrSize, 1, 1), Y_VAL);

  Vector a(arrSize, x);
  Vector b(arrSize, y);

  delete[] x;
  delete[] y;

  float result = NAN;
  float elapsed = VectorCos(blockSize, a, b, result, ReducePyramidal);
  printf("r,");
  Report(arrSize, blockSize, elapsed);

  assert(result == -1);

  result = NAN;
  elapsed = VectorCos(blockSize, a, b, result, ReduceStride);
  printf("s,");
  Report(arrSize, blockSize, elapsed);

  assert(result == -1);

  return 0;
}

