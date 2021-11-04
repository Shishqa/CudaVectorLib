#include <cstdio>
#include <cstdlib>
#include <cassert>

#include <MatrixOps.cuh>
#include <CommonUtils.cuh>

void PrintUsage(const char *programName)
{
  printf("usage: %s matrix_size block_size\n\n", programName);
}

int main(int argc, char *argv[])
{
  if (argc != 3) {
    PrintUsage(argc == 0 ? "kernel_add" : argv[0]);
    return 0;
  }

  size_t matrixSz = strtoull(argv[1], NULL, 10);
  dim3 size(matrixSz, matrixSz);

  size_t blockSz = strtoull(argv[2], NULL, 10);
  dim3 blockSize(blockSz, blockSz);

  float *x = new float[size.x * size.y];
  float *y = new float[size.x * size.y];

  static const float X_VAL = 1.0f;
  static const float Y_VAL = 2.0f;
  FillMatrix(x, size, X_VAL);
  FillMatrix(y, size, Y_VAL);

  Matrix a(size, x);
  Matrix b(size, y);

  delete[] x;
  delete[] y;

  Matrix c(size);
  float elapsed = MatrixAdd(blockSz, a, b, c);
  Report(matrixSz, blockSz, elapsed);

  float *r = new float[size.x * size.y];
  c.Fetch(size, r);

  float maxError = 0.0f;
  for (size_t i = 0; i < size.x * size.y; i++) {
    maxError = fmax(maxError, fabs(r[i] - (X_VAL + Y_VAL)));
  }
  assert(maxError < 0.001);

  delete[] r;
  return 0;
}
