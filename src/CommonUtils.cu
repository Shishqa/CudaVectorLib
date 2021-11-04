#include <cstdio>

#include <CommonUtils.cuh>

float MeasureTime(Routine routine)
{
  cudaEvent_t start;
  cudaEvent_t stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  routine();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float elapsed = 0;
  cudaEventElapsedTime(&elapsed, start, stop);

  return elapsed;
}

void PrintMatrix(float *m, dim3 size, const char *sep)
{
  for (size_t i = 0; i < size.y; ++i) {
    for (size_t j = 0; j < size.x; ++j) {
      printf("%f%s", m[j + size.x * i], sep);
    }
    printf("\n");
  }
  printf("\n");
}

void FillMatrix(float *m, dim3 size, float value)
{
  for (size_t i = 0; i < size.x * size.y; ++i) {
    m[i] = value;
  }
}

void Report(size_t array_size, size_t block_size, float time)
{
  printf("%lu,%lu,%f\n", array_size, block_size, time);
}
