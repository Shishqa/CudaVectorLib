#pragma once

#include <functional>


using Routine = std::function<void(void)>;

float MeasureTime(Routine routine);


void PrintMatrix(float *m, dim3 size, const char *sep = "\t");

void FillMatrix(float *m, dim3 size, float value);

void Report(size_t array_size, size_t block_size, float time);
