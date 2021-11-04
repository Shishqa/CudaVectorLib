#pragma once


__global__
void MatrixMul(size_t resultH, size_t resultW, size_t commonSide,
               cudaPitchedPtr A, cudaPitchedPtr B, cudaPitchedPtr result);
