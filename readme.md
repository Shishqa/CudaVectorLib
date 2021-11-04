# Cuda Vector Lib

- [the task](https://gitlab.com/fpmi-atp/pd2021-supplementary/global/-/blob/master/homeworks/task2_cuda.md)

## Features implemented

- [x] vector addition
- [x] vector multiplication
- [x] vector dot product
- [x] cos between vectors
- [x] matrix addition
- [x] matrix multiplication
- [ ] matrix by vector multiplication
- [ ] filter

## Benchmarks

To see benchmarks results please visit [docs/benchmarks.md](./docs/benchmarks.md)

## Usage

To see usage examples please visit [docs/usage.md](./docs/usage.md)

## Build

```bash
mkdir build && cd build
cmake ..
make

./01-add 4096 16
```
