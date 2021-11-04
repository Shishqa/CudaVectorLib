cmake_minimum_required(VERSION 3.10)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(CudaTasks)
find_package(CUDA REQUIRED)

include_directories(include)


cuda_add_executable(01-add runners/01-add.cu)
cuda_add_executable(02-mul runners/02-mul.cu)
cuda_add_executable(03-matrix-add runners/03-matrix-add.cu)
cuda_add_executable(05-scalar-mul runners/05-scalar-mul.cu)
cuda_add_executable(06-cosine-vector runners/06-cosine-vector.cu)
cuda_add_executable(07-matrix-mul runners/07-matrix-mul.cu)

target_link_libraries(01-add algebra)
target_link_libraries(02-mul algebra)
target_link_libraries(03-matrix-add algebra)
target_link_libraries(05-scalar-mul algebra)
target_link_libraries(06-cosine-vector algebra)
target_link_libraries(07-matrix-mul algebra)


add_subdirectory(src)



