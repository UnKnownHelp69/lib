cmake_minimum_required(VERSION 3.28)
project(MlLib LANGUAGES CXX)

add_subdirectory(lib)

enable_testing()
add_subdirectory(test)