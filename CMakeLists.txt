cmake_minimum_required(VERSION 3.28)
project(MlLib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)

add_subdirectory(lib)

add_subdirectory(test)

find_package(TBB REQUIRED)

target_link_libraries(ml-lib PRIVATE TBB::tbb)