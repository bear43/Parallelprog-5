cmake_minimum_required(VERSION 3.13)
project(parallel_prog5)

set(CMAKE_CXX_STANDARD 17)

link_libraries(-lmpi -lmpich)

add_executable(parallel_prog5 main.cpp)