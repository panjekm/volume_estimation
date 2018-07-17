cmake_minimum_required(VERSION 2.8.3)
project(volume_estimation)

add_compile_options (-fcolor-diagnostics)

add_definitions(-std=c++11)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

cs_add_executable(compute_volume
  src/compute_volume.cc
)
