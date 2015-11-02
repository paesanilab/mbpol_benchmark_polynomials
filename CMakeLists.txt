cmake_minimum_required (VERSION 2.6)

project(test2b)

find_package( Boost 1.55 COMPONENTS timer system REQUIRED )
find_package( CUDA REQUIRED )
#include_directories( ${Boost_INCLUDE_DIR} )

#add_definitions(-DDISABLEPOLYCPU)
#add_definitions(-DDISABLEPOLYGPU)

add_library(poly-2b-v6x poly-2b-v6x.cpp)
add_library(cpu_direct cpu_direct.cpp)
cuda_add_library(cuda_2b_direct kernels/2b_direct.cu)
cuda_add_library(cuda_2b_maple kernels/2b_maple.cu)

add_executable(run_test run_test.cpp)

target_link_libraries(run_test poly-2b-v6x cpu_direct cuda_2b_direct cuda_2b_maple ${Boost_LIBRARIES})
