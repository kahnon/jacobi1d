cmake_minimum_required(VERSION 3.11...3.17)

#project description
project(jacobi_1d
  VERSION 0.1
  DESCRIPTION "1D jacobi solver skeleton for LSE"
  LANGUAGES CXX
)

#prevent in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR "In-source builds not allowed. Use -B option for out-of-source build.")
endif()

set( CMAKE_BUILD_TYPE DEBUG )
option( USE_ADDRESS_SANITIZER on)

set( CMAKE_CXX_STANDARD 20 )
set( CMAKE_CXX_FLAGS "-fopenmp -Wall -pedantic-errors -march=native -pipe" )
set( CMAKE_CXX_FLAGS_RELEASE "-O3 -fno-exceptions -fno-rtti -DNDEBUG" )
set( CMAKE_CXX_FLAGS_DEBUG "-Og -D_GLIBCXX_ASSERTIONS -D_GLIBCXX_DEBUG -D_GLIBCXX_DEBUG_PEDANTIC" )

if ( ${USE_ADDRESS_SANITIZER} )
  set( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address" )
endif()

add_executable(bin_jacobi_1d jacobi_1d.cpp)
#target_include_directories( bin_jacobi_1d )
#target_link_libraries( bin_jacobi_1d )

