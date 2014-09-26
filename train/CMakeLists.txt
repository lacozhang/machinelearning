
cmake_minimum_required(VERSION 2.8.4)
set(target_name train)
PROJECT(${target_name})

include_directories(~/src/tools/eigen)
include_directories(~/src/tools/include/boost_1_56_0)
include_directories(~/src/lbfgs/include)


ADD_EXECUTABLE(${target_name} 
 cmdline.h
 dataop.h
 parameters.h
 train.h
 util.h
 cmdline.cpp
 dataop.cpp
 main.cpp
 train.cpp
 util.cpp
 ../lib/arithmetic_ansi.h
 ../lib/arithmetic_sse_double.h
 ../lib/arithmetic_sse_float.h
 ../lib/lbfgs.c
)

TARGET_LINK_LIBRARIES(${target_name} boost_regex boost_program_options boost_system boost_filesystem)
