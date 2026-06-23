cmake_minimum_required(VERSION 3.20)

project(Sparsix VERSION 0.1.0 LANGUAGES CXX)

option(SPARSIX_BUILD_EXAMPLES "Build Sparsix examples" ON)

add_library(Sparsix INTERFACE)
add_library(Sparsix::Sparsix ALIAS Sparsix)

target_include_directories(Sparsix INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(Sparsix INTERFACE cxx_std_17)

if(SPARSIX_BUILD_EXAMPLES)
    add_executable(sparsix_main examples/main.cpp)
    target_link_libraries(sparsix_main PRIVATE Sparsix::Sparsix)
    target_compile_features(sparsix_main PRIVATE cxx_std_17)
endif()
