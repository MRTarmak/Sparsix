cmake_minimum_required(VERSION 3.20)

project(Sparsix VERSION 0.1.0 LANGUAGES CXX)

option(SPARSIX_BUILD_EXAMPLES "Build Sparsix examples" ON)
option(SPARSIX_BUILD_TESTS "Build Sparsix tests" ON)
option(SPARSIX_ENABLE_COVERAGE "Enable coverage instrumentation" OFF)

include(CTest)
include(FetchContent)

add_library(Sparsix INTERFACE)
add_library(Sparsix::Sparsix ALIAS Sparsix)

target_include_directories(Sparsix INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(Sparsix INTERFACE cxx_std_20)

if(SPARSIX_ENABLE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(Sparsix INTERFACE --coverage -O0 -g)
    target_link_options(Sparsix INTERFACE --coverage)
endif()

if(SPARSIX_BUILD_EXAMPLES)
    add_executable(sparsix_main examples/main.cpp)
    target_link_libraries(sparsix_main PRIVATE Sparsix::Sparsix)
    target_compile_features(sparsix_main PRIVATE cxx_std_20)
endif()

if(SPARSIX_BUILD_TESTS)
    enable_testing()

    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        )
        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
        set(gtest_force_shared_crt OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()

    add_executable(sparsix_tests
        tests/sparsix_tests.cpp
        tests/algorithms_tests.cpp
        tests/solvers_and_utilities_tests.cpp
    )
    target_link_libraries(sparsix_tests PRIVATE Sparsix::Sparsix GTest::gtest_main)
    target_compile_features(sparsix_tests PRIVATE cxx_std_20)

    if(SPARSIX_ENABLE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(sparsix_tests PRIVATE --coverage -O0 -g)
        target_link_options(sparsix_tests PRIVATE --coverage)
    endif()

    include(GoogleTest)
    gtest_discover_tests(sparsix_tests)

    add_custom_target(coverage
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --test-dir ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND gcov -b -r -s ${CMAKE_CURRENT_SOURCE_DIR} -o ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/sparsix_tests.dir/tests ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/sparsix_tests.dir/tests/sparsix_tests.cpp.gcno
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS sparsix_tests
        COMMENT "Running tests and collecting gcov coverage"
        VERBATIM
    )
endif()
