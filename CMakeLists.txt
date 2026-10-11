cmake_minimum_required(VERSION 3.16)
project(time_series_lab VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_library(ts_core
    src/csv.cpp
    src/statistics.cpp
    src/arma11.cpp
    src/forecast.cpp
    src/diagnostics.cpp
)

target_include_directories(ts_core PUBLIC include)

target_compile_options(ts_core PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

add_executable(ts_lab src/main.cpp)
target_link_libraries(ts_lab PRIVATE ts_core)

add_executable(ts_tests tests/test_main.cpp)
target_link_libraries(ts_tests PRIVATE ts_core)

enable_testing()
add_test(NAME time_series_tests COMMAND ts_tests)

file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/CTestConfigFallback.cmake"
    CONTENT "if(NOT DEFINED CTEST_CONFIGURATION_TYPE OR \"\${CTEST_CONFIGURATION_TYPE}\" STREQUAL \"\")
    if(EXISTS \"\${CMAKE_CURRENT_LIST_DIR}/Release/$<TARGET_FILE_NAME:ts_tests>\")
        set(CTEST_CONFIGURATION_TYPE \"Release\")
    elseif(EXISTS \"\${CMAKE_CURRENT_LIST_DIR}/Debug/$<TARGET_FILE_NAME:ts_tests>\")
        set(CTEST_CONFIGURATION_TYPE \"Debug\")
    else()
        set(CTEST_CONFIGURATION_TYPE \"Release\")
    endif()
endif()
")

set_property(DIRECTORY PROPERTY TEST_INCLUDE_FILES "${CMAKE_CURRENT_BINARY_DIR}/CTestConfigFallback.cmake")

