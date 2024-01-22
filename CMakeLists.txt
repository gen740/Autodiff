cmake_minimum_required(VERSION 3.28)

project(autodiff CXX)
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_FLAGS "-Werror -Wall -Wextra")
set(CMAKE_CXX_STANDARD 23)

add_library(autodiff INTERFACE)
target_include_directories(autodiff INTERFACE ${CMAKE_CURRENT_LIST_DIR}/include)

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_LIST_DIR})
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
  )
  FetchContent_MakeAvailable(googletest)
  file(GLOB AUTODIFF_TEST_SOURCES ${CMAKE_CURRENT_LIST_DIR}/tests/*.cc)
  add_executable(test-autodiff ${AUTODIFF_TEST_SOURCES})
  target_link_libraries(test-autodiff autodiff GTest::gtest GTest::gtest_main
                        GTest::gmock)
endif()
