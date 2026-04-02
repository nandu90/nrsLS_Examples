cmake_minimum_required(VERSION 3.21)
project(nrsLSTests NONE)
enable_testing()

if(NOT DEFINED ENV{NEKRS_HOME})
    message(FATAL_ERROR "NEKRS_HOME is not defined. Please set the environment variable.")
else()
    message(STATUS "NEKRS_HOME is set to: $ENV{NEKRS_HOME}")
endif()
set (NEKRS_HOME "$ENV{NEKRS_HOME}")

set(BACKEND "@BACKEND_DEFAULT@")
set(EXAMPLES_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(LAUNCHER "${NEKRS_HOME}/bin/nrsmpi" CACHE STRING "Path to the launcher executable")
set(NRANKS "0" CACHE STRING "Number of MPI ranks to run a test")
set(TESTS "")

 message(STATUS "backend: ${BACKEND}")
if(NRANKS GREATER 0)
  message(STATUS "numer of MPI ranks to run tests: ${NRANKS}")
endif()

function(get_tests filter_string result_var)
  get_property(all_tests GLOBAL PROPERTY CTEST_TESTS)
  set(matching_tests "")
  foreach(test_name ${all_tests})
    string(FIND "${test_name}" "${filter_string}" found_index)
    if(found_index GREATER -1)
      list(APPEND matching_tests ${test_name})
    endif()
  endforeach()
  set(${result_var} "${matching_tests}" PARENT_SCOPE)
endfunction()

function(add target num_processes par_file cimodes fp32 working_directory build_only)
  if(working_directory STREQUAL "" )
    set(working_directory "${EXAMPLES_DIR}/${target}")
  endif()

  if(fp32)
    set(fp32_value 1)
  else()
    set(fp32_value 0)
  endif()

  foreach(cimode ${cimodes})
    set (name "${target}_${cimode}")
    list(APPEND TESTS ${test_name}) 
    message(STATUS "adding test ${name}")
 
    if(NOT TARGET ${target})
      add_custom_target(${target})
    endif()
 
    if(NRANKS GREATER 0)
      set(num_processes ${NRANKS})
    endif()
 
    if(build_only)
      add_test(NAME ${name} WORKING_DIRECTORY ${working_directory}
               COMMAND ${LAUNCHER} ${par_file} ${num_processes} --cimode ${cimode} --build-only ${num_processes})
    else()
      add_test(NAME ${name} WORKING_DIRECTORY ${working_directory}
               COMMAND ${LAUNCHER} ${par_file} ${num_processes} --cimode ${cimode})
    endif()

    set_tests_properties(${name} PROPERTIES ENVIRONMENT "FP32=${fp32_value}")
    set_tests_properties(${name} PROPERTIES RESOURCE_LOCK ${working_directory})
  endforeach()
endfunction()

set(USE_FP32 OFF)
set(BUILD_ONLY OFF)
set(dir "")

## TESTS ##

add("sineWave" 2 "sine.par" "1" "${USE_FP32}" "${dir}" "${BUILD_ONLY}")

add("linear1D" 2 "linear.par" "1" "${USE_FP32}" "${dir}" "${BUILD_ONLY}")

add("circles" 2 "circles.par" "1" "${USE_FP32}" "${dir}" "${BUILD_ONLY}")

add("tlsTest" 2 "tls.par" "1;2" "${USE_FP32}" "${dir}" "${BUILD_ONLY}")

add("clsTest" 2 "cls.par" "1;2;3;4" "${USE_FP32}" "${dir}" "${BUILD_ONLY}")

###

foreach(test ${TESTS})
  set_tests_properties(${test} PROPERTIES EXCLUDE_FROM_ALL TRUE)
endforeach()
get_property(all_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
add_custom_target(run_all_targets ALL DEPENDS ${all_targets})
