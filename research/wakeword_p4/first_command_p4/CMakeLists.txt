cmake_minimum_required(VERSION 3.16)
set(COMPONENTS main)
# The early component-requirements scan runs in a separate CMake process.
set(ENV{NYX_SR_COMPONENT} "${NYX_SR_COMPONENT}")
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(nyx_first_command)
