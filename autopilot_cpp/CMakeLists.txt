cmake_minimum_required(VERSION 3.16)
project(autopilot_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(SDL2 REQUIRED sdl2)

set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}" CACHE PATH "ONNX Runtime install prefix")
if(NOT ONNXRUNTIME_ROOT)
    message(FATAL_ERROR
        "Set ONNXRUNTIME_ROOT to the extracted onnxruntime-linux-* directory.\n"
        "Example: cmake -B build -DONNXRUNTIME_ROOT=/opt/onnxruntime")
endif()

find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    HINTS "${ONNXRUNTIME_ROOT}/include")
find_library(ONNXRUNTIME_LIB onnxruntime
    HINTS "${ONNXRUNTIME_ROOT}/lib")

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR "ONNX Runtime not found under ONNXRUNTIME_ROOT=${ONNXRUNTIME_ROOT}")
endif()

add_executable(autopilot_cpp
    src/main.cpp
    src/config.cpp
    src/vision.cpp
    src/model.cpp
    src/vesc.cpp
    src/camera.cpp
    src/stream.cpp
    src/gamepad.cpp
)

target_include_directories(autopilot_cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${SDL2_INCLUDE_DIRS}
)

target_link_libraries(autopilot_cpp PRIVATE
    ${OpenCV_LIBS}
    ${SDL2_LIBRARIES}
    ${ONNXRUNTIME_LIB}
    pthread
)

install(TARGETS autopilot_cpp RUNTIME DESTINATION bin)
