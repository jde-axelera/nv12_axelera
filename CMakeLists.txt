cmake_minimum_required(VERSION 3.12)
project(yolov5s_nv12 LANGUAGES C CXX VERSION 1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(axruntime REQUIRED axruntime)
pkg_check_modules(opencv4 REQUIRED opencv4)

set(SHARED_SOURCES
    src/preprocess.cpp
    src/yolo_decode.cpp
    src/annotate.cpp
)

set(COMPILE_FLAGS
    -Wall -Werror -O3 -march=native
    -Wno-sign-compare -Wno-unused-variable
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-enum-enum-conversion>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-declarations>
)

# ── NV12 pipeline (original) ──────────────────────────────────────────────────
add_executable(${PROJECT_NAME} src/main.cpp ${SHARED_SOURCES})
target_compile_options(${PROJECT_NAME} PRIVATE ${COMPILE_FLAGS})
target_include_directories(${PROJECT_NAME} PRIVATE
    include ${axruntime_INCLUDE_DIRS} ${opencv4_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PUBLIC
    ${axruntime_LIBRARIES} ${opencv4_LIBRARIES})
target_link_directories(${PROJECT_NAME} PUBLIC
    ${axruntime_LIBRARY_DIRS} ${opencv4_LIBRARY_DIRS})

# ── RGBA pipeline (FPGA direct output, file-based test) ──────────────────────
add_executable(yolov5s_rgba src/main_rgba.cpp ${SHARED_SOURCES})
target_compile_options(yolov5s_rgba PRIVATE ${COMPILE_FLAGS})
target_include_directories(yolov5s_rgba PRIVATE
    include ${axruntime_INCLUDE_DIRS} ${opencv4_INCLUDE_DIRS})
target_link_libraries(yolov5s_rgba PUBLIC
    ${axruntime_LIBRARIES} ${opencv4_LIBRARIES})
target_link_directories(yolov5s_rgba PUBLIC
    ${axruntime_LIBRARY_DIRS} ${opencv4_LIBRARY_DIRS})

# ── RGBA pipeline with simulated FPGA DMA integration ────────────────────────
add_executable(yolov5s_rgba_fpga src/main_rgba_fpga.cpp ${SHARED_SOURCES})
target_compile_options(yolov5s_rgba_fpga PRIVATE ${COMPILE_FLAGS})
target_include_directories(yolov5s_rgba_fpga PRIVATE
    include ${axruntime_INCLUDE_DIRS} ${opencv4_INCLUDE_DIRS})
target_link_libraries(yolov5s_rgba_fpga PUBLIC
    ${axruntime_LIBRARIES} ${opencv4_LIBRARIES})
target_link_directories(yolov5s_rgba_fpga PUBLIC
    ${axruntime_LIBRARY_DIRS} ${opencv4_LIBRARY_DIRS})

install(TARGETS ${PROJECT_NAME} yolov5s_rgba yolov5s_rgba_fpga RUNTIME DESTINATION bin)
