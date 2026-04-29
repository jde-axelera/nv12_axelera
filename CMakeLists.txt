cmake_minimum_required(VERSION 3.12)
project(yolov5s_nv12 LANGUAGES C CXX VERSION 1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(axruntime REQUIRED axruntime)
pkg_check_modules(opencv4 REQUIRED opencv4)

add_executable(${PROJECT_NAME}
    src/main.cpp
    src/preprocess.cpp
    src/yolo_decode.cpp
    src/annotate.cpp
)

target_compile_options(${PROJECT_NAME} PRIVATE
    -Wall -Werror -O3 -march=native
    -Wno-sign-compare -Wno-unused-variable
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-enum-enum-conversion>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-declarations>
)

target_include_directories(${PROJECT_NAME} PRIVATE
    include
    ${axruntime_INCLUDE_DIRS}
    ${opencv4_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PUBLIC
    ${axruntime_LIBRARIES}
    ${opencv4_LIBRARIES}
)

target_link_directories(${PROJECT_NAME} PUBLIC
    ${axruntime_LIBRARY_DIRS}
    ${opencv4_LIBRARY_DIRS}
)

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
