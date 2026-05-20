cmake_minimum_required(VERSION 3.12)
project(axelera_evs LANGUAGES C CXX VERSION 1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(axruntime REQUIRED axruntime)
pkg_check_modules(opencv4 REQUIRED opencv4)

set(COMPILE_FLAGS
    -Wall -Werror -O3 -march=native
    -Wno-sign-compare -Wno-unused-variable
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-enum-enum-conversion>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-declarations>
)

# ── Helper macro ──────────────────────────────────────────────────────────────
macro(axelera_target name)
    target_compile_options(${name} PRIVATE ${COMPILE_FLAGS})
    target_include_directories(${name} PRIVATE
        include ${axruntime_INCLUDE_DIRS} ${opencv4_INCLUDE_DIRS})
    target_link_libraries(${name} PUBLIC
        ${axruntime_LIBRARIES} ${opencv4_LIBRARIES})
    target_link_directories(${name} PUBLIC
        ${axruntime_LIBRARY_DIRS} ${opencv4_LIBRARY_DIRS})
endmacro()

# ── yolov5s_nv12 — NV12 object detection ──────────────────────────────────────
add_executable(yolov5s_nv12
    src/main.cpp
    src/preprocess.cpp
    src/yolo_decode.cpp
    src/annotate.cpp
)
axelera_target(yolov5s_nv12)

# ── feature_extraction — generic embedding model runner ───────────────────────
add_executable(feature_extraction
    src/main_feature_extraction.cpp
    src/preprocess.cpp
)
axelera_target(feature_extraction)

install(TARGETS yolov5s_nv12 feature_extraction RUNTIME DESTINATION bin)
