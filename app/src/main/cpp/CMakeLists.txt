project(neon_simd)

# Sets the minimum version of CMake required to build your native library.
# This ensures that a certain set of CMake features is available to
# your build.
cmake_minimum_required(VERSION 3.10)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if (${ANDROID_ABI} STREQUAL "armeabi-v7a")
    # make a list of neon files and add neon compiling flags to them
    set(neon_SRCS neon_ops.cpp)

    set_property(SOURCE ${neon_SRCS}
            APPEND_STRING PROPERTY COMPILE_FLAGS " -mfpu=neon")
    add_definitions("-DHAVE_NEON=1")
elseif (${ANDROID_ABI} STREQUAL "x86")
    set(neon_SRCS neon_ops.cpp)
    set_property(SOURCE ${neon_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS
            " -mssse3  -Wno-unknown-attributes \
                   -Wno-deprecated-declarations \
                   -Wno-constant-conversion \
                   -Wno-static-in-inline")
    add_definitions(-DHAVE_NEON_X86=1 -DHAVE_NEON=1)
else ()
    set(neon_SRCS)
endif ()

find_library( # Sets the name of the path variable.
        android-lib

        # Specifies the name of the NDK library that
        # you want CMake to locate.
        android )

find_library(
        log-lib
        log
)

add_library(
        neon_op
        SHARED
        neon_ops.cpp
        ${neon_SRCS}
)

target_link_libraries(
        neon_op
        ${android-lib}
        ${log-lib}
)
