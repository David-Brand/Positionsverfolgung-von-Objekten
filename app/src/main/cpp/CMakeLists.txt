cmake_minimum_required(VERSION 3.22.1)

project("native-lib" CXX)

# OpenCV is provided by the prebuilt Android SDK. OpenCV_DIR is passed from build.gradle.kts and
# points at <OpenCV-android-sdk>/sdk/native/jni; it defines the imported target `opencv_java`
# (libopencv_java4.so), which is the library the Java side loads as well.
find_package(OpenCV REQUIRED)

add_library(${CMAKE_PROJECT_NAME} SHARED
        native-lib.cpp
        TrackerManager.cpp)

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
target_compile_features(${CMAKE_PROJECT_NAME} PRIVATE cxx_std_17)
target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE -Wall -Wextra)

# 16 KiB page-size support (required on Android 15+ devices with 16 KiB pages). NDK r28+ does
# this by default; the explicit flags keep the library correct with older NDKs as well.
target_link_options(${CMAKE_PROJECT_NAME} PRIVATE
        "-Wl,-z,max-page-size=16384"
        "-Wl,-z,common-page-size=16384")

target_link_libraries(${CMAKE_PROJECT_NAME}
        android
        log
        opencv_java)
