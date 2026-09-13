# Minification is currently disabled (see build.gradle.kts). These rules keep the build
# working if it is ever enabled.

# JNI: native method declarations must keep their names and owning class.
-keepclasseswithmembers class de.tudarmstadt.physics.trackingplot.tracker.NativeTracker {
    native <methods>;
}

# OpenCV's Java bindings are looked up by name from libopencv_java4.so.
-keep class org.opencv.** { *; }

# Keep line numbers for readable crash reports.
-keepattributes SourceFile,LineNumberTable
