import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.kotlin.serialization)
}

/**
 * Location of the OpenCV Android SDK (the directory that contains `sdk/`).
 * Resolution order: -Popencvsdk=..., `opencvsdk` in gradle.properties / ~/.gradle/gradle.properties,
 * the OPENCV_ANDROID_SDK environment variable, then ~/Downloads/OpenCV-android-sdk.
 * settings.gradle.kts uses the same rule to locate the `:opencv` module.
 */
val opencvSdkDir: File = run {
    val fromProperty = (findProperty("opencvsdk") as String?)
        ?.takeIf { it.isNotBlank() && !it.startsWith("/path/to") }
    val fromEnv = System.getenv("OPENCV_ANDROID_SDK")?.takeIf { it.isNotBlank() }
    val candidate = fromProperty ?: fromEnv
        ?: File(System.getProperty("user.home"), "Downloads/OpenCV-android-sdk").path
    File(candidate).also {
        check(File(it, "sdk/native/jni/OpenCVConfig.cmake").exists()) {
            "OpenCV Android SDK not found at '$it'. Set opencvsdk=<path> in gradle.properties " +
                "(or the OPENCV_ANDROID_SDK environment variable) to the directory containing 'sdk/'."
        }
    }
}

android {
    namespace = "de.tudarmstadt.physics.trackingplot"
    compileSdk {
        version = release(36)
    }

    // NDK r28+ links shared libraries with 16 KiB page alignment by default, which is required
    // on devices that run with a 16 KiB page size (Pixel 8/9 in 16 KB mode, many newer devices).
    // The linker flags in CMakeLists.txt enforce it even if this is changed to an older NDK.
    ndkVersion = "28.2.13676358"

    defaultConfig {
        applicationId = "de.tudarmstadt.physics.trackingplot"
        minSdk = 24
        targetSdk = 36
        versionCode = 2
        versionName = "1.1"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        externalNativeBuild {
            cmake {
                cppFlags.addAll(listOf("-frtti", "-fexceptions", "-std=c++17"))
                // Use the same C++ runtime as libopencv_java4.so (which needs libc++_shared.so).
                // Mixing a static libc++ in our library with the shared one is unsupported by the NDK.
                arguments(
                    "-DANDROID_STL=c++_shared",
                    "-DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON",
                    "-DOpenCV_DIR=${opencvSdkDir.path}/sdk/native/jni"
                )
                // arm64-v8a covers every current phone; x86_64 makes the app runnable on emulators.
                abiFilters.addAll(listOf("arm64-v8a", "x86_64"))
            }
        }
    }

    splits {
        abi {
            isEnable = true
            reset()
            include("arm64-v8a", "x86_64")
            isUniversalApk = false
        }
    }

    buildTypes {
        debug {
            externalNativeBuild {
                cmake {
                    // Make libc++ precondition violations (e.g. dereferencing an empty
                    // std::optional) abort loudly in debug builds instead of being silent UB.
                    cppFlags.add("-D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_EXTENSIVE")
                }
            }
        }
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
            // The APK is distributed directly (see README), not through a store, so the debug
            // key is used on purpose. Replace with a real signing config before publishing.
            signingConfig = signingConfigs.getByName("debug")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
    buildFeatures {
        compose = true
    }
    testOptions {
        unitTests.isReturnDefaultValues = true
    }
}

kotlin {
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_17)
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.lifecycle.runtime.compose)
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation(libs.androidx.activity.compose)
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.ui.graphics)
    implementation(libs.androidx.compose.ui.tooling.preview)
    implementation(libs.androidx.compose.material3)
    implementation(libs.androidx.compose.material.icons.core)
    implementation(libs.androidx.compose.foundation)
    implementation(libs.androidx.navigation.compose)
    implementation(libs.kotlinx.serialization.json)
    implementation(libs.kotlinx.coroutines.android)

    implementation(libs.androidx.camera.core)
    implementation(libs.androidx.camera.camera2)
    implementation(libs.androidx.camera.lifecycle)
    implementation(libs.androidx.camera.compose)

    implementation(libs.mpandroidchart)

    implementation(project(":opencv"))

    testImplementation(libs.junit)
    testImplementation(libs.kotlinx.coroutines.test)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(platform(libs.androidx.compose.bom))
    androidTestImplementation(libs.androidx.compose.ui.test.junit4)
    debugImplementation(libs.androidx.compose.ui.tooling)
    debugImplementation(libs.androidx.compose.ui.test.manifest)
}
