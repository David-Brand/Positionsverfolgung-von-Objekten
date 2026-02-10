plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)

    alias(libs.plugins.kotlin.serialization)
}

android {
    namespace = "de.tudarmstadt.physics.trackingplot"
    compileSdk {
        version = release(36)
    }

    defaultConfig {
        applicationId = "de.tudarmstadt.physics.trackingplot"
        minSdk = 24
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        externalNativeBuild {
            cmake {
                cppFlags.addAll(listOf("-frtti", "-fexceptions"))
                abiFilters.addAll(listOf("armeabi-v7a", "arm64-v8a", "x86_64", "x86"))

//                arguments("-DOpenCV_DIR=${project.property("opencvsdk")}/sdk/native/jni")
                val path = File(System.getProperty("user.home"), "Downloads/OpenCV-android-sdk").path
                arguments("-DOpenCV_DIR=${path}/sdk/native/jni")
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
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
}

//val cameraxVersion = "1.5.2"
dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.activity.compose)
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.ui.graphics)
    implementation(libs.androidx.compose.ui.tooling.preview)
    implementation(libs.androidx.compose.material3)
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(platform(libs.androidx.compose.bom))
    androidTestImplementation(libs.androidx.compose.ui.test.junit4)
    debugImplementation(libs.androidx.compose.ui.tooling)
    debugImplementation(libs.androidx.compose.ui.test.manifest)

    implementation("androidx.navigation:navigation-compose:2.8.0")
    implementation(libs.kotlinx.serialization.json)
    implementation(libs.androidx.lifecycle.viewmodel.compose)

    implementation("androidx.camera:camera-core:1.5.2")
    implementation("androidx.camera:camera-camera2:1.5.2")
    implementation("androidx.camera:camera-lifecycle:1.5.2")
    implementation("androidx.camera:camera-video:1.5.2")
    implementation("androidx.camera:camera-compose:1.5.2")
    implementation("androidx.camera:camera-view:1.5.2")

    implementation(libs.mpandroidchart)

    implementation(project(":opencv"))
}