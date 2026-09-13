pluginManagement {
    repositories {
        google {
            content {
                includeGroupByRegex("com\\.android.*")
                includeGroupByRegex("com\\.google.*")
                includeGroupByRegex("androidx.*")
            }
        }
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        maven { url = uri("https://jitpack.io") }
    }
}

rootProject.name = "TrackingPlot"
include(":app")

// The OpenCV Android SDK is attached as the ":opencv" module. Its location is resolved from
// (in order) the `opencvsdk` Gradle property, the OPENCV_ANDROID_SDK environment variable and
// finally ~/Downloads/OpenCV-android-sdk. app/build.gradle.kts applies the same rule.
val opencvSdkDir: File = run {
    val fromProperty = providers.gradleProperty("opencvsdk").orNull
        ?.takeIf { it.isNotBlank() && !it.startsWith("/path/to") }
    val fromEnv = System.getenv("OPENCV_ANDROID_SDK")?.takeIf { it.isNotBlank() }
    File(fromProperty ?: fromEnv ?: File(System.getProperty("user.home"), "Downloads/OpenCV-android-sdk").path)
}
include(":opencv")
project(":opencv").projectDir = File(opencvSdkDir, "sdk")
