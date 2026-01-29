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
        maven { url = uri("https://jitpack.io")  }
    }
}

rootProject.name = "TrackingPlot"
include(":app")

include(":opencv")
/*val opencvsdk: String by settings
project(":opencv").projectDir = File("$opencvsdk/sdk")*/
val home: String? = System.getProperty("user.home")
project(":opencv").projectDir = File(home, "Downloads/OpenCV-android-sdk/sdk")
