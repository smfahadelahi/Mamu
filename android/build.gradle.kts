// Top-level build file. Plugin versions live in gradle/libs.versions.toml.
buildscript {
    // Gradle resolves its own embedded Kotlin dependencies for this build script.
    repositories {
        google()
        mavenCentral()
    }
}

plugins {
    alias(libs.plugins.android.application) apply false
    alias(libs.plugins.kotlin.android) apply false
    alias(libs.plugins.kotlin.compose) apply false
}
