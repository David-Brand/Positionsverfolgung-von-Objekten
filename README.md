# Tracking Plot

> A fast and lightweight Android tool for tracking coloured objects with the camera and plotting their motion.

---

## Features

- Tracks up to 3 distinctly coloured objects (colour thresholding in native OpenCV code)
- Live plot of each object's x and y position over time, with adjustable time window
- Optional reference length (ruler) so positions are plotted and exported in real-world units (m, cm, mm)
- Optional region of interest to restrict tracking to part of the image
- Adjustable sampling rate (1–60 samples per second) and colour tolerance per tracker
- Experiments are stored on the device and can be reviewed, exported as CSV and deleted later
- Works in portrait and landscape; can be remote controlled from a computer with scrcpy (see below)

## How an experiment works

1. **Reference length** – tap two points on an object of known size and enter its length. Pinch to zoom the camera
   first if needed; the zoom is stored with the experiment. This step can be skipped (positions are then plotted as
   fractions of the image size).
2. **Region of interest** – optionally tap two corners of the area in which the objects move.
3. **Trackers** – for every object tap *Pick colour*, tap the object in the frozen frame and confirm. The tracked
   objects are outlined live; adjust the tolerance slider until only the object is detected. Then *Start experiment*,
   choose the sampling rate and an optional description.
4. **Live experiment** – *Start* / *Stop* recording as often as needed (the time axis keeps running); *Finish* opens
   the results. Leaving the screen keeps everything recorded so far.
5. **Results** – review the plot, export a CSV file or delete the experiment.

Coordinates: all positions refer to the camera frame in portrait orientation, x to the right and y downwards
(image convention), normalised to 0..1. With a reference length both axes are converted to the chosen unit.

### CSV format

```
trackerIndex,timestamp,centroidX,centroidY[,x_<unit>,y_<unit>]
```

`timestamp` is in milliseconds since the experiment was started, `centroidX`/`centroidY` are normalised
(0..1) image coordinates of the object's centroid, and the last two columns (only present when a reference length
was set) are the same positions in real-world units. Rows are ordered by tracker, then time.

---

## Installation (android device)

#### Option 1 — Download APK file on android device and tap to install
#### Option 2 — Download APK file to a USB stick, plug in and tap to install
#### Option 3 — Install via ADB
1. Download APK file to computer.
2. Enable USB Debugging on android device (see below).
3. Connect computer to android device via USB cable.
4. Execute:
```bash
adb install /path/to/app-file.apk
```
or if multiple android devices are connected:
```bash
adb devices -l # find your device id in the first column
adb -s YOURDEVICEID install /path/to/app-file.apk
```

The app requires a 64-bit device (arm64-v8a) running Android 7.0 or newer; it also runs on x86_64 emulators.
Devices with 16 KB memory pages (e.g. Pixel 8/9 in 16 KB mode) are supported.

## Enable USB Debugging (android device)
1. Open **Settings**
2. Go to **About Phone**
3. Tap **Build Number** multiple times until **Developer Options** are enabled
4. Go back to **Settings** then **System**&rarr;**Developer Options**
5. Enable **USB Debugging**

When you connect the android device to the computer, accept the **Allow USB Debugging** prompt.

## Remote control device

### Android device preparation
- APK must be installed on android device
- **USB Debugging** must be enabled

### macOS

#### Setup:
**scrcpy** and **android-platform-tools** are required.
Using homebrew for installation is recommend.
Otherwise please seek the respective projects installation guides for help.
```bash
brew install scrcpy
# if you are asked to install android-platform-tools do so:
brew install android-platform-tools
```
You can test the installation with:
```bash
adb --version
scrcpy --version
```
#### Remote control device:
- On your device in **Developer Options** enable **Wireless debugging**
- On your mac execute the following using the information provided by the android device:
```bash
adb pair <pairing-ip>:<pairing-port> # then enter the pairing code
adb connect <ip>:<port>
adb devices # OPTIONAL. checks if successful
scrcpy
```

### Windows

#### Setup:
**scrcpy** and **android-platform-tools** are required.
Using winget (built into Windows 10/11) for installation is recommended.
Otherwise please seek the respective projects installation guides for help.
```powershell
winget install Genymobile.scrcpy
winget install Google.PlatformTools
```
You can test the installation with (open a new terminal after installing):
```powershell
adb --version
scrcpy --version
```
#### Remote control device:
- On your device in **Developer Options** enable **Wireless debugging**
- On your PC execute the following using the information provided by the android device:
```powershell
adb pair <pairing-ip>:<pairing-port> # then enter the pairing code
adb connect <ip>:<port>
adb devices # OPTIONAL. checks if successful
scrcpy
```

## Remote app usage
- Scaling a plots x and y axis can be done by holding **ctrl** and then **clicking** and **dragging** horizontally or vertically respectively.
- Sliding a plots viewport can be done by **dragging** horizontally or vertically.

---

# Building from source

Requirements:

- Android Studio (or the Android SDK command line tools) with SDK platform 36, CMake 3.22.1 and **NDK 28.2.13676358**
  (installed automatically by Android Studio when the project is synced)
- JDK 17 (Android Studio's bundled JetBrains Runtime is fine)
- The **OpenCV Android SDK 4.12.0** (https://opencv.org/releases/), unpacked somewhere on disk

Tell the build where OpenCV is with one of:

- `opencvsdk=/path/to/OpenCV-android-sdk` in `gradle.properties` (or `~/.gradle/gradle.properties`)
- the environment variable `OPENCV_ANDROID_SDK`
- or simply unpack it to `~/Downloads/OpenCV-android-sdk` (the default)

Then:

```bash
./gradlew :app:assembleDebug        # APKs in app/build/outputs/apk/debug/
./gradlew :app:testDebugUnitTest    # unit tests
```

The native library is built with 16 KB page alignment (required for current Android devices); this can be
verified with `llvm-readelf -lW libnative-lib.so` (all `LOAD` segments must show `Align 0x4000`).

---

# License

This project is licensed under the **GPLv3 License**.

See `LICENSE` for details.

---

# Support

If you encounter issues:

- Open an issue on GitHub
- Provide logs and steps to reproduce
