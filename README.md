# Tracking Plot

> A fast and lightweight tool for tracking and processing objects by color.

---

## Features

- Tracking of up to 3 distinctly colored objects
- Live plotting of each objects x and y coordinates
- Exporting of measurements
- Works on android devices (can be remote controlled from most platforms)

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

[//]: # (#### Option 1 — Homebrew &#40;Recommended&#41;)

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
adb connect
adb device # OPTIONAL. checks if successful
scrcpy
```

## Remote app usage
- Scaling a plots x and y axis can be done by holding **ctrl** and then **clicking** and **dragging** horizontally or vertically respectively.
- Sliding a plots viewport can be done by **dragging** horizontally or vertically.


[//]: # (#### Option 2 — Download Binary)

[//]: # ()
[//]: # (1. Go to the **Releases** page.)

[//]: # (2. Download the latest `macOS` archive.)

[//]: # (3. Extract it.)

[//]: # ()
[//]: # (```bash)

[//]: # (tar -xvf project-name-macos.tar.gz)

[//]: # (cd project-name)

[//]: # (./project-name)

[//]: # (```)

[//]: # ()
[//]: # (#### Option 3 — Build From Source)

[//]: # ()
[//]: # (See **Building from Source** below.)

---

[//]: # (### Windows)

[//]: # ()
[//]: # (#### Option 1 — Download Binary)

[//]: # ()
[//]: # (1. Go to the **Releases** page.)

[//]: # (2. Download the latest `Windows` zip file.)

[//]: # (3. Extract it.)

[//]: # (4. Run:)

[//]: # ()
[//]: # (```powershell)

[//]: # (project-name.exe)

[//]: # (```)

[//]: # ()
[//]: # (#### Option 2 — Using Scoop)

[//]: # ()
[//]: # (```powershell)

[//]: # (scoop install project-name)

[//]: # (```)

[//]: # ()
[//]: # (#### Option 3 — Build From Source)

[//]: # ()
[//]: # (See **Building from Source** below.)

[//]: # ()
[//]: # (---)

[//]: # ()
[//]: # (### Linux &#40;Optional&#41;)

[//]: # ()
[//]: # (```bash)

[//]: # (curl -L https://example.com/project-name-linux.tar.gz | tar xz)

[//]: # (cd project-name)

[//]: # (./project-name)

[//]: # (```)

[//]: # ()
[//]: # (Or install via your package manager.)

[//]: # ()
[//]: # (---)

[//]: # ()
[//]: # (# Building from Source)

[//]: # ()
[//]: # (## Requirements)

[//]: # ()
[//]: # (Install the following dependencies:)

[//]: # ()
[//]: # (- Git)

[//]: # (- Language runtime / compiler &#40;e.g. Go, Node.js, Rust, Python, etc.&#41;)

[//]: # (- Build tools &#40;make, cmake, etc.&#41;)

[//]: # ()
[//]: # (Example:)

[//]: # ()
[//]: # (```bash)

[//]: # (git clone https://github.com/username/project-name.git)

[//]: # (cd project-name)

[//]: # (```)

[//]: # ()
[//]: # (### Build)

[//]: # ()
[//]: # (```bash)

[//]: # (make build)

[//]: # (```)

[//]: # ()
[//]: # (or)

[//]: # ()
[//]: # (```bash)

[//]: # (npm install)

[//]: # (npm run build)

[//]: # (```)

[//]: # ()
[//]: # (or)

[//]: # ()
[//]: # (```bash)

[//]: # (cargo build --release)

[//]: # (```)

[//]: # ()
[//]: # (The compiled binary will appear in:)

[//]: # ()
[//]: # (```)

[//]: # (./build)

[//]: # (```)

[//]: # ()
[//]: # (or)

[//]: # ()
[//]: # (```)

[//]: # (./target/release)

[//]: # (```)

[//]: # ()
[//]: # (---)

[//]: # ()
[//]: # (# Usage)

[//]: # ()
[//]: # (Basic usage:)

[//]: # ()
[//]: # (```bash)

[//]: # (project-name [options])

[//]: # (```)

[//]: # ()
[//]: # (Example:)

[//]: # ()
[//]: # (```bash)

[//]: # (project-name input.txt --output result.txt)

[//]: # (```)

[//]: # ()
[//]: # (### Command Line Options)

[//]: # ()
[//]: # (| Option | Description |)

[//]: # (|------|-------------|)

[//]: # (| `-h`, `--help` | Show help |)

[//]: # (| `-v`, `--version` | Show version |)

[//]: # (| `-o` | Output file |)

[//]: # ()
[//]: # (Example:)

[//]: # ()
[//]: # (```bash)

[//]: # (project-name --help)

[//]: # (```)

[//]: # ()
[//]: # (---)

[//]: # ()
[//]: # (# Configuration)

[//]: # ()
[//]: # (You can optionally configure the program using a config file:)

[//]: # ()
[//]: # (```)

[//]: # (~/.config/project-name/config.yaml)

[//]: # (```)

[//]: # ()
[//]: # (Example:)

[//]: # ()
[//]: # (```yaml)

[//]: # (output_directory: ./output)

[//]: # (log_level: info)

[//]: # (```)

[//]: # ()
[//]: # (---)

[//]: # ()
[//]: # (# Development)

[//]: # ()
[//]: # (Clone the repository:)

[//]: # ()
[//]: # (```bash)

[//]: # (git clone https://github.com/username/project-name.git)

[//]: # (cd project-name)

[//]: # (```)

[//]: # ()
[//]: # (Install dependencies:)

[//]: # ()
[//]: # (```bash)

[//]: # (make install-dev)

[//]: # (```)

[//]: # ()
[//]: # (Run tests:)

[//]: # ()
[//]: # (```bash)

[//]: # (make test)

[//]: # (```)

[//]: # ()
[//]: # (---)

[//]: # ()
[//]: # (# Contributing)

[//]: # ()
[//]: # (Contributions are welcome!)

[//]: # ()
[//]: # (1. Fork the repository)

[//]: # (2. Create a new branch)

[//]: # ()
[//]: # (```bash)

[//]: # (git checkout -b feature/my-feature)

[//]: # (```)

[//]: # ()
[//]: # (3. Commit your changes)

[//]: # (4. Push the branch)

[//]: # (5. Open a Pull Request)

[//]: # ()
[//]: # (---)

# License

This project is licensed under the **GPLv3 License**.

See `LICENSE` for details.

---

# Support

If you encounter issues:

- Open an issue on GitHub
- Provide logs and steps to reproduce

---

[//]: # (# Acknowledgements)

[//]: # ()
[//]: # (- Library / framework used)

[//]: # (- Contributors)

[//]: # (- Inspiration)