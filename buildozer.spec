[app]

# (str) Title of your application
title = Bochra AI

# (str) Package name
package.name = bochra

# (str) Package domain (reverse DNS)
package.domain = org.bochra

# (str) Source code where main.py is located
source.dir = .

# (list) File extensions to package into the APK.
# gguf/onnx removed: large models bloat the APK. Download them to app storage at first run instead.
source.include_exts = py,pyc,png,jpg,kv,atlas,json,mp3,wav,ttf,ttc,db,txt,pdf,docx

# (list) Patterns to include (assets folder)
source.include_patterns = assets/*

# (str) Application versioning
version = 0.1
android.version_code = 1

# (str) Icon and presplash - remove these two lines if you don't have the files
icon.filename = assets/icon.png
presplash.filename = presplash.png
android.presplash_color = #000000

# (str) Supported orientation
orientation = portrait

# (bool) Fullscreen
fullscreen = 1

# (list) Python/p4a requirements
# llama-cpp-python needs a working recipe at ./recipes/llama-cpp-python/__init__.py
requirements = python3,hostpython3,kivy==2.3.0,pyjnius,android,plyer,openssl,sqlite3,requests,urllib3,idna,certifi,charset-normalizer,psutil,pypdf,pillow,arabic_reshaper,python-bidi,numpy,llama-cpp-python==0.3.16

# Local recipes folder (cross-compiles the inference engine for Android)
p4a.local_recipes = ./recipes

# python-for-android branch (develop is needed for custom CMake recipes on recent NDKs)
p4a.branch = develop

# (str) p4a bootstrap
p4a.bootstrap = sdl2

# (list) Supported architectures
android.archs = arm64-v8a

# Android API levels / NDK
android.api = 34
android.minapi = 24
android.ndk =25b

# Accept SDK licenses automatically (required for non-interactive builds)
android.accept_sdk_license = True

# Needed if your Java code uses AndroidX
android.enable_androidx = True

# (list) Android permissions
android.permissions = INTERNET, RECORD_AUDIO, WRITE_EXTERNAL_STORAGE, READ_EXTERNAL_STORAGE, MANAGE_EXTERNAL_STORAGE, FOREGROUND_SERVICE, FOREGROUND_SERVICE_SPECIAL_USE, SYSTEM_ALERT_WINDOW, VIBRATE, ACCESS_NETWORK_STATE, WAKE_LOCK, POST_NOTIFICATIONS

# Extra manifest XML (accessibility <service> entry goes here). File must exist.
android.extra_manifest_xml = android_extras.xml

# Java sources (e.g. your AccessibilityService class). Folder must exist.
android.add_src = src

# Background service: uncomment once service.py exists next to main.py
# services = Assistant:service.py:foreground

[buildozer]

# (int) log level (0 = error only, 1 = info, 2 = debug)
log_level = 2

# (int) warn when running as root
warn_on_root = 1
