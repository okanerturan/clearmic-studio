#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
APP_DIR="$ROOT/ClearMic Studio.app"
CONTENTS="$APP_DIR/Contents"
MACOS="$CONTENTS/MacOS"
RESOURCES="$CONTENTS/Resources"
DRIVER_DIR="$RESOURCES/ClearMic.driver"

# Auto-detect Developer ID Application signing identity (or override via CLEARMIC_SIGN_ID)
if [ -n "${CLEARMIC_SIGN_ID:-}" ]; then
    SIGN_ID="$CLEARMIC_SIGN_ID"
else
    SIGN_ID="$(security find-identity -v -p codesigning | sed -n 's/.*"Developer ID Application: \([^"]*\)".*/Developer ID Application: \1/p' | head -n 1 || true)"
    [ -n "$SIGN_ID" ] || SIGN_ID="-"
fi

# Ensure native C++ Accelerate vDSP / NEON DSP library is built
if [ ! -f "$ROOT/engine/libnative_dsp.dylib" ]; then
    echo "[build] 0/4 Compiling native Apple Accelerate vDSP + NEON DSP library..."
    clang++ -std=c++20 -O3 -mcpu=native -ffast-math -shared -fPIC \
      -framework Accelerate "$ROOT/engine/native_dsp.cpp" \
      -o "$ROOT/engine/libnative_dsp.dylib"
fi

rm -rf "$APP_DIR"
mkdir -p "$MACOS" "$RESOURCES" "$DRIVER_DIR/Contents/MacOS"

echo "[build] 1/4 Compiling native CoreAudio HAL driver (ClearMic.driver)..."
clang -O3 -arch arm64 -bundle \
  -Wall -Wextra \
  "$ROOT/macos_app/Driver/ClearMicDriver.c" \
  -framework CoreAudio \
  -framework CoreFoundation \
  -o "$DRIVER_DIR/Contents/MacOS/ClearMic"
cp "$ROOT/macos_app/Driver/Info.plist" "$DRIVER_DIR/Contents/Info.plist"

if [ "$SIGN_ID" != "-" ]; then
    codesign --force --sign "$SIGN_ID" --timestamp "$DRIVER_DIR"
else
    codesign --force --sign - "$DRIVER_DIR"
fi

echo "[build] 2/4 Compiling native SwiftUI + CoreAudio ClearMic Studio binary..."
swiftc \
  -O \
  -parse-as-library \
  -target arm64-apple-macos14.0 \
  "$ROOT"/macos_app/Sources/*.swift \
  -framework AppKit \
  -framework SwiftUI \
  -framework AVFoundation \
  -framework CoreAudio \
  -framework AudioToolbox \
  -o "$MACOS/ClearMicStudio"

echo "[build] 3/4 Generating crisp macOS app icon (.icns)..."
ICONSET="/tmp/ClearMicIcon.iconset"
rm -rf "$ICONSET"
mkdir -p "$ICONSET"
swift - <<'SWIFT'
import AppKit

func renderIcon(size: Int, outPath: String) {
    let s = CGFloat(size)
    let img = NSImage(size: NSSize(width: s, height: s))
    img.lockFocus()
    let rect = NSRect(x: 0, y: 0, width: s, height: s)
    let path = NSBezierPath(roundedRect: rect.insetBy(dx: s*0.06, dy: s*0.06), xRadius: s*0.22, yRadius: s*0.22)
    let grad = NSGradient(colors: [
        NSColor(srgbRed: 0.094, green: 0.098, blue: 0.110, alpha: 1.0),
        NSColor(srgbRed: 0.059, green: 0.063, blue: 0.071, alpha: 1.0),
    ])!
    grad.draw(in: path, angle: -90)

    NSColor(white: 1.0, alpha: 0.12).setStroke()
    path.lineWidth = max(1, s * 0.008)
    path.stroke()

    let heights: [CGFloat] = [0.24, 0.48, 0.68, 0.42, 0.22]
    let barW = s * 0.085
    let gap = s * 0.055
    let totalW = CGFloat(heights.count) * barW + CGFloat(heights.count - 1) * gap
    let startX = (s - totalW) / 2.0
    NSColor(srgbRed: 0.169, green: 0.831, blue: 0.490, alpha: 1.0).setFill()
    for (i, hFrac) in heights.enumerated() {
        let h = s * hFrac
        let x = startX + CGFloat(i) * (barW + gap)
        let y = (s - h) / 2.0
        let bp = NSBezierPath(roundedRect: NSRect(x: x, y: y, width: barW, height: h), xRadius: barW/2, yRadius: barW/2)
        bp.fill()
    }
    img.unlockFocus()
    if let tiff = img.tiffRepresentation,
       let rep = NSBitmapImageRep(data: tiff),
       let png = rep.representation(using: .png, properties: [:]) {
        try? png.write(to: URL(fileURLWithPath: outPath))
    }
}

let sizes = [(16,"16x16"), (32,"16x16@2x"), (32,"32x32"), (64,"32x32@2x"), (128,"128x128"), (256,"128x128@2x"), (256,"256x256"), (512,"256x256@2x"), (512,"512x512"), (1024,"512x512@2x")]
for (px, name) in sizes {
    renderIcon(size: px, outPath: "/tmp/ClearMicIcon.iconset/icon_\(name).png")
}
SWIFT
iconutil -c icns "$ICONSET" -o "$RESOURCES/AppIcon.icns"
rm -rf "$ICONSET"

cat > "$CONTENTS/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>ClearMic Studio</string>
    <key>CFBundleDisplayName</key>
    <string>ClearMic Studio</string>
    <key>CFBundleIdentifier</key>
    <string>com.okanerturan.clearmic-studio</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundleExecutable</key>
    <string>ClearMicStudio</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>LSMinimumSystemVersion</key>
    <string>14.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSMicrophoneUsageDescription</key>
    <string>ClearMic Studio captures your microphone to isolate your voice and remove background noise and other speakers in real time.</string>
    <key>NSAppleEventsUsageDescription</key>
    <string>ClearMic Studio uses macOS administrator authorization to install the bundled ClearMic Virtual Mic CoreAudio driver.</string>
</dict>
</plist>
PLIST

echo "[build] 4/4 Signing with: $SIGN_ID"
if [ "$SIGN_ID" != "-" ]; then
    codesign --force --sign "$SIGN_ID" --timestamp --options runtime \
      --entitlements "$ROOT/macos_app/entitlements.plist" "$APP_DIR"
else
    codesign --force --deep --sign - "$APP_DIR"
fi

codesign --verify --deep --strict --verbose=1 "$APP_DIR"
codesign --verify --strict --verbose=1 "$DRIVER_DIR"
echo "[build] SUCCESS -> $APP_DIR"
ls -lh "$MACOS/ClearMicStudio" "$DRIVER_DIR/Contents/MacOS/ClearMic" "$RESOURCES/AppIcon.icns"
