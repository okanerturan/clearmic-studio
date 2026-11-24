#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
APP_DIR="$ROOT/ClearMic Studio.app"
DMG_OUT="$ROOT/dist/ClearMic-Studio.dmg"
STAGE="/tmp/clearmic_dmg_stage"

if [ ! -d "$APP_DIR" ]; then
    "$SCRIPT_DIR/build_app.sh"
fi

mkdir -p "$ROOT/dist"
rm -rf "$STAGE" "$DMG_OUT"
mkdir -p "$STAGE"
cp -R "$APP_DIR" "$STAGE/"
ln -s /Applications "$STAGE/Applications"

echo "[dmg] Creating drag-to-Applications disk image -> $DMG_OUT"
hdiutil create \
  -volname "ClearMic Studio" \
  -srcfolder "$STAGE" \
  -ov \
  -format UDZO \
  "$DMG_OUT"

rm -rf "$STAGE"

SIGN_ID="$(security find-identity -v -p codesigning | sed -n 's/.*"Developer ID Application: \([^"]*\)".*/Developer ID Application: \1/p' | head -n 1 || true)"
if [ -n "$SIGN_ID" ]; then
    codesign --force --sign "$SIGN_ID" --timestamp "$DMG_OUT"
fi
ls -lh "$DMG_OUT"
