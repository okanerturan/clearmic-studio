import AppKit
import AVFoundation
import SwiftUI

/// Native macOS setup & permissions checklist shown on first run (or via "Setup" in the header).
/// Follows the exact Palette, Space, Radius, Typo and button pair from Theme.swift.
/// Wrapped in AudioDropSheetWrapper so Voice Memos.app file promises and Finder audio files
/// can be dropped anywhere on the sheet.
struct SetupSheet: View {
    @ObservedObject var devices: AudioDeviceManager
    @ObservedObject var installer: DriverInstaller
    @ObservedObject var daemon: DaemonClient
    @ObservedObject var bridge: CoreAudioBridge
    let hasInput: Bool
    let onRecordVoicePrint: () -> Void
    let onDropVoiceFile: (URL, String) -> Void
    let onChooseVoiceFile: () -> Void
    let onFinish: () -> Void

    @StateObject private var sheetDrop = AudioDropController()

    private var micGranted: Bool { installer.micAuthStatus == .authorized }
    private var micDenied: Bool { installer.micAuthStatus == .denied || installer.micAuthStatus == .restricted }
    private var driverReady: Bool { devices.hasClearMicDriver || devices.hasVirtualMic }
    private var coreReady: Bool { micGranted && driverReady }

    var body: some View {
        AudioDropSheetWrapper(controller: sheetDrop) {
            sheetContent
        }
        .onAppear {
            sheetDrop.onDropStarted = {
                installer.isImportingVoice = true
                installer.installError = nil
            }
            sheetDrop.onAudioFile = { url, label in
                onDropVoiceFile(url, label)
            }
            sheetDrop.onDropError = { msg in
                installer.isImportingVoice = false
                installer.installError = msg
            }
        }
    }

    private var sheetContent: some View {
        VStack(alignment: .leading, spacing: Space.s4) {
            // Header
            VStack(alignment: .leading, spacing: Space.s1) {
                Text("Set up ClearMic Studio")
                    .font(Typo.headline)
                    .foregroundStyle(Palette.ink)
                Text("Grant microphone access and install the signed virtual microphone once so Zoom, Google Meet and Discord can use your cleaned voice.")
                    .font(Typo.body)
                    .foregroundStyle(Palette.muted)
                    .fixedSize(horizontal: false, vertical: true)
            }

            if let err = installer.installError ?? daemon.lastError {
                ErrorNotice(message: err)
            }

            // Step 1: Microphone Permission
            stepRow(
                number: 1,
                title: "Microphone access",
                subtitle: micGranted
                    ? "ClearMic Studio can capture your microphone."
                    : (micDenied
                        ? "Turn on ClearMic Studio in System Settings → Privacy & Security → Microphone."
                        : "Required to hear your microphone before isolating your voice."),
                done: micGranted
            ) {
                if micGranted {
                    Button("Settings") { installer.openMicrophoneSettings() }
                        .buttonStyle(LinkButtonStyle())
                } else if micDenied {
                    Button("Open Privacy Settings") { installer.openMicrophoneSettings() }
                        .buttonStyle(ActionButtonStyle(.secondary))
                } else {
                    Button("Allow microphone") { installer.requestMicAccess() }
                        .buttonStyle(ActionButtonStyle(.primary))
                }
            }

            // Step 2: CoreAudio Virtual Microphone Driver
            stepRow(
                number: 2,
                title: "ClearMic virtual microphone",
                subtitle: driverReady
                    ? "Installed and active in CoreAudio. Pick it as your microphone in calls."
                    : "Installs the signed ClearMic.driver into CoreAudio. macOS asks for Touch ID or password once.",
                done: driverReady
            ) {
                if installer.isInstalling {
                    HStack(spacing: Space.s2) {
                        ProgressView().controlSize(.small)
                        Text("Installing").font(Typo.strong)
                    }
                    .frame(height: 32)
                } else if driverReady {
                    HStack(spacing: Space.s3) {
                        Button("Sound Settings") { installer.openSoundSettings() }
                            .buttonStyle(LinkButtonStyle())
                        if installer.isDriverFileInstalled {
                            Button("Uninstall") {
                                installer.uninstallDriver { _ in
                                    devices.refresh()
                                }
                            }
                            .buttonStyle(LinkButtonStyle())
                        }
                    }
                } else {
                    Button("Install virtual mic") {
                        installer.installBundledDriver { ok in
                            if ok {
                                devices.refresh()
                                DispatchQueue.main.asyncAfter(deadline: .now() + 0.8) {
                                    devices.refresh()
                                }
                            }
                        }
                    }
                    .buttonStyle(ActionButtonStyle(micGranted ? .primary : .secondary))
                }
            }

            // Step 3: Voice Print (Record, Choose File, or Drag & Drop Voice Memo)
            stepRow(
                number: 3,
                title: "Target voice print",
                subtitle: daemon.hasEnrollment
                    ? "\(daemon.enrollLabel) is loaded. Drop a Voice Memo anywhere here or record 5 seconds to replace it."
                    : "Record 5 seconds of your voice alone, choose a file, or drop a Voice Memo here.",
                done: daemon.hasEnrollment
            ) {
                if bridge.isRecordingEnroll {
                    HStack(spacing: Space.s2) {
                        ProgressView(value: bridge.enrollProgress).frame(width: 76)
                        Text("Recording").font(Typo.strong)
                    }
                    .frame(height: 32)
                } else if installer.isImportingVoice || daemon.isEnrolling {
                    HStack(spacing: Space.s2) {
                        ProgressView().controlSize(.small)
                        Text(installer.isImportingVoice ? "Importing" : "Enrolling").font(Typo.strong)
                    }
                    .frame(height: 32)
                } else {
                    HStack(spacing: Space.s2) {
                        Button("Choose file", action: onChooseVoiceFile)
                            .buttonStyle(LinkButtonStyle())
                        Button(daemon.hasEnrollment ? "Re-record 5s" : "Record 5s") {
                            onRecordVoicePrint()
                        }
                        .buttonStyle(ActionButtonStyle(.secondary))
                        .disabled(!daemon.isReady || !hasInput || !micGranted)
                    }
                }
            }

            Rectangle()
                .fill(Palette.line)
                .frame(height: 1)

            // Footer
            HStack(spacing: Space.s3) {
                Button("Open Sound Settings") { installer.openSoundSettings() }
                    .buttonStyle(LinkButtonStyle())
                Spacer()
                Button(coreReady ? "Done" : "Skip for now") {
                    onFinish()
                }
                .buttonStyle(ActionButtonStyle(coreReady ? .primary : .secondary))
            }
        }
        .padding(Space.s5)
        .frame(width: 560)
        .background(Palette.ground)
        .foregroundStyle(Palette.ink)
        .tint(Palette.accent)
        .preferredColorScheme(.dark)
        .overlay {
            if sheetDrop.isDraggingAudio {
                AudioDropOverlay()
            }
        }
        .onAppear {
            installer.refreshMicStatus()
            devices.refresh()
        }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
            installer.refreshMicStatus()
            devices.refresh()
        }
    }

    private func stepRow<Action: View>(
        number: Int,
        title: String,
        subtitle: String,
        done: Bool,
        @ViewBuilder action: () -> Action
    ) -> some View {
        let shape = RoundedRectangle(cornerRadius: Radius.tile, style: .continuous)

        return HStack(alignment: .center, spacing: Space.s3) {
            ZStack {
                Circle()
                    .fill(done ? Palette.accent.opacity(0.16) : Palette.fill)
                    .frame(width: 28, height: 28)
                if done {
                    Image(systemName: "checkmark")
                        .font(.system(size: 12, weight: .bold))
                        .foregroundStyle(Palette.accent)
                } else {
                    Text("\(number)")
                        .font(Typo.figure)
                        .foregroundStyle(Palette.muted)
                }
            }

            VStack(alignment: .leading, spacing: 2) {
                Text(title).font(Typo.title).foregroundStyle(Palette.ink)
                Text(subtitle)
                    .font(Typo.caption)
                    .foregroundStyle(Palette.muted)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Spacer(minLength: Space.s3)
            action()
        }
        .padding(Space.s3)
        .background(shape.fill(Palette.surface))
        .overlay(shape.strokeBorder(Palette.line, lineWidth: 1))
    }
}
