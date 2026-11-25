import AppKit
import AVFoundation
import Combine
import CoreAudio
import SwiftUI

struct PresetOption: Identifiable {
    let id: String
    let title: String
    let latency: String
    let summary: String
    /// What the engine runs for this mode, shown as the tile's tooltip.
    let detail: String
}

let kPresets: [PresetOption] = [
    PresetOption(
        id: "studio_call",
        title: "Studio call",
        latency: "95 ms",
        summary: "Strongest removal of other voices.",
        detail: "Three VoxLock passes with look-ahead."
    ),
    PresetOption(
        id: "ultra_low",
        title: "Low delay",
        latency: "74 ms",
        summary: "Shortest delay with voice lock.",
        detail: "Two VoxLock passes, no look-ahead."
    ),
    PresetOption(
        id: "natural",
        title: "Natural",
        latency: "95 ms",
        summary: "Lightest touch on your voice.",
        detail: "One VoxLock pass with look-ahead."
    ),
    PresetOption(
        id: "denoise_only",
        title: "Noise removal",
        latency: "62 ms",
        summary: "Removes noise, keeps every voice.",
        detail: "Inworld 48 kHz denoiser only, no VoxLock."
    ),
]

struct ContentView: View {
    @ObservedObject var devices: AudioDeviceManager
    @ObservedObject var installer: DriverInstaller
    @ObservedObject var dropController: AudioDropController
    // Neither of these is observed here. While live the bridge publishes meters 30 times a
    // second and the daemon publishes stats 20 times a second; the views that draw those
    // values observe them, and this view mirrors only the few that change with a click or
    // an engine event, so the pickers, sliders and tiles are not rebuilt on every meter tick.
    let daemon: DaemonClient
    let bridge: CoreAudioBridge

    init(
        devices: AudioDeviceManager,
        installer: DriverInstaller,
        daemon: DaemonClient,
        bridge: CoreAudioBridge,
        dropController: AudioDropController
    ) {
        self.devices = devices
        self.installer = installer
        self.daemon = daemon
        self.bridge = bridge
        self.dropController = dropController
    }

    @State private var selectedInputID: AudioDeviceID = 0
    @State private var selectedOutputID: AudioDeviceID = 0
    @State private var preset: String = "studio_call"
    @State private var studioEQ: Bool = true
    @State private var speechGate: Bool = true
    @State private var bypass: Bool = false
    @State private var gateGamma: Double = 1.6
    @State private var outputGain: Double = 1.0

    @State private var isRunning: Bool = false
    @State private var engineReady: Bool = false
    @State private var hasSocket: Bool = false
    @State private var errorText: String? = nil
    @State private var showSetupSheet: Bool = false
    @State private var checkedFirstLaunchSetup: Bool = false

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            headerSection
            if installer.micAuthStatus == .denied || installer.micAuthStatus == .restricted {
                micPermissionNotice
            }
            if let errorText = installer.installError ?? errorText {
                ErrorNotice(message: errorText)
            }
            SignalPanel(
                bridge: bridge,
                daemon: daemon,
                bypassed: isRunning && bypass,
                idleDelay: selectedPreset.latency
            )
            devicesSection
            modeSection
            toneSection
            VoicePrintSection(
                daemon: daemon,
                bridge: bridge,
                installer: installer,
                hasInput: selectedInputID != 0,
                onRecord: startEnroll,
                onChooseFile: chooseVoiceFile
            )
        }
        .padding(.horizontal, Space.s5)
        .padding(.top, Space.s2)
        .padding(.bottom, Space.s5)
        .frame(minWidth: 720)
        .background(Palette.ground)
        .foregroundStyle(Palette.ink)
        .tint(Palette.accent)
        .preferredColorScheme(.dark)
        .overlay {
            if dropController.isDraggingAudio {
                AudioDropOverlay()
            }
        }
        .sheet(isPresented: $showSetupSheet) {
            SetupSheet(
                devices: devices,
                installer: installer,
                daemon: daemon,
                bridge: bridge,
                hasInput: selectedInputID != 0,
                onRecordVoicePrint: startEnroll,
                onDropVoiceFile: handleDroppedVoiceFile,
                onChooseVoiceFile: chooseVoiceFile,
                onFinish: { showSetupSheet = false }
            )
        }
        .onAppear {
            installer.refreshMicStatus()
            selectDefaultDevices()
            daemon.start()
            dropController.onDropStarted = {
                installer.isImportingVoice = true
                installer.installError = nil
            }
            dropController.onAudioFile = { url, label in
                handleDroppedVoiceFile(url, label: label)
            }
            dropController.onDropError = { msg in
                installer.isImportingVoice = false
                installer.installError = msg
            }
            if !checkedFirstLaunchSetup {
                checkedFirstLaunchSetup = true
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.35) {
                    if installer.micAuthStatus != .authorized || !devices.hasVirtualMic {
                        showSetupSheet = true
                    }
                }
            }
        }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
            installer.refreshMicStatus()
            devices.refresh()
        }
        .onChange(of: devices.inputDevices) { selectDefaultDevices() }
        .onChange(of: devices.outputDevices) { selectDefaultDevices() }
        .onReceive(bridge.$isRunning) { isRunning = $0 }
        .onReceive(daemon.$isReady) { engineReady = $0 }
        .onReceive(daemon.$socketPath) { hasSocket = ($0 != nil) }
        .onReceive(bridge.$errorMessage.combineLatest(daemon.$lastError)) { errorText = $0 ?? $1 }
    }

    private var selectedPreset: PresetOption {
        kPresets.first { $0.id == preset } ?? kPresets[0]
    }

    private var selectedOutput: AudioDeviceInfo? {
        devices.outputDevices.first { $0.id == selectedOutputID }
    }

    // MARK: - Header

    private var headerSection: some View {
        HStack(alignment: .center, spacing: Space.s4) {
            StatusHeader(
                daemon: daemon,
                isRunning: isRunning,
                bypassed: bypass,
                destination: selectedOutput?.name ?? "the output"
            )

            Spacer(minLength: Space.s4)

            if !isRunning {
                Button(action: { showSetupSheet = true }) {
                    Label("Setup", systemImage: "checkmark.shield")
                }
                .buttonStyle(LinkButtonStyle())
                .help("Check microphone permissions, virtual mic driver and voice print.")
            }

            if isRunning {
                Toggle(isOn: $bypass) {
                    Text("Bypass").font(Typo.strong)
                }
                .toggleStyle(.switch)
                .controlSize(.small)
                .tint(Palette.warning)
                .help("Send the microphone through without processing.")
                .onChange(of: bypass) { pushConfig() }
            }

            Button(action: toggleLive) {
                Label {
                    Text(isRunning ? "Stop live mic" : "Start live mic")
                } icon: {
                    if isRunning {
                        Image(systemName: "stop.fill").foregroundStyle(Palette.danger)
                    } else {
                        Image(systemName: "mic.fill")
                    }
                }
            }
            .buttonStyle(ActionButtonStyle(isRunning ? .secondary : .primary))
            .disabled(!engineReady || !hasSocket || selectedInputID == 0 || selectedOutputID == 0)
        }
    }

    // MARK: - Devices

    private var devicesSection: some View {
        VStack(alignment: .leading, spacing: Space.s2) {
            HStack(alignment: .bottom, spacing: Space.s3) {
                VStack(alignment: .leading, spacing: Space.s2) {
                    SectionLabel("Microphone")
                    DeviceMenu(
                        label: "Microphone",
                        items: devices.inputDevices.map {
                            DeviceMenu.Item(id: $0.id, title: "\($0.name) · \(String(format: "%g", $0.sampleRate / 1000)) kHz")
                        },
                        selection: $selectedInputID
                    )
                    .disabled(isRunning)
                }
                .frame(maxWidth: .infinity, alignment: .leading)

                Image(systemName: "arrow.right")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(Palette.muted)
                    .frame(height: 22)
                    .accessibilityHidden(true)

                VStack(alignment: .leading, spacing: Space.s2) {
                    HStack(spacing: Space.s3) {
                        SectionLabel("Send to")
                        Spacer()
                        Button(action: { installer.openSoundSettings() }) {
                            Label("Sound Settings", systemImage: "gearshape")
                        }
                        .buttonStyle(LinkButtonStyle())
                        Button(action: { devices.refresh() }) {
                            Label("Refresh", systemImage: "arrow.clockwise")
                        }
                        .buttonStyle(LinkButtonStyle())
                    }
                    DeviceMenu(
                        label: "Send to",
                        items: devices.outputDevices.map {
                            DeviceMenu.Item(id: $0.id, title: $0.isVirtualMic ? "\($0.name) · for calls" : $0.name)
                        },
                        selection: $selectedOutputID
                    )
                    .disabled(isRunning)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }

            if devices.hasVirtualMic {
                Text(callNote)
                    .font(Typo.caption)
                    .foregroundStyle(Palette.muted)
                    .fixedSize(horizontal: false, vertical: true)
            } else {
                installNotice
            }
        }
    }

    private var callNote: String {
        if let out = selectedOutput, out.isVirtualMic {
            return "In Zoom or Google Meet, choose \(out.name) as the microphone."
        }
        let target = devices.outputDevices.first(where: { $0.isVirtualMic })?.name ?? "ClearMic Virtual Mic"
        return "To use ClearMic in a call, send to \(target) and choose it as the microphone in Zoom or Google Meet."
    }

    private var micPermissionNotice: some View {
        Notice(symbol: "mic.slash.fill", tone: Palette.warning) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Microphone access is turned off for ClearMic Studio.")
                    .font(Typo.strong)
                Text("Turn on ClearMic Studio under Privacy & Security → Microphone, then return here.")
                    .font(Typo.caption)
                    .foregroundStyle(Palette.muted)
            }
        } action: {
            Button("Open Microphone Settings") {
                installer.openMicrophoneSettings()
            }
            .buttonStyle(ActionButtonStyle(.secondary))
        }
    }

    private var installNotice: some View {
        Notice(symbol: "waveform.badge.plus", tone: Palette.warning) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Install the ClearMic virtual microphone to use it in Zoom and Google Meet.")
                    .font(Typo.strong)
                Text("Signed CoreAudio driver bundled with this app. macOS will ask for your password once.")
                    .font(Typo.caption)
                    .foregroundStyle(Palette.muted)
            }
        } action: {
            if installer.isInstalling {
                HStack(spacing: Space.s2) {
                    ProgressView().controlSize(.small)
                    Text("Installing").font(Typo.strong)
                }
                .frame(height: 32)
            } else {
                Button("Install virtual mic") {
                    installer.installBundledDriver { ok in
                        if ok {
                            devices.refresh()
                            DispatchQueue.main.asyncAfter(deadline: .now() + 0.8) {
                                devices.refresh()
                                selectDefaultDevices()
                            }
                        }
                    }
                }
                .buttonStyle(ActionButtonStyle(.secondary))
            }
        }
    }

    // MARK: - Mode

    private var modeSection: some View {
        VStack(alignment: .leading, spacing: Space.s2) {
            SectionLabel("Mode")

            HStack(alignment: .top, spacing: Space.s2) {
                ForEach(kPresets) { opt in
                    let active = (preset == opt.id)
                    Button(action: {
                        preset = opt.id
                        pushConfig()
                    }) {
                        VStack(alignment: .leading, spacing: Space.s1) {
                            HStack(alignment: .firstTextBaseline, spacing: Space.s1) {
                                Text(opt.title)
                                    .font(Typo.title)
                                    .foregroundStyle(Palette.ink)
                                Spacer(minLength: Space.s1)
                                Text(opt.latency)
                                    .font(Typo.figure)
                                    .foregroundStyle(active ? Palette.ink : Palette.muted)
                            }
                            Text(opt.summary)
                                .font(Typo.caption)
                                .foregroundStyle(Palette.muted)
                                .multilineTextAlignment(.leading)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                        .padding(Space.s3)
                        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                    }
                    .buttonStyle(TileButtonStyle(selected: active))
                    .help(opt.detail)
                    .accessibilityLabel("\(opt.title), \(opt.latency) delay. \(opt.summary)")
                    .accessibilityAddTraits(active ? .isSelected : [])
                }
            }
            .fixedSize(horizontal: false, vertical: true)
        }
    }

    // MARK: - Tone

    private var toneSection: some View {
        VStack(alignment: .leading, spacing: Space.s3) {
            SectionLabel("Tone")

            HStack(alignment: .center, spacing: Space.s5) {
                Toggle(isOn: $studioEQ) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Studio EQ").font(Typo.strong)
                        Text("Warmth, clarity and air. No added delay.")
                            .font(Typo.caption)
                            .foregroundStyle(Palette.muted)
                    }
                    .fixedSize()
                }
                .toggleStyle(.switch)
                .controlSize(.small)
                .onChange(of: studioEQ) { pushConfig() }

                Toggle(isOn: $speechGate) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Speech gate").font(Typo.strong)
                        Text("Mutes mouth sounds, clicks and coughs.")
                            .font(Typo.caption)
                            .foregroundStyle(Palette.muted)
                    }
                    .fixedSize()
                }
                .toggleStyle(.switch)
                .controlSize(.small)
                .help("Uses Silero VAD (0.1 ms) to mute non-speech sounds like lip smacks, clicks, coughs and mic bumps.")
                .onChange(of: speechGate) { pushConfig() }

                Spacer(minLength: 0)
            }

            HStack(alignment: .center, spacing: Space.s5) {
                toneSlider(
                    "Highs gate",
                    value: stepped($gateGamma),
                    range: 1.0...2.4,
                    readout: String(format: "%.1f", gateGamma)
                )
                .help("Higher values pass less high-frequency sound that is not your voice.")
                .onChange(of: gateGamma) { pushConfig() }

                toneSlider(
                    "Output gain",
                    value: stepped($outputGain),
                    range: 0.5...2.0,
                    readout: String(format: "%.1f×", outputGain)
                )
                .onChange(of: outputGain) { pushConfig() }
            }
        }
    }

    private func toneSlider(
        _ title: String,
        value: Binding<Double>,
        range: ClosedRange<Double>,
        readout: String
    ) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(title).font(Typo.strong)
                Spacer()
                Text(readout)
                    .font(Typo.figure)
                    .foregroundStyle(Palette.muted)
            }
            Slider(value: value, in: range) {
                Text(title)
            }
            .labelsHidden()
            .controlSize(.small)
        }
        .frame(minWidth: 140, maxWidth: .infinity)
    }

    /// Snaps a slider to tenths without the row of tick marks a stepped slider draws.
    private func stepped(_ value: Binding<Double>) -> Binding<Double> {
        Binding(
            get: { value.wrappedValue },
            set: { value.wrappedValue = ($0 * 10).rounded() / 10 }
        )
    }

    // MARK: - Actions

    private func selectDefaultDevices() {
        if selectedInputID == 0 || !devices.inputDevices.contains(where: { $0.id == selectedInputID }) {
            if let defIn = devices.defaultInputDeviceID(), devices.inputDevices.contains(where: { $0.id == defIn }) {
                selectedInputID = defIn
            } else if let first = devices.inputDevices.first {
                selectedInputID = first.id
            }
        }
        if selectedOutputID == 0 || !devices.outputDevices.contains(where: { $0.id == selectedOutputID }) {
            if let vm = devices.outputDevices.first(where: { $0.isClearMic }) ?? devices.outputDevices.first(where: { $0.isVirtualMic }) {
                selectedOutputID = vm.id
            } else if let defOut = devices.defaultOutputDeviceID(), devices.outputDevices.contains(where: { $0.id == defOut }) {
                selectedOutputID = defOut
            } else if let first = devices.outputDevices.first {
                selectedOutputID = first.id
            }
        } else if let vm = devices.outputDevices.first(where: { $0.isClearMic }),
                  let cur = selectedOutput, !cur.isVirtualMic, !isRunning {
            // Auto-select ClearMic Virtual Mic right after the user finishes installing it
            selectedOutputID = vm.id
        }
    }

    private func toggleLive() {
        if bridge.isRunning {
            bridge.stop()
            daemon.resetStream()
        } else if let sock = daemon.socketPath {
            pushConfig()
            bridge.start(inputDeviceID: selectedInputID, outputDeviceID: selectedOutputID, socketPath: sock)
        }
    }

    private func pushConfig() {
        daemon.configure(
            preset: preset,
            studioEQ: studioEQ,
            speechGate: speechGate,
            gateGamma: gateGamma,
            outputGain: outputGain,
            bypass: bypass
        )
    }

    private func startEnroll() {
        bridge.recordEnrollment(inputDeviceID: selectedInputID, seconds: 5.0) { wavPath in
            if let p = wavPath {
                daemon.enroll(wavPath: p, label: "Your voice")
            }
        }
    }

    private func chooseVoiceFile() {
        presentAudioFilePicker { url in
            let label = url.deletingPathExtension().lastPathComponent
            handleDroppedVoiceFile(url, label: label)
        }
    }

    private func handleDroppedVoiceFile(_ url: URL, label: String) {
        installer.convertDroppedAudioToWav48k(from: url) { wavPath in
            if let p = wavPath {
                daemon.enroll(wavPath: p, label: label)
            }
        }
    }
}

// MARK: - Status

/// The one line that says what the app is doing right now and where the audio is going.
struct StatusHeader: View {
    @ObservedObject var daemon: DaemonClient
    let isRunning: Bool
    let bypassed: Bool
    let destination: String

    private enum Phase {
        case starting
        case stopped
        case ready
        case live
        case bypassed
    }

    private var phase: Phase {
        if isRunning { return bypassed ? .bypassed : .live }
        if daemon.isReady { return .ready }
        // DaemonClient reports a failed launch or an exit only through these two messages.
        let failed = daemon.statusMessage == "Launch failed" || daemon.statusMessage.hasPrefix("Engine exited")
        return failed ? .stopped : .starting
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: Space.s2) {
                indicator
                Text(headline).font(Typo.headline)
            }
            // The engine's own loading and exit lines arrive at runtime and can be long.
            Text(detail)
                .font(Typo.body)
                .foregroundStyle(Palette.muted)
                .lineLimit(2)
                .fixedSize(horizontal: false, vertical: true)
        }
        .accessibilityElement(children: .combine)
    }

    @ViewBuilder private var indicator: some View {
        if phase == .starting {
            ProgressView()
                .controlSize(.small)
                .frame(width: 16, height: 16)
        } else {
            Circle()
                .fill(indicatorColor)
                .frame(width: 9, height: 9)
                .frame(width: 16, height: 16)
        }
    }

    private var indicatorColor: Color {
        switch phase {
        case .live: return Palette.accent
        case .bypassed: return Palette.warning
        case .stopped: return Palette.danger
        case .ready, .starting: return Palette.muted
        }
    }

    private var headline: String {
        switch phase {
        case .starting: return "Starting engine"
        case .stopped: return "Engine stopped"
        case .ready: return "Ready"
        case .live: return "Live"
        case .bypassed: return "Bypassed"
        }
    }

    private var detail: String {
        switch phase {
        case .starting, .stopped: return daemon.statusMessage
        case .ready: return "Not sending audio."
        case .live: return "Sending your cleaned voice to \(destination)."
        case .bypassed: return "Sending your microphone to \(destination) unprocessed."
        }
    }
}

// MARK: - Errors

/// Shows an error from the audio bridge or the engine. A Python traceback can run to dozens of
/// lines, so the notice shows its last line, which names the exception, and copies the whole text.
struct ErrorNotice: View {
    let message: String

    @State private var copied = false

    private var lastLine: String {
        let lines = message.split(whereSeparator: \.isNewline)
        let line = lines.last.map(String.init) ?? message
        return line.trimmingCharacters(in: .whitespaces)
    }

    var body: some View {
        Notice(symbol: "exclamationmark.octagon.fill", tone: Palette.danger) {
            Text(lastLine)
                .font(Typo.body)
                .lineLimit(4)
                .textSelection(.enabled)
        } action: {
            Button(action: copyMessage) {
                SwapLabel(rest: "Copy error", done: "Copied", isDone: copied)
            }
            .buttonStyle(ActionButtonStyle(.secondary))
        }
    }

    private func copyMessage() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(message, forType: .string)
        copied = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 2) { copied = false }
    }
}

// MARK: - Voice print

struct VoicePrintSection: View {
    @ObservedObject var daemon: DaemonClient
    @ObservedObject var bridge: CoreAudioBridge
    @ObservedObject var installer: DriverInstaller
    let hasInput: Bool
    let onRecord: () -> Void
    let onChooseFile: () -> Void

    /// The engine names the voice print by its file; the two files it ships with get plain names.
    private var printName: String {
        guard daemon.hasEnrollment else { return "No voice print yet" }
        switch daemon.enrollLabel {
        case "00_clean_enroll_16k.wav", "Default Voice Print": return "Default voice print"
        case "user_enroll_16k.wav", "voice_print_16k.wav": return "Your saved voice print"
        default: return daemon.enrollLabel
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: Space.s2) {
            SectionLabel("Voice print")

            HStack(alignment: .center, spacing: Space.s3) {
                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 6) {
                        Image(systemName: daemon.hasEnrollment ? "person.wave.2.fill" : "person.badge.plus")
                            .font(.system(size: 12, weight: .medium))
                            .foregroundStyle(daemon.hasEnrollment ? Palette.accent : Palette.warning)
                            .accessibilityHidden(true)
                        Text(printName).font(Typo.strong)
                    }
                    Text("Speak alone for 5 seconds, or drop a Voice Memo anywhere on this window.")
                        .font(Typo.caption)
                        .foregroundStyle(Palette.muted)
                }
                .fixedSize(horizontal: false, vertical: true)

                Spacer(minLength: Space.s3)

                if bridge.isRecordingEnroll {
                    progress("Recording", value: bridge.enrollProgress)
                } else if installer.isImportingVoice || daemon.isEnrolling {
                    HStack(spacing: Space.s2) {
                        ProgressView().controlSize(.small)
                        Text(installer.isImportingVoice ? "Importing" : "Enrolling").font(Typo.strong)
                    }
                    .frame(height: 32)
                } else {
                    Button("Choose file", action: onChooseFile)
                        .buttonStyle(LinkButtonStyle())
                        .help("Import a Voice Memo or audio file (.m4a, .wav, .mp3) as your voice print.")

                    Button("Record voice print", action: onRecord)
                        .buttonStyle(ActionButtonStyle(.secondary))
                        .disabled(!daemon.isReady || !hasInput)
                }

                if bridge.isSoundCheckRecording {
                    progress("Capturing", value: bridge.soundCheckProgress)
                } else if bridge.isSoundCheckPlaying {
                    Label("Playing back", systemImage: "speaker.wave.2.fill")
                        .font(Typo.strong)
                        .foregroundStyle(Palette.accent)
                        .frame(height: 32)
                } else {
                    Button("Run sound check") { bridge.runSoundCheck(seconds: 5.0) }
                        .buttonStyle(ActionButtonStyle(.secondary))
                        .disabled(!bridge.isRunning)
                        .help("While live, records 5 seconds of the cleaned output and plays it back.")
                }
            }
        }
    }

    private func progress(_ title: String, value: Double) -> some View {
        HStack(spacing: Space.s2) {
            ProgressView(value: value)
                .frame(width: 88)
            Text(title).font(Typo.strong)
        }
        .frame(height: 32)
        .accessibilityElement(children: .combine)
    }
}
