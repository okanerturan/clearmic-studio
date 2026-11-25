import AppKit
import AVFoundation
import Foundation

/// Installs/uninstalls the bundled, Developer-ID-signed ClearMic.driver into
/// /Library/Audio/Plug-Ins/HAL/ via macOS's native Touch ID / Admin Password sheet,
/// converts dropped Voice Memos (.m4a/.wav/.mp3/.flac) via macOS afconvert, and
/// opens System Settings panes directly.
final class DriverInstaller: ObservableObject {
    @Published var isInstalling: Bool = false
    @Published var isImportingVoice: Bool = false
    @Published var installError: String? = nil
    @Published var micAuthStatus: AVAuthorizationStatus = AVCaptureDevice.authorizationStatus(for: .audio)

    private let halDir = "/Library/Audio/Plug-Ins/HAL"
    private let installedPath = "/Library/Audio/Plug-Ins/HAL/ClearMic.driver"

    var isDriverFileInstalled: Bool {
        FileManager.default.fileExists(atPath: installedPath)
    }

    var bundledDriverPath: String? {
        if let res = Bundle.main.resourceURL {
            let p = res.appendingPathComponent("ClearMic.driver").path
            if FileManager.default.fileExists(atPath: p) { return p }
        }
        return nil
    }

    func refreshMicStatus() {
        DispatchQueue.main.async {
            self.micAuthStatus = AVCaptureDevice.authorizationStatus(for: .audio)
        }
    }

    func requestMicAccess() {
        let status = AVCaptureDevice.authorizationStatus(for: .audio)
        if status == .notDetermined {
            AVCaptureDevice.requestAccess(for: .audio) { [weak self] _ in
                self?.refreshMicStatus()
            }
        } else if status == .denied || status == .restricted {
            openMicrophoneSettings()
        }
    }

    func installBundledDriver(completion: @escaping (Bool) -> Void) {
        guard let src = bundledDriverPath else {
            installError = "Bundled ClearMic.driver was not found inside the app."
            completion(false)
            return
        }
        let escapedSrc = src.replacingOccurrences(of: "'", with: "'\\''")
        let shellCmd = """
        mkdir -p '\(halDir)' && \
        rm -rf '\(installedPath)' && \
        cp -R '\(escapedSrc)' '\(installedPath)' && \
        chown -R root:wheel '\(installedPath)' && \
        chmod -R a+rX '\(installedPath)' && \
        killall -9 coreaudiod || true
        """
        runAdminScript(shellCmd, waitSeconds: 1.5, completion: completion)
    }

    func uninstallDriver(completion: @escaping (Bool) -> Void) {
        let shellCmd = """
        rm -rf '\(installedPath)' && \
        killall -9 coreaudiod || true
        """
        runAdminScript(shellCmd, waitSeconds: 1.2, completion: completion)
    }

    /// Converts any macOS-readable audio file (.m4a Voice Memo, .wav, .mp3, .aiff, .caf)
    /// into a clean 48 kHz mono float32 WAV for Voice Print enrollment.
    func convertDroppedAudioToWav48k(from url: URL, completion: @escaping (String?) -> Void) {
        DispatchQueue.main.async {
            self.isImportingVoice = true
            self.installError = nil
        }
        DispatchQueue.global(qos: .userInitiated).async {
            let resolved = (url as NSURL).filePathURL ?? url
            let accessed = resolved.startAccessingSecurityScopedResource()
            defer {
                if accessed { resolved.stopAccessingSecurityScopedResource() }
            }

            let outPath = NSTemporaryDirectory() + "clearmic_dropped_\(UUID().uuidString).wav"
            try? FileManager.default.removeItem(atPath: outPath)

            let proc = Process()
            proc.executableURL = URL(fileURLWithPath: "/usr/bin/afconvert")
            proc.arguments = ["-f", "WAVE", "-d", "LEF32@48000", "-c", "1", resolved.path, outPath]
            let errPipe = Pipe()
            proc.standardError = errPipe
            proc.standardOutput = errPipe

            do {
                try proc.run()
                proc.waitUntilExit()
                let errData = errPipe.fileHandleForReading.readDataToEndOfFile()
                let errMsg = String(data: errData, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                let ok = (proc.terminationStatus == 0) && FileManager.default.fileExists(atPath: outPath)
                DispatchQueue.main.async {
                    self.isImportingVoice = false
                    if !ok {
                        let detail = errMsg.isEmpty ? "exit \(proc.terminationStatus)" : errMsg
                        self.installError = "Could not decode \(resolved.lastPathComponent): \(detail)"
                    }
                    completion(ok ? outPath : nil)
                }
            } catch {
                DispatchQueue.main.async {
                    self.isImportingVoice = false
                    self.installError = "Could not decode \(resolved.lastPathComponent): \(error.localizedDescription)"
                    completion(nil)
                }
            }
        }
    }

    private func runAdminScript(_ shellCmd: String, waitSeconds: Double, completion: @escaping (Bool) -> Void) {
        isInstalling = true
        installError = nil
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            let escapedForAS = shellCmd
                .replacingOccurrences(of: "\\", with: "\\\\")
                .replacingOccurrences(of: "\"", with: "\\\"")
            let scriptSrc = "do shell script \"\(escapedForAS)\" with administrator privileges"

            var errDict: NSDictionary?
            let script = NSAppleScript(source: scriptSrc)
            let result = script?.executeAndReturnError(&errDict)

            if result != nil {
                Thread.sleep(forTimeInterval: waitSeconds)
            }

            DispatchQueue.main.async {
                self.isInstalling = false
                if result != nil {
                    self.installError = nil
                    completion(true)
                } else {
                    let msg = (errDict?[NSAppleScript.errorMessage] as? String) ?? "Action cancelled."
                    if !msg.localizedCaseInsensitiveContains("cancel") && !msg.contains("-128") {
                        self.installError = msg
                    }
                    completion(false)
                }
            }
        }
    }

    func openSoundSettings() {
        if let url = URL(string: "x-apple.systempreferences:com.apple.Sound-Settings.extension") {
            NSWorkspace.shared.open(url)
        }
    }

    func openMicrophoneSettings() {
        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone") {
            NSWorkspace.shared.open(url)
        }
    }
}
