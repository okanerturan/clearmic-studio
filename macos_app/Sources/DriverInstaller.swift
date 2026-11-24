import AppKit
import AVFoundation
import Foundation

/// Installs the bundled, Developer-ID-signed ClearMic.driver into /Library/Audio/Plug-Ins/HAL/
/// via macOS's native Touch ID / Admin Password sheet — no Terminal, no Homebrew, no reboot.
/// Also provides direct links to macOS Sound & Microphone System Settings panes.
final class DriverInstaller: ObservableObject {
    @Published var isInstalling: Bool = false
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
        isInstalling = true
        installError = nil

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            let escapedSrc = src.replacingOccurrences(of: "'", with: "'\\''")
            let shellCmd = """
            mkdir -p '\(self.halDir)' && \
            rm -rf '\(self.installedPath)' && \
            cp -R '\(escapedSrc)' '\(self.installedPath)' && \
            chown -R root:wheel '\(self.installedPath)' && \
            chmod -R a+rX '\(self.installedPath)' && \
            killall -9 coreaudiod || true
            """
            let escapedForAS = shellCmd
                .replacingOccurrences(of: "\\", with: "\\\\")
                .replacingOccurrences(of: "\"", with: "\\\"")
            let scriptSrc = "do shell script \"\(escapedForAS)\" with administrator privileges"

            var errDict: NSDictionary?
            let script = NSAppleScript(source: scriptSrc)
            let result = script?.executeAndReturnError(&errDict)

            // Wait briefly for coreaudiod to respawn and register ClearMic Virtual Mic
            if result != nil {
                Thread.sleep(forTimeInterval: 1.4)
            }

            DispatchQueue.main.async {
                self.isInstalling = false
                if result != nil {
                    self.installError = nil
                    completion(true)
                } else {
                    let msg = (errDict?[NSAppleScript.errorMessage] as? String) ?? "Installation cancelled."
                    // User cancelling the password sheet isn't an error worth shouting about
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
