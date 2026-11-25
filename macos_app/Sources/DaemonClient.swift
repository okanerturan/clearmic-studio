import Foundation

final class DaemonClient: ObservableObject {
    @Published var isReady: Bool = false
    @Published var statusMessage: String = "Starting Apple Silicon engine..."
    @Published var socketPath: String? = nil
    @Published var hasEnrollment: Bool = false
    @Published var isEnrolling: Bool = false
    @Published var enrollLabel: String = "No voice print yet"
    @Published var mlxMemMB: Double = 0.0
    @Published var computeMs: Double = 0.0
    @Published var algoMs: Double = 85.3
    @Published var e2eMs: Double = 95.0
    @Published var lastError: String? = nil

    private var process: Process?
    private var stdinPipe: Pipe?
    private var stdoutPipe: Pipe?
    private var stderrPipe: Pipe?
    private let writeQueue = DispatchQueue(label: "clearmic.daemon.write")

    let repoRoot: String
    let sockFile: String
    let appSupportDir: String

    init(repoRoot: String? = nil) {
        if let explicit = repoRoot {
            self.repoRoot = explicit
        } else {
            // Resolve repo root relative to ClearMic Studio.app location
            let appParent = Bundle.main.bundleURL.deletingLastPathComponent().path
            if FileManager.default.fileExists(atPath: "\(appParent)/engine/live_daemon.py") {
                self.repoRoot = appParent
            } else {
                self.repoRoot = FileManager.default.currentDirectoryPath
            }
        }
        let appSup = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            .appendingPathComponent("ClearMic Studio", isDirectory: true)
        try? FileManager.default.createDirectory(at: appSup, withIntermediateDirectories: true)
        self.appSupportDir = appSup.path
        self.sockFile = "/tmp/clearmic_audio_\(ProcessInfo.processInfo.processIdentifier).sock"
    }

    private func resolvePython() -> String {
        if let envPy = ProcessInfo.processInfo.environment["CLEARMIC_PYTHON"],
           FileManager.default.isExecutableFile(atPath: envPy) {
            return envPy
        }
        let candidates = [
            "\(repoRoot)/.venv/bin/python",
            "\(repoRoot)/../voice-isolator/.venv/bin/python",
            "/opt/homebrew/bin/python3",
            "/usr/bin/python3",
        ]
        for c in candidates {
            if FileManager.default.isExecutableFile(atPath: c) {
                return c
            }
        }
        return "/usr/bin/env"
    }

    private func resolveDaemonScript() -> String {
        let enginePath = "\(repoRoot)/engine/live_daemon.py"
        if FileManager.default.fileExists(atPath: enginePath) { return enginePath }
        return "\(repoRoot)/studio/live_daemon.py"
    }

    func start() {
        guard process == nil else { return }
        let py = resolvePython()
        let script = resolveDaemonScript()
        let enrollPath = "\(appSupportDir)/voice_print_16k.wav"

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: py)
        proc.arguments = [script, "--socket", sockFile, "--enroll", enrollPath]
        proc.currentDirectoryURL = URL(fileURLWithPath: repoRoot)

        let inPipe = Pipe()
        let outPipe = Pipe()
        let errPipe = Pipe()
        proc.standardInput = inPipe
        proc.standardOutput = outPipe
        proc.standardError = errPipe

        self.stdinPipe = inPipe
        self.stdoutPipe = outPipe
        self.stderrPipe = errPipe
        self.process = proc

        let outHandle = outPipe.fileHandleForReading
        var lineBuffer = Data()
        outHandle.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            if data.isEmpty {
                handle.readabilityHandler = nil
                return
            }
            lineBuffer.append(data)
            while let nl = lineBuffer.firstIndex(of: 0x0A) {
                let lineData = lineBuffer.subdata(in: 0..<nl)
                lineBuffer.removeSubrange(0...nl)
                self?.handleEventLine(lineData)
            }
        }

        let errHandle = errPipe.fileHandleForReading
        errHandle.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            if data.isEmpty {
                handle.readabilityHandler = nil
                return
            }
            if let s = String(data: data, encoding: .utf8), !s.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                if s.contains("Traceback") || s.contains("Error:") {
                    DispatchQueue.main.async {
                        self?.lastError = s.trimmingCharacters(in: .whitespacesAndNewlines)
                    }
                }
            }
        }

        proc.terminationHandler = { [weak self] p in
            DispatchQueue.main.async {
                self?.isReady = false
                if p.terminationStatus != 0 {
                    self?.statusMessage = "Engine exited (\(p.terminationStatus))"
                }
            }
        }

        do {
            try proc.run()
        } catch {
            DispatchQueue.main.async {
                self.lastError = "Failed to launch engine: \(error.localizedDescription)"
                self.statusMessage = "Launch failed"
            }
        }
    }

    func stop() {
        sendJSON(["cmd": "quit"])
        if let p = process, p.isRunning {
            p.terminate()
            p.waitUntilExit()
        }
        process = nil
        stdinPipe = nil
        stdoutPipe?.fileHandleForReading.readabilityHandler = nil
        stderrPipe?.fileHandleForReading.readabilityHandler = nil
        try? FileManager.default.removeItem(atPath: sockFile)
    }

    func configure(preset: String, studioEQ: Bool, gateGamma: Double, outputGain: Double, bypass: Bool) {
        sendJSON([
            "cmd": "configure",
            "preset": preset,
            "studio_eq": studioEQ,
            "gate_gamma": gateGamma,
            "output_gain": outputGain,
            "bypass": bypass,
        ])
    }

    func resetStream() {
        sendJSON(["cmd": "reset"])
    }

    func enroll(wavPath: String, label: String? = nil) {
        DispatchQueue.main.async {
            self.isEnrolling = true
            self.lastError = nil
        }
        let savePath = "\(appSupportDir)/voice_print_16k.wav"
        var payload: [String: Any] = [
            "cmd": "enroll",
            "wav_path": wavPath,
            "save_path": savePath,
        ]
        if let label = label, !label.isEmpty {
            payload["label"] = label
        }
        sendJSON(payload)
    }

    private func sendJSON(_ dict: [String: Any]) {
        writeQueue.async { [weak self] in
            guard let pipe = self?.stdinPipe,
                  let data = try? JSONSerialization.data(withJSONObject: dict) else { return }
            var out = data
            out.append(0x0A)
            try? pipe.fileHandleForWriting.write(contentsOf: out)
        }
    }

    private func handleEventLine(_ data: Data) {
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let event = obj["event"] as? String else { return }

        DispatchQueue.main.async {
            switch event {
            case "loading":
                self.statusMessage = (obj["message"] as? String) ?? "Loading..."
            case "ready":
                self.isReady = true
                self.statusMessage = "Apple Silicon Engine Ready"
                self.hasEnrollment = (obj["has_enrollment"] as? Bool) ?? false
                self.mlxMemMB = (obj["mlx_mem_mb"] as? Double) ?? 0.0
                if self.hasEnrollment {
                    self.enrollLabel = "Your saved voice print"
                }
            case "socket_ready":
                self.socketPath = obj["socket"] as? String
            case "stats":
                self.computeMs = (obj["compute_ms"] as? Double) ?? self.computeMs
                self.algoMs = (obj["algo_ms"] as? Double) ?? self.algoMs
                self.e2eMs = (obj["e2e_ms"] as? Double) ?? self.e2eMs
            case "enrolled":
                self.isEnrolling = false
                self.hasEnrollment = true
                let dur = (obj["duration_s"] as? Double) ?? 5.0
                if let label = obj["label"] as? String, !label.isEmpty {
                    self.enrollLabel = String(format: "%@ (%.1fs)", label, dur)
                } else {
                    self.enrollLabel = String(format: "Your voice (%.1fs)", dur)
                }
                self.statusMessage = "Voice Print Updated"
            case "enroll_error":
                self.isEnrolling = false
                self.lastError = (obj["error"] as? String) ?? "Failed to enroll voice print."
            default:
                break
            }
        }
    }
}
