import AVFoundation
import CoreAudio
import Foundation
import os

/// Lock-protected float32 ring buffer for real-time audio callback <-> socket thread handoff.
final class FloatRingBuffer {
    private var buf: [Float]
    private var readIdx: Int = 0
    private var writeIdx: Int = 0
    private var count: Int = 0
    private let capacity: Int
    private let lock = OSAllocatedUnfairLock()

    init(capacity: Int = 48000 * 2) {
        self.capacity = capacity
        self.buf = [Float](repeating: 0, count: capacity)
    }

    func reset() {
        lock.withLock {
            readIdx = 0
            writeIdx = 0
            count = 0
        }
    }

    func available() -> Int {
        lock.withLock { count }
    }

    func write(_ samples: UnsafePointer<Float>, count n: Int) {
        lock.withLock {
            // If overflowing (e.g. consumer stalled), drop oldest to keep latency low!
            if count + n > capacity {
                let drop = (count + n) - capacity
                readIdx = (readIdx + drop) % capacity
                count -= drop
            }
            let first = min(n, capacity - writeIdx)
            for i in 0..<first { buf[writeIdx + i] = samples[i] }
            let rem = n - first
            for i in 0..<rem { buf[i] = samples[first + i] }
            writeIdx = (writeIdx + n) % capacity
            count += n
        }
    }

    func read(into dst: UnsafeMutablePointer<Float>, count n: Int) -> Int {
        lock.withLock {
            let take = min(n, count)
            let first = min(take, capacity - readIdx)
            for i in 0..<first { dst[i] = buf[readIdx + i] }
            let rem = take - first
            for i in 0..<rem { dst[first + i] = buf[i] }
            readIdx = (readIdx + take) % capacity
            count -= take
            return take
        }
    }
}

/// Manages real-time CoreAudio capture (selected mic) -> Unix socket -> M5 engine -> selected output (BlackHole 2ch).
/// Also supports 5s Voice Print enrollment recording and 5s Sound Check self-monitoring playback.
final class CoreAudioBridge: ObservableObject {
    @Published var isRunning: Bool = false
    @Published var inLevel: Float = 0.0
    @Published var outLevel: Float = 0.0
    @Published var inWave: [Float] = Array(repeating: 0.02, count: 48)
    @Published var outWave: [Float] = Array(repeating: 0.02, count: 48)
    @Published var isRecordingEnroll: Bool = false
    @Published var enrollProgress: Double = 0.0
    @Published var isSoundCheckRecording: Bool = false
    @Published var isSoundCheckPlaying: Bool = false
    @Published var soundCheckProgress: Double = 0.0
    @Published var errorMessage: String? = nil

    private var inEngine: AVAudioEngine?
    private var outEngine: AVAudioEngine?
    private let inRing = FloatRingBuffer(capacity: 48000)
    private let outRing = FloatRingBuffer(capacity: 48000)

    private var sockThread: Thread?
    private var shouldRun: Bool = false
    private let chunkSamples = 960 // 20 ms @ 48 kHz

    // Recording buffers for enrollment & sound check
    private let recLock = OSAllocatedUnfairLock()
    private var enrollBuf: [Float] = []
    private var enrollingActive: Bool = false
    private var soundCheckBuf: [Float] = []
    private var soundCheckRecActive: Bool = false
    private var scPlayer: AVAudioPlayer?

    private var meterTimer: Timer?
    private var latestInRMS: Float = 0
    private var latestOutRMS: Float = 0

    func requestMicPermission(completion: @escaping (Bool) -> Void) {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized:
            completion(true)
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .audio) { granted in
                DispatchQueue.main.async { completion(granted) }
            }
        default:
            completion(false)
        }
    }

    func start(inputDeviceID: AudioDeviceID, outputDeviceID: AudioDeviceID, socketPath: String) {
        stop()
        errorMessage = nil

        requestMicPermission { [weak self] granted in
            guard let self = self else { return }
            guard granted else {
                self.errorMessage = "Microphone permission denied. Enable ClearMic Studio in System Settings → Privacy & Security → Microphone."
                return
            }
            self.startEngines(inputDeviceID: inputDeviceID, outputDeviceID: outputDeviceID, socketPath: socketPath)
        }
    }

    private func startEngines(inputDeviceID: AudioDeviceID, outputDeviceID: AudioDeviceID, socketPath: String) {
        inRing.reset()
        outRing.reset()

        let inEng = AVAudioEngine()
        let outEng = AVAudioEngine()

        // Bind input engine to selected physical microphone
        if let inAU = inEng.inputNode.audioUnit {
            var dev = inputDeviceID
            let st = AudioUnitSetProperty(
                inAU,
                kAudioOutputUnitProperty_CurrentDevice,
                kAudioUnitScope_Global,
                0,
                &dev,
                UInt32(MemoryLayout<AudioDeviceID>.size)
            )
            if st != noErr {
                errorMessage = "Could not select input microphone (CoreAudio error \(st))."
                return
            }
        }

        // Bind output engine to selected output (e.g. BlackHole 2ch)
        if let outAU = outEng.outputNode.audioUnit {
            var dev = outputDeviceID
            let st = AudioUnitSetProperty(
                outAU,
                kAudioOutputUnitProperty_CurrentDevice,
                kAudioUnitScope_Global,
                0,
                &dev,
                UInt32(MemoryLayout<AudioDeviceID>.size)
            )
            if st != noErr {
                errorMessage = "Could not select output device (CoreAudio error \(st))."
                return
            }
        }

        // 1. Setup Input Capture Tap -> convert ANY mic rate/channels to 48 kHz mono float32!
        let hwInFmt = inEng.inputNode.inputFormat(forBus: 0)
        guard hwInFmt.sampleRate > 0 && hwInFmt.channelCount > 0 else {
            errorMessage = "Selected microphone returned an invalid audio format."
            return
        }
        guard let targetFmt = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 48000, channels: 1, interleaved: false),
              let converter = AVAudioConverter(from: hwInFmt, to: targetFmt) else {
            errorMessage = "Could not initialize 48 kHz sample-rate converter."
            return
        }

        let tapFrames = AVAudioFrameCount(max(256, Int(hwInFmt.sampleRate * 0.02)))
        inEng.inputNode.installTap(onBus: 0, bufferSize: tapFrames, format: hwInFmt) { [weak self] pcmBuf, _ in
            guard let self = self else { return }
            let ratio = 48000.0 / hwInFmt.sampleRate
            let outCap = AVAudioFrameCount(ceil(Double(pcmBuf.frameLength) * ratio) + 64)
            guard let outBuf = AVAudioPCMBuffer(pcmFormat: targetFmt, frameCapacity: outCap) else { return }
            var err: NSError?
            var consumed = false
            converter.convert(to: outBuf, error: &err) { _, outStatus in
                if consumed {
                    outStatus.pointee = .noDataNow
                    return nil
                }
                consumed = true
                outStatus.pointee = .haveData
                return pcmBuf
            }
            guard err == nil, let ch = outBuf.floatChannelData?[0] else { return }
            let n = Int(outBuf.frameLength)
            if n > 0 {
                // Sanitize non-finite samples (e.g. Continuity Camera iPhone mic connect glitches)
                // and compute input RMS for live meter
                var sumSq: Float = 0
                for i in 0..<n {
                    var v = ch[i]
                    if !v.isFinite {
                        v = 0
                        ch[i] = 0
                    }
                    sumSq += v * v
                }
                self.latestInRMS = sqrtf(sumSq / Float(n))

                // If recording a 5s voice print, stash raw 48k audio
                self.recLock.withLock {
                    if self.enrollingActive {
                        let ptr = UnsafeBufferPointer(start: ch, count: n)
                        self.enrollBuf.append(contentsOf: ptr)
                    }
                }
                self.inRing.write(ch, count: n)
            }
        }

        // 2. Setup Output Source Node (48 kHz mono -> output device mixer handles stereo duplication + rate!)
        let srcFmt = targetFmt
        var scratch = [Float](repeating: 0, count: 4096)
        let srcNode = AVAudioSourceNode(format: srcFmt) { [weak self] _, _, frameCount, audioBufferList -> OSStatus in
            guard let self = self else { return noErr }
            let n = Int(frameCount)
            if n > scratch.count { scratch = [Float](repeating: 0, count: n) }
            let got = self.outRing.read(into: &scratch, count: n)
            if got < n {
                for i in got..<n { scratch[i] = 0 }
            }
            let abl = UnsafeMutableAudioBufferListPointer(audioBufferList)
            for buf in abl {
                if let dst = buf.mData?.assumingMemoryBound(to: Float.self) {
                    memcpy(dst, scratch, n * MemoryLayout<Float>.size)
                }
            }
            return noErr
        }

        outEng.attach(srcNode)
        outEng.connect(srcNode, to: outEng.mainMixerNode, format: srcFmt)

        // Start Socket I/O thread before starting audio engines
        shouldRun = true
        let th = Thread { [weak self] in
            self?.socketWorker(socketPath: socketPath)
        }
        th.qualityOfService = .userInteractive
        self.sockThread = th
        th.start()

        do {
            try inEng.start()
            try outEng.start()
            self.inEngine = inEng
            self.outEngine = outEng
            self.isRunning = true
            startMeterTimer()
        } catch {
            shouldRun = false
            inEng.inputNode.removeTap(onBus: 0)
            inEng.stop()
            outEng.stop()
            errorMessage = "Audio engine start failed: \(error.localizedDescription)"
        }
    }

    func stop() {
        shouldRun = false
        meterTimer?.invalidate()
        meterTimer = nil

        if let inEng = inEngine {
            inEng.inputNode.removeTap(onBus: 0)
            inEng.stop()
            inEngine = nil
        }
        if let outEng = outEngine {
            outEng.stop()
            outEngine = nil
        }
        sockThread = nil
        inRing.reset()
        outRing.reset()
        isRunning = false
        inLevel = 0
        outLevel = 0
    }

    /// Records 5 seconds of the user's raw mic voice to a WAV file and invokes callback with the path.
    func recordEnrollment(inputDeviceID: AudioDeviceID, seconds: Double = 5.0, completion: @escaping (String?) -> Void) {
        let wasRunning = isRunning
        recLock.withLock {
            enrollBuf.removeAll(keepingCapacity: true)
            enrollBuf.reserveCapacity(48000 * Int(seconds + 1))
            enrollingActive = true
        }
        isRecordingEnroll = true
        enrollProgress = 0.0

        // If engine isn't live yet, spin up a temporary capture-only AVAudioEngine for 5 seconds!
        var tempEngine: AVAudioEngine? = nil
        if !wasRunning {
            let eng = AVAudioEngine()
            if let inAU = eng.inputNode.audioUnit {
                var dev = inputDeviceID
                AudioUnitSetProperty(inAU, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0, &dev, UInt32(MemoryLayout<AudioDeviceID>.size))
            }
            let hwFmt = eng.inputNode.inputFormat(forBus: 0)
            if let tgt = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 48000, channels: 1, interleaved: false),
               let conv = AVAudioConverter(from: hwFmt, to: tgt) {
                eng.inputNode.installTap(onBus: 0, bufferSize: 1024, format: hwFmt) { [weak self] pcmBuf, _ in
                    guard let self = self else { return }
                    let cap = AVAudioFrameCount(ceil(Double(pcmBuf.frameLength) * (48000.0 / hwFmt.sampleRate)) + 64)
                    guard let outBuf = AVAudioPCMBuffer(pcmFormat: tgt, frameCapacity: cap) else { return }
                    var err: NSError?; var done = false
                    conv.convert(to: outBuf, error: &err) { _, st in
                        if done { st.pointee = .noDataNow; return nil }
                        done = true; st.pointee = .haveData; return pcmBuf
                    }
                    if err == nil, let ch = outBuf.floatChannelData?[0] {
                        let n = Int(outBuf.frameLength)
                        self.recLock.withLock {
                            if self.enrollingActive {
                                self.enrollBuf.append(contentsOf: UnsafeBufferPointer(start: ch, count: n))
                            }
                        }
                    }
                }
                try? eng.start()
                tempEngine = eng
            }
        }

        let startT = Date()
        Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak self] t in
            guard let self = self else { t.invalidate(); return }
            let elapsed = Date().timeIntervalSince(startT)
            self.enrollProgress = min(1.0, elapsed / seconds)
            if elapsed >= seconds {
                t.invalidate()
                if let eng = tempEngine {
                    eng.inputNode.removeTap(onBus: 0)
                    eng.stop()
                }
                var samples: [Float] = []
                self.recLock.withLock {
                    self.enrollingActive = false
                    samples = self.enrollBuf
                    self.enrollBuf.removeAll()
                }
                self.isRecordingEnroll = false
                self.enrollProgress = 0.0
                let path = "/tmp/clearmic_enroll_rec.wav"
                if self.writeWav48k(samples: samples, to: path) {
                    completion(path)
                } else {
                    completion(nil)
                }
            }
        }
    }

    /// Captures 5 seconds of LIVE processed output and plays it straight back to default speakers!
    func runSoundCheck(seconds: Double = 5.0) {
        guard isRunning, !isSoundCheckRecording else { return }
        recLock.withLock {
            soundCheckBuf.removeAll(keepingCapacity: true)
            soundCheckBuf.reserveCapacity(48000 * Int(seconds + 1))
            soundCheckRecActive = true
        }
        isSoundCheckRecording = true
        soundCheckProgress = 0.0

        let startT = Date()
        Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak self] t in
            guard let self = self else { t.invalidate(); return }
            let elapsed = Date().timeIntervalSince(startT)
            self.soundCheckProgress = min(1.0, elapsed / seconds)
            if elapsed >= seconds {
                t.invalidate()
                var samples: [Float] = []
                self.recLock.withLock {
                    self.soundCheckRecActive = false
                    samples = self.soundCheckBuf
                    self.soundCheckBuf.removeAll()
                }
                self.isSoundCheckRecording = false
                self.soundCheckProgress = 0.0
                let path = "/tmp/clearmic_soundcheck.wav"
                if self.writeWav48k(samples: samples, to: path) {
                    do {
                        let player = try AVAudioPlayer(contentsOf: URL(fileURLWithPath: path))
                        self.scPlayer = player
                        self.isSoundCheckPlaying = true
                        player.play()
                        DispatchQueue.main.asyncAfter(deadline: .now() + player.duration + 0.1) {
                            self.isSoundCheckPlaying = false
                        }
                    } catch {
                        self.errorMessage = "Sound check playback failed: \(error.localizedDescription)"
                    }
                }
            }
        }
    }

    private func writeWav48k(samples: [Float], to path: String) -> Bool {
        guard !samples.isEmpty else { return false }
        guard let fmt = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 48000, channels: 1, interleaved: false),
              let buf = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: AVAudioFrameCount(samples.count)) else {
            return false
        }
        buf.frameLength = AVAudioFrameCount(samples.count)
        if let dst = buf.floatChannelData?[0] {
            memcpy(dst, samples, samples.count * MemoryLayout<Float>.size)
        }
        do {
            let url = URL(fileURLWithPath: path)
            try? FileManager.default.removeItem(at: url)
            let file = try AVAudioFile(forWriting: url, settings: fmt.settings, commonFormat: .pcmFormatFloat32, interleaved: false)
            try file.write(from: buf)
            return true
        } catch {
            return false
        }
    }

    private func startMeterTimer() {
        meterTimer?.invalidate()
        meterTimer = Timer.scheduledTimer(withTimeInterval: 1.0 / 30.0, repeats: true) { [weak self] _ in
            guard let self = self else { return }
            // Map RMS to smooth perceptual 0..1 visual meter (-54 dB .. -3 dB)
            let inDB = 20.0 * log10f(max(self.latestInRMS, 1e-5))
            let outDB = 20.0 * log10f(max(self.latestOutRMS, 1e-5))
            let inNorm = max(0.0, min(1.0, (inDB + 54.0) / 51.0))
            let outNorm = max(0.0, min(1.0, (outDB + 54.0) / 51.0))
            self.inLevel = self.inLevel * 0.45 + inNorm * 0.55
            self.outLevel = self.outLevel * 0.45 + outNorm * 0.55
            self.inWave.removeFirst()
            self.inWave.append(max(0.03, self.inLevel))
            self.outWave.removeFirst()
            self.outWave.append(max(0.03, self.outLevel))
        }
    }

    // MARK: - Unix Domain Socket Worker (Exact 960-sample = 20ms lock-step PCM exchange)
    private func socketWorker(socketPath: String) {
        let fd = Darwin.socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else { return }
        defer { Darwin.close(fd) }

        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        let maxPathLen = MemoryLayout.size(ofValue: addr.sun_path) - 1
        socketPath.withCString { cstr in
            withUnsafeMutablePointer(to: &addr.sun_path) { ptr in
                let raw = UnsafeMutableRawPointer(ptr).assumingMemoryBound(to: CChar.self)
                strncpy(raw, cstr, maxPathLen)
            }
        }

        // Wait briefly for daemon socket to be ready
        var connected = false
        for _ in 0..<25 {
            if !shouldRun { return }
            let res = withUnsafePointer(to: &addr) { ptr in
                ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sa in
                    Darwin.connect(fd, sa, socklen_t(MemoryLayout<sockaddr_un>.size))
                }
            }
            if res == 0 {
                connected = true
                break
            }
            usleep(100_000)
        }
        guard connected else {
            DispatchQueue.main.async { self.errorMessage = "Could not connect to M5 audio daemon socket." }
            return
        }

        var inChunk = [Float](repeating: 0, count: chunkSamples)
        var outChunk = [Float](repeating: 0, count: chunkSamples)
        var hdr = UInt32(chunkSamples).littleEndian
        let byteLen = chunkSamples * MemoryLayout<Float>.size

        while shouldRun {
            if inRing.available() < chunkSamples {
                usleep(2000) // 2 ms poll wait
                continue
            }
            _ = inRing.read(into: &inChunk, count: chunkSamples)

            // Send [uint32 count][float32 payload]
            let okHdr = withUnsafeBytes(of: &hdr) { sendAll(fd, $0.baseAddress!, 4) }
            let okPay = inChunk.withUnsafeBytes { sendAll(fd, $0.baseAddress!, byteLen) }
            if !okHdr || !okPay { break }

            // Receive [uint32 count][float32 payload]
            var rxHdr: UInt32 = 0
            let okRxH = withUnsafeMutableBytes(of: &rxHdr) { recvAll(fd, $0.baseAddress!, 4) }
            if !okRxH { break }
            let rxN = Int(UInt32(littleEndian: rxHdr))
            if rxN != chunkSamples { break }
            let okRxP = outChunk.withUnsafeMutableBytes { recvAll(fd, $0.baseAddress!, byteLen) }
            if !okRxP { break }

            // Sanitize & update output RMS meter + optional sound-check capture
            var sumSq: Float = 0
            for i in 0..<chunkSamples {
                var v = outChunk[i]
                if !v.isFinite {
                    v = 0
                    outChunk[i] = 0
                }
                sumSq += v * v
            }
            latestOutRMS = sqrtf(sumSq / Float(chunkSamples))

            recLock.withLock {
                if soundCheckRecActive {
                    soundCheckBuf.append(contentsOf: outChunk)
                }
            }
            outChunk.withUnsafeBufferPointer { ptr in
                outRing.write(ptr.baseAddress!, count: chunkSamples)
            }
        }
    }

    private func sendAll(_ fd: Int32, _ ptr: UnsafeRawPointer, _ len: Int) -> Bool {
        var sent = 0
        while sent < len {
            let n = Darwin.send(fd, ptr.advanced(by: sent), len - sent, 0)
            if n <= 0 { return false }
            sent += n
        }
        return true
    }

    private func recvAll(_ fd: Int32, _ ptr: UnsafeMutableRawPointer, _ len: Int) -> Bool {
        var got = 0
        while got < len {
            let n = Darwin.recv(fd, ptr.advanced(by: got), len - got, 0)
            if n <= 0 { return false }
            got += n
        }
        return true
    }
}
