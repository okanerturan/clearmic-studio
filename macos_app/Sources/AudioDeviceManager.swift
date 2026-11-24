import CoreAudio
import Foundation

struct AudioDeviceInfo: Identifiable, Hashable {
    let id: AudioDeviceID
    let name: String
    let uid: String
    let isInput: Bool
    let isOutput: Bool
    let sampleRate: Double
    let isClearMic: Bool
    let isBlackHole: Bool

    var isVirtualMic: Bool { isClearMic || isBlackHole }
}

final class AudioDeviceManager: ObservableObject {
    @Published var inputDevices: [AudioDeviceInfo] = []
    @Published var outputDevices: [AudioDeviceInfo] = []
    @Published var hasVirtualMic: Bool = false
    @Published var hasClearMicDriver: Bool = false

    private var listenerBlock: AudioObjectPropertyListenerBlock?

    init() {
        refresh()
        startListening()
    }

    deinit {
        stopListening()
    }

    func refresh() {
        var addr = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDevices,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var size: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &size) == noErr else {
            return
        }
        let count = Int(size) / MemoryLayout<AudioDeviceID>.size
        var ids = [AudioDeviceID](repeating: 0, count: count)
        guard AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &size, &ids) == noErr else {
            return
        }

        var ins: [AudioDeviceInfo] = []
        var outs: [AudioDeviceInfo] = []

        for devID in ids {
            guard let name = getStringProperty(devID, kAudioDevicePropertyDeviceNameCFString),
                  let uid = getStringProperty(devID, kAudioDevicePropertyDeviceUID) else {
                continue
            }
            let inCh = channelCount(devID, scope: kAudioDevicePropertyScopeInput)
            let outCh = channelCount(devID, scope: kAudioDevicePropertyScopeOutput)
            let sr = getNominalSampleRate(devID) ?? 48000.0
            let isCM = (uid == "com.clearmic.virtualmic.device") || name.localizedCaseInsensitiveContains("clearmic")
            let isBH = name.localizedCaseInsensitiveContains("blackhole")
            let info = AudioDeviceInfo(
                id: devID,
                name: name,
                uid: uid,
                isInput: inCh > 0,
                isOutput: outCh > 0,
                sampleRate: sr,
                isClearMic: isCM,
                isBlackHole: isBH
            )
            // Exclude our own virtual mics from Mic Input list to prevent accidental feedback loops
            if inCh > 0 && !info.isVirtualMic {
                ins.append(info)
            }
            if outCh > 0 {
                outs.append(info)
            }
        }

        // Put ClearMic Virtual Mic first, then BlackHole, then physical outputs
        outs.sort { a, b in
            if a.isClearMic != b.isClearMic { return a.isClearMic }
            if a.isBlackHole != b.isBlackHole { return a.isBlackHole }
            return a.name < b.name
        }

        DispatchQueue.main.async {
            self.inputDevices = ins
            self.outputDevices = outs
            self.hasClearMicDriver = outs.contains { $0.isClearMic }
            self.hasVirtualMic = outs.contains { $0.isVirtualMic }
        }
    }

    func defaultInputDeviceID() -> AudioDeviceID? {
        var addr = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultInputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var devID: AudioDeviceID = 0
        var size = UInt32(MemoryLayout<AudioDeviceID>.size)
        let status = AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &size, &devID)
        return status == noErr ? devID : nil
    }

    func defaultOutputDeviceID() -> AudioDeviceID? {
        var addr = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultOutputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var devID: AudioDeviceID = 0
        var size = UInt32(MemoryLayout<AudioDeviceID>.size)
        let status = AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &size, &devID)
        return status == noErr ? devID : nil
    }

    private func startListening() {
        var addr = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDevices,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        let block: AudioObjectPropertyListenerBlock = { [weak self] _, _ in
            self?.refresh()
        }
        listenerBlock = block
        AudioObjectAddPropertyListenerBlock(AudioObjectID(kAudioObjectSystemObject), &addr, DispatchQueue.main, block)
    }

    private func stopListening() {
        guard let block = listenerBlock else { return }
        var addr = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDevices,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        AudioObjectRemovePropertyListenerBlock(AudioObjectID(kAudioObjectSystemObject), &addr, DispatchQueue.main, block)
        listenerBlock = nil
    }

    private func getStringProperty(_ deviceID: AudioDeviceID, _ selector: AudioObjectPropertySelector) -> String? {
        var addr = AudioObjectPropertyAddress(
            mSelector: selector,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var cfStr: Unmanaged<CFString>?
        var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
        let status = AudioObjectGetPropertyData(deviceID, &addr, 0, nil, &size, &cfStr)
        guard status == noErr, let str = cfStr?.takeRetainedValue() else { return nil }
        return str as String
    }

    private func getNominalSampleRate(_ deviceID: AudioDeviceID) -> Double? {
        var addr = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyNominalSampleRate,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var sr: Float64 = 0
        var size = UInt32(MemoryLayout<Float64>.size)
        let status = AudioObjectGetPropertyData(deviceID, &addr, 0, nil, &size, &sr)
        return status == noErr ? sr : nil
    }

    private func channelCount(_ deviceID: AudioDeviceID, scope: AudioObjectPropertyScope) -> Int {
        var addr = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyStreamConfiguration,
            mScope: scope,
            mElement: kAudioObjectPropertyElementMain
        )
        var size: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(deviceID, &addr, 0, nil, &size) == noErr, size > 0 else {
            return 0
        }
        let raw = UnsafeMutableRawPointer.allocate(byteCount: Int(size), alignment: MemoryLayout<AudioBufferList>.alignment)
        defer { raw.deallocate() }
        guard AudioObjectGetPropertyData(deviceID, &addr, 0, nil, &size, raw) == noErr else {
            return 0
        }
        let listPtr = UnsafeMutableAudioBufferListPointer(raw.assumingMemoryBound(to: AudioBufferList.self))
        return listPtr.reduce(0) { $0 + Int($1.mNumberChannels) }
    }
}
