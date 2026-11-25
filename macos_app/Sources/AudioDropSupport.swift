import AppKit
import SwiftUI
import UniformTypeIdentifiers

let kSupportedAudioExtensions: Set<String> = [
    "m4a", "wav", "mp3", "flac", "aiff", "aif", "caf", "aac", "mp4", "mov", "opus", "ogg"
]

func isSupportedAudioURL(_ url: URL) -> Bool {
    let resolved = (url as NSURL).filePathURL ?? url
    let ext = resolved.pathExtension.lowercased()
    if kSupportedAudioExtensions.contains(ext) { return true }
    if !ext.isEmpty, let ut = UTType(filenameExtension: ext) {
        return ut.conforms(to: .audio)
    }
    return false
}

func presentAudioFilePicker(onSelect: @escaping (URL) -> Void) {
    let panel = NSOpenPanel()
    panel.title = "Choose a Voice Memo or audio file"
    panel.message = "Select a recording of your voice alone (.m4a, .wav, .mp3, .aiff, .caf)."
    panel.prompt = "Use for voice print"
    panel.canChooseFiles = true
    panel.canChooseDirectories = false
    panel.allowsMultipleSelection = false
    var types: [UTType] = [.audio, .mpeg4Audio, .wav, .mp3, .aiff]
    if let flac = UTType(filenameExtension: "flac") { types.append(flac) }
    if let caf = UTType(filenameExtension: "caf") { types.append(caf) }
    panel.allowedContentTypes = types
    if panel.runModal() == .OK, let url = panel.url {
        onSelect((url as NSURL).filePathURL ?? url)
    }
}

/// Shared state connecting the AppKit NSDraggingDestination host view to SwiftUI.
final class AudioDropController: ObservableObject {
    @Published var isDraggingAudio: Bool = false
    var onDropStarted: (() -> Void)?
    var onAudioFile: ((URL, String) -> Void)?
    var onDropError: ((String) -> Void)?
}

/// Plain AppKit NSView container implementing NSDraggingDestination.
/// Because it wraps NSHostingView as a parent (rather than relying on SwiftUI's .onDrop),
/// it receives BOTH:
///   1. Apple Voice Memos.app NSFilePromiseReceiver drags (com.apple.m4a-audio file promises)
///   2. Finder / Desktop / Downloads file URLs (including file-reference URLs like file:///.file/id=...)
final class AudioDropHostView: NSView {
    let controller: AudioDropController
    private let dropQueue: OperationQueue = {
        let q = OperationQueue()
        q.name = "com.okanerturan.clearmic.drop"
        q.qualityOfService = .userInitiated
        return q
    }()

    init(childView: NSView, controller: AudioDropController) {
        self.controller = controller
        super.init(frame: .zero)
        childView.unregisterDraggedTypes()
        childView.translatesAutoresizingMaskIntoConstraints = false
        addSubview(childView)
        NSLayoutConstraint.activate([
            childView.leadingAnchor.constraint(equalTo: leadingAnchor),
            childView.trailingAnchor.constraint(equalTo: trailingAnchor),
            childView.topAnchor.constraint(equalTo: topAnchor),
            childView.bottomAnchor.constraint(equalTo: bottomAnchor),
        ])
        registerAudioDragTypes()
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    private func registerAudioDragTypes() {
        var types = NSFilePromiseReceiver.readableDraggedTypes.map { NSPasteboard.PasteboardType($0) }
        types.append(.fileURL)
        types.append(NSPasteboard.PasteboardType("NSFilenamesPboardType"))
        types.append(NSPasteboard.PasteboardType(UTType.audio.identifier))
        types.append(NSPasteboard.PasteboardType(UTType.mpeg4Audio.identifier))
        types.append(NSPasteboard.PasteboardType("com.apple.m4a-audio"))
        registerForDraggedTypes(types)
    }

    // MARK: - NSDraggingDestination

    override func draggingEntered(_ sender: NSDraggingInfo) -> NSDragOperation {
        let ok = canAccept(sender.draggingPasteboard)
        setDragging(ok)
        return ok ? .copy : []
    }

    override func draggingUpdated(_ sender: NSDraggingInfo) -> NSDragOperation {
        let ok = canAccept(sender.draggingPasteboard)
        setDragging(ok)
        return ok ? .copy : []
    }

    override func draggingExited(_ sender: NSDraggingInfo?) {
        setDragging(false)
    }

    override func draggingEnded(_ sender: NSDraggingInfo) {
        setDragging(false)
    }

    override func prepareForDragOperation(_ sender: NSDraggingInfo) -> Bool {
        return canAccept(sender.draggingPasteboard)
    }

    override func performDragOperation(_ sender: NSDraggingInfo) -> Bool {
        setDragging(false)
        let pb = sender.draggingPasteboard

        // 1. Direct local file URL (Finder, Desktop, Downloads, iCloud Drive)
        if let url = extractDirectAudioURL(from: pb) {
            let label = url.deletingPathExtension().lastPathComponent
            DispatchQueue.main.async {
                self.controller.onDropStarted?()
                self.controller.onAudioFile?(url, label)
            }
            return true
        }

        // 2. File Promise (Apple Voice Memos.app, Mail, Messages, etc.)
        if let receivers = pb.readObjects(forClasses: [NSFilePromiseReceiver.self], options: nil) as? [NSFilePromiseReceiver],
           let receiver = receivers.first {
            let destDir = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
                .appendingPathComponent("clearmic_drops", isDirectory: true)
                .appendingPathComponent(UUID().uuidString, isDirectory: true)
            do {
                try FileManager.default.createDirectory(at: destDir, withIntermediateDirectories: true)
            } catch {
                DispatchQueue.main.async {
                    self.controller.onDropError?("Could not create temporary folder for dropped audio: \(error.localizedDescription)")
                }
                return false
            }

            DispatchQueue.main.async {
                self.controller.onDropStarted?()
            }

            receiver.receivePromisedFiles(atDestination: destDir, options: [:], operationQueue: dropQueue) { [weak self] fileURL, error in
                guard let self = self else { return }
                DispatchQueue.main.async {
                    if let error = error {
                        self.controller.onDropError?("Could not receive dropped Voice Memo: \(error.localizedDescription)")
                        return
                    }
                    let resolved = (fileURL as NSURL).filePathURL ?? fileURL
                    let label = resolved.deletingPathExtension().lastPathComponent
                    self.controller.onAudioFile?(resolved, label)
                }
            }
            return true
        }

        // 3. Raw audio bytes on pasteboard (fallback for apps that drag audio data directly)
        let rawTypes: [(NSPasteboard.PasteboardType, String)] = [
            (NSPasteboard.PasteboardType("com.apple.m4a-audio"), "m4a"),
            (NSPasteboard.PasteboardType(UTType.mpeg4Audio.identifier), "m4a"),
            (NSPasteboard.PasteboardType(UTType.wav.identifier), "wav"),
            (NSPasteboard.PasteboardType(UTType.mp3.identifier), "mp3"),
            (NSPasteboard.PasteboardType(UTType.audio.identifier), "m4a"),
        ]
        for (pType, ext) in rawTypes {
            if let data = pb.data(forType: pType), !data.isEmpty {
                let tmpURL = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
                    .appendingPathComponent("clearmic_drop_\(UUID().uuidString).\(ext)")
                do {
                    try data.write(to: tmpURL)
                    DispatchQueue.main.async {
                        self.controller.onDropStarted?()
                        self.controller.onAudioFile?(tmpURL, "Voice Memo")
                    }
                    return true
                } catch {
                    DispatchQueue.main.async {
                        self.controller.onDropError?("Could not read dropped audio: \(error.localizedDescription)")
                    }
                    return false
                }
            }
        }

        return false
    }

    // MARK: - Helpers

    private func setDragging(_ active: Bool) {
        DispatchQueue.main.async {
            if self.controller.isDraggingAudio != active {
                self.controller.isDraggingAudio = active
            }
        }
    }

    private func canAccept(_ pb: NSPasteboard) -> Bool {
        if extractDirectAudioURL(from: pb) != nil {
            return true
        }
        if let receivers = pb.readObjects(forClasses: [NSFilePromiseReceiver.self], options: nil) as? [NSFilePromiseReceiver],
           !receivers.isEmpty {
            // Voice Memos.app promises com.apple.m4a-audio / public.mpeg-4-audio / public.audio.
            for r in receivers {
                if r.fileTypes.isEmpty { return true }
                for ft in r.fileTypes {
                    let lower = ft.lowercased()
                    if lower.contains("audio") || lower.contains("m4a") || lower.contains("wav") || lower.contains("mp3") {
                        return true
                    }
                    if let ut = UTType(ft) {
                        if ut.conforms(to: .audio) || ut.conforms(to: .audiovisualContent) || ut == .data || ut == .item {
                            return true
                        }
                    }
                }
            }
        }
        let rawAudioTypes: [NSPasteboard.PasteboardType] = [
            NSPasteboard.PasteboardType("com.apple.m4a-audio"),
            NSPasteboard.PasteboardType(UTType.mpeg4Audio.identifier),
            NSPasteboard.PasteboardType(UTType.wav.identifier),
            NSPasteboard.PasteboardType(UTType.audio.identifier),
        ]
        if pb.availableType(from: rawAudioTypes) != nil {
            return true
        }
        return false
    }

    private func extractDirectAudioURL(from pb: NSPasteboard) -> URL? {
        if let urls = pb.readObjects(forClasses: [NSURL.self], options: [.urlReadingFileURLsOnly: true]) as? [NSURL] {
            for nsurl in urls {
                let resolved = nsurl.filePathURL ?? (nsurl as URL)
                if isSupportedAudioURL(resolved) {
                    return resolved
                }
            }
        }
        if let paths = pb.propertyList(forType: NSPasteboard.PasteboardType("NSFilenamesPboardType")) as? [String] {
            for p in paths {
                let u = URL(fileURLWithPath: p)
                if isSupportedAudioURL(u) {
                    return u
                }
            }
        }
        return nil
    }
}

/// SwiftUI wrapper around AudioDropHostView for modal sheets (like SetupSheet).
struct AudioDropSheetWrapper<Content: View>: NSViewRepresentable {
    let controller: AudioDropController
    let content: Content

    init(controller: AudioDropController, @ViewBuilder content: () -> Content) {
        self.controller = controller
        self.content = content()
    }

    func makeNSView(context: Context) -> AudioDropSheetContainer<Content> {
        AudioDropSheetContainer(rootView: content, controller: controller)
    }

    func updateNSView(_ nsView: AudioDropSheetContainer<Content>, context: Context) {
        nsView.hostingView.rootView = content
    }

    func sizeThatFits(_ proposal: ProposedViewSize, nsView: AudioDropSheetContainer<Content>, context: Context) -> CGSize? {
        let fit = nsView.hostingView.fittingSize
        return CGSize(
            width: proposal.width ?? fit.width,
            height: proposal.height ?? fit.height
        )
    }
}

final class AudioDropSheetContainer<Content: View>: NSView {
    let hostingView: NSHostingView<Content>
    let dropView: AudioDropHostView

    init(rootView: Content, controller: AudioDropController) {
        self.hostingView = NSHostingView(rootView: rootView)
        self.dropView = AudioDropHostView(childView: hostingView, controller: controller)
        super.init(frame: .zero)
        dropView.translatesAutoresizingMaskIntoConstraints = false
        addSubview(dropView)
        NSLayoutConstraint.activate([
            dropView.leadingAnchor.constraint(equalTo: leadingAnchor),
            dropView.trailingAnchor.constraint(equalTo: trailingAnchor),
            dropView.topAnchor.constraint(equalTo: topAnchor),
            dropView.bottomAnchor.constraint(equalTo: bottomAnchor),
        ])
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
}

/// Restrained full-view overlay displayed while dragging an audio file or Voice Memo over the window.
struct AudioDropOverlay: View {
    var body: some View {
        let shape = RoundedRectangle(cornerRadius: Radius.panel, style: .continuous)
        ZStack {
            Palette.ground.opacity(0.86)
            VStack(spacing: Space.s2) {
                Image(systemName: "waveform.badge.plus")
                    .font(.system(size: 28, weight: .medium))
                    .foregroundStyle(Palette.accent)
                Text("Drop audio to enroll your voice")
                    .font(Typo.headline)
                    .foregroundStyle(Palette.ink)
                Text("Voice Memos, .m4a, .wav, .mp3, .aiff or .flac")
                    .font(Typo.caption)
                    .foregroundStyle(Palette.muted)
            }
            .padding(Space.s5)
            .background(shape.fill(Palette.surface))
            .overlay(shape.strokeBorder(Palette.accent, lineWidth: 1.5))
        }
        .allowsHitTesting(false)
    }
}
