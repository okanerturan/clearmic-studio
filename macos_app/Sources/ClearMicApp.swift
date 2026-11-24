import AppKit
import SwiftUI

final class AppDelegate: NSObject, NSApplicationDelegate {
    var window: NSWindow!
    let devices = AudioDeviceManager()
    let installer = DriverInstaller()
    let daemon = DaemonClient()
    let bridge = CoreAudioBridge()

    func applicationDidFinishLaunching(_ notification: Notification) {
        let rootView = ContentView(devices: devices, installer: installer, daemon: daemon, bridge: bridge)
        let hosting = NSHostingView(rootView: rootView)
        window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 760, height: 660),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.title = "ClearMic Studio"
        window.titlebarAppearsTransparent = true
        window.appearance = NSAppearance(named: .darkAqua)
        window.backgroundColor = Palette.groundNS
        window.contentView = hosting
        window.center()
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true
    }

    func applicationWillTerminate(_ notification: Notification) {
        bridge.stop()
        daemon.stop()
    }
}

@main
struct ClearMicMain {
    static func main() {
        let app = NSApplication.shared
        app.setActivationPolicy(.regular)
        let delegate = AppDelegate()
        app.delegate = delegate
        app.run()
    }
}
