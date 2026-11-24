import AppKit
import CoreAudio
import SwiftUI

/// A pop-up menu of audio devices that fills the width it is given. SwiftUI's own Picker keeps
/// the width of its longest item on macOS, which left the two device menus at different widths.
struct DeviceMenu: NSViewRepresentable {
    struct Item: Equatable {
        let id: AudioDeviceID
        let title: String
    }

    let label: String
    let items: [Item]
    @Binding var selection: AudioDeviceID

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    func makeNSView(context: Context) -> NSPopUpButton {
        let button = NSPopUpButton(frame: .zero, pullsDown: false)
        button.target = context.coordinator
        button.action = #selector(Coordinator.selectionChanged(_:))
        button.setContentHuggingPriority(.defaultLow, for: .horizontal)
        button.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
        return button
    }

    func updateNSView(_ button: NSPopUpButton, context: Context) {
        context.coordinator.parent = self
        if context.coordinator.shown != items {
            // Built item by item: addItems(withTitles:) drops a second device with the same name.
            button.removeAllItems()
            for item in items {
                button.menu?.addItem(NSMenuItem(title: item.title, action: nil, keyEquivalent: ""))
            }
            context.coordinator.shown = items
        }
        if let index = items.firstIndex(where: { $0.id == selection }) {
            button.selectItem(at: index)
        } else {
            button.select(nil)
        }
        button.isEnabled = context.environment.isEnabled
        button.setAccessibilityLabel(label)
    }

    func sizeThatFits(_ proposal: ProposedViewSize, nsView: NSPopUpButton, context: Context) -> CGSize? {
        let natural = nsView.intrinsicContentSize
        guard let proposed = proposal.width, proposed.isFinite else { return natural }
        return CGSize(width: max(proposed, 80), height: natural.height)
    }

    final class Coordinator: NSObject {
        var parent: DeviceMenu
        var shown: [Item] = []

        init(_ parent: DeviceMenu) {
            self.parent = parent
        }

        @objc func selectionChanged(_ sender: NSPopUpButton) {
            let index = sender.indexOfSelectedItem
            guard parent.items.indices.contains(index) else { return }
            parent.selection = parent.items[index].id
        }
    }
}
