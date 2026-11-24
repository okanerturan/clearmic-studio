import AppKit
import SwiftUI

// One system for the whole window: a neutral dark ground, one ink, one accent,
// one spacing scale, one radius scale and one pair of buttons. Amber and red
// are kept for warnings and errors; nothing else in the window is coloured.

enum Palette {
    static let groundNS = NSColor(srgbRed: 0.059, green: 0.063, blue: 0.071, alpha: 1.0)
    static let ground = Color(nsColor: groundNS)
    static let surface = Color(red: 0.094, green: 0.098, blue: 0.110)
    static let ink = Color(red: 0.945, green: 0.949, blue: 0.937)
    static let muted = Color(red: 0.635, green: 0.647, blue: 0.667)
    static let line = Color.white.opacity(0.10)
    static let lineStrong = Color.white.opacity(0.24)
    static let fill = Color.white.opacity(0.06)
    static let fillHover = Color.white.opacity(0.11)
    static let accent = Color(red: 0.169, green: 0.831, blue: 0.490)
    static let onAccent = Color(red: 0.020, green: 0.086, blue: 0.047)
    static let warning = Color(red: 0.949, green: 0.663, blue: 0.231)
    static let danger = Color(red: 0.961, green: 0.431, blue: 0.392)
    static let inputBars = Color.white.opacity(0.36)
}

enum Space {
    static let s1: CGFloat = 4
    static let s2: CGFloat = 8
    static let s3: CGFloat = 12
    static let s4: CGFloat = 16
    static let s5: CGFloat = 24
}

enum Radius {
    static let control: CGFloat = 8
    static let tile: CGFloat = 10
    static let panel: CGFloat = 14
}

enum Typo {
    static let headline = Font.system(size: 20, weight: .semibold)
    static let title = Font.system(size: 13, weight: .semibold)
    static let strong = Font.system(size: 13, weight: .medium)
    static let body = Font.system(size: 13)
    static let label = Font.system(size: 12, weight: .medium)
    static let caption = Font.system(size: 12)
    static let figure = Font.system(size: 12, weight: .medium).monospacedDigit()
    static let code = Font.system(size: 12, design: .monospaced)
}

/// Hover in 150 ms, out 120 ms, press 90 ms, on one ease-out curve that never overshoots.
enum Motion {
    static let hoverIn = Animation.timingCurve(0.2, 0.8, 0.2, 1.0, duration: 0.15)
    static let hoverOut = Animation.timingCurve(0.2, 0.8, 0.2, 1.0, duration: 0.12)
    static let press = Animation.timingCurve(0.2, 0.8, 0.2, 1.0, duration: 0.09)
}

struct SectionLabel: View {
    let text: String

    init(_ text: String) {
        self.text = text
    }

    var body: some View {
        Text(text)
            .font(Typo.label)
            .foregroundStyle(Palette.muted)
            .accessibilityAddTraits(.isHeader)
    }
}

// MARK: - Buttons

/// The window's one button pair. Exactly one primary is on screen at a time.
struct ActionButtonStyle: ButtonStyle {
    enum Kind {
        case primary
        case secondary
    }

    let kind: Kind

    init(_ kind: Kind) {
        self.kind = kind
    }

    func makeBody(configuration: Configuration) -> some View {
        ActionButtonBody(kind: kind, configuration: configuration)
    }
}

private struct ActionButtonBody: View {
    let kind: ActionButtonStyle.Kind
    let configuration: ButtonStyleConfiguration

    @Environment(\.isEnabled) private var isEnabled
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var hovering = false

    var body: some View {
        let pressed = configuration.isPressed
        let shape = RoundedRectangle(cornerRadius: Radius.control, style: .continuous)

        configuration.label
            .font(Typo.title)
            .foregroundStyle(foreground)
            .lineLimit(1)
            .fixedSize()
            .padding(.horizontal, Space.s4)
            .frame(height: 32)
            .background(shape.fill(fill(pressed: pressed)))
            .overlay(shape.strokeBorder(border, lineWidth: 1))
            .contentShape(shape)
            .scaleEffect(pressed && !reduceMotion ? 0.985 : 1.0)
            .offset(y: lift(pressed: pressed))
            .animation(pressed ? Motion.press : Motion.hoverOut, value: pressed)
            .animation(hovering ? Motion.hoverIn : Motion.hoverOut, value: hovering)
            .onHover { hovering = $0 }
    }

    private var active: Bool { hovering && isEnabled }

    private var foreground: Color {
        guard isEnabled else { return Palette.muted.opacity(0.7) }
        return kind == .primary ? Palette.onAccent : Palette.ink
    }

    private var border: Color {
        guard isEnabled else { return Palette.line }
        if kind == .primary { return .clear }
        return active ? Palette.lineStrong : Palette.line
    }

    private func fill(pressed: Bool) -> Color {
        guard isEnabled else { return Palette.fill }
        switch kind {
        case .primary:
            if pressed { return Palette.accent.opacity(0.82) }
            return active ? Color(red: 0.36, green: 0.89, blue: 0.62) : Palette.accent
        case .secondary:
            if pressed { return Color.white.opacity(0.04) }
            return active ? Palette.fillHover : Palette.fill
        }
    }

    /// One point up on hover, one point down on press; colour alone under reduced motion.
    private func lift(pressed: Bool) -> CGFloat {
        guard isEnabled, !reduceMotion else { return 0 }
        if pressed { return 1 }
        return hovering ? -1 : 0
    }
}

/// A quiet text button for a side action such as refreshing a list.
struct LinkButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        LinkButtonBody(configuration: configuration)
    }
}

private struct LinkButtonBody: View {
    let configuration: ButtonStyleConfiguration

    @Environment(\.isEnabled) private var isEnabled
    @State private var hovering = false

    var body: some View {
        configuration.label
            .font(Typo.label)
            .foregroundStyle(hovering && isEnabled ? Palette.ink : Palette.muted)
            .opacity(configuration.isPressed ? 0.7 : 1.0)
            .contentShape(Rectangle())
            .animation(hovering ? Motion.hoverIn : Motion.hoverOut, value: hovering)
            .onHover { hovering = $0 }
    }
}

/// A selectable tile: the chosen one takes the accent border and a faint accent fill.
struct TileButtonStyle: ButtonStyle {
    let selected: Bool

    func makeBody(configuration: Configuration) -> some View {
        TileButtonBody(selected: selected, configuration: configuration)
    }
}

private struct TileButtonBody: View {
    let selected: Bool
    let configuration: ButtonStyleConfiguration

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var hovering = false

    var body: some View {
        let pressed = configuration.isPressed
        let shape = RoundedRectangle(cornerRadius: Radius.tile, style: .continuous)

        configuration.label
            .background(shape.fill(fill))
            .overlay(shape.strokeBorder(border, lineWidth: selected ? 1.5 : 1))
            .contentShape(shape)
            .scaleEffect(pressed && !reduceMotion ? 0.985 : 1.0)
            .animation(pressed ? Motion.press : Motion.hoverOut, value: pressed)
            .animation(hovering ? Motion.hoverIn : Motion.hoverOut, value: hovering)
            .animation(Motion.hoverIn, value: selected)
            .onHover { hovering = $0 }
    }

    private var fill: Color {
        if selected { return Palette.accent.opacity(0.12) }
        return hovering ? Palette.fillHover : Palette.fill
    }

    private var border: Color {
        if selected { return Palette.accent }
        return hovering ? Palette.lineStrong : Palette.line
    }
}

// MARK: - Notices

/// A boxed notice for something the person has to act on: a missing driver or an error.
struct Notice<Content: View, Action: View>: View {
    let symbol: String
    let tone: Color
    @ViewBuilder let content: () -> Content
    @ViewBuilder let action: () -> Action

    var body: some View {
        let shape = RoundedRectangle(cornerRadius: Radius.tile, style: .continuous)

        HStack(alignment: .center, spacing: Space.s3) {
            Image(systemName: symbol)
                .font(.system(size: 14, weight: .semibold))
                .foregroundStyle(tone)
                .accessibilityHidden(true)
            content()
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: Space.s2)
            action()
        }
        .padding(Space.s3)
        .background(shape.fill(tone.opacity(0.08)))
        .overlay(shape.strokeBorder(tone.opacity(0.30), lineWidth: 1))
    }
}

/// A button label that swaps to a confirmation and back without changing width.
struct SwapLabel: View {
    let rest: String
    let done: String
    let isDone: Bool

    var body: some View {
        ZStack {
            Text(rest).opacity(isDone ? 0 : 1)
            Text(done).opacity(isDone ? 1 : 0)
        }
        .animation(Motion.hoverIn, value: isDone)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(isDone ? done : rest)
    }
}
