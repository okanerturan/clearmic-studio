import SwiftUI

/// The level history of one lane as a single path, so 30 updates a second redraw
/// two shapes instead of one view per bar. The values are never animated.
struct LevelBars: Shape {
    let samples: [Float]
    let growsUp: Bool

    func path(in rect: CGRect) -> Path {
        var path = Path()
        let count = samples.count
        guard count > 0, rect.width > 0 else { return path }

        let gap: CGFloat = 3
        let width = max(1, (rect.width - gap * CGFloat(count - 1)) / CGFloat(count))
        let corner = min(width / 2, 2)
        for (index, sample) in samples.enumerated() {
            let level = CGFloat(min(max(sample, 0), 1))
            let height = max(2, rect.height * level)
            let x = rect.minX + CGFloat(index) * (width + gap)
            let y = growsUp ? rect.maxY - height : rect.minY
            path.addRoundedRect(
                in: CGRect(x: x, y: y, width: width, height: height),
                cornerSize: CGSize(width: corner, height: corner)
            )
        }
        return path
    }
}

/// What the microphone hears above the centre line and what leaves the app below it,
/// on one time axis, with the engine's live figures underneath.
struct SignalPanel: View {
    @ObservedObject var bridge: CoreAudioBridge
    @ObservedObject var daemon: DaemonClient
    let bypassed: Bool
    /// The chosen mode's delay, shown until the engine reports a measured one.
    let idleDelay: String

    private let laneLabelWidth: CGFloat = 132

    var body: some View {
        let shape = RoundedRectangle(cornerRadius: Radius.panel, style: .continuous)

        VStack(spacing: 0) {
            VStack(spacing: 2) {
                lane(
                    name: "Microphone",
                    level: bridge.inLevel,
                    samples: bridge.inWave,
                    color: Palette.inputBars,
                    growsUp: true
                )
                lane(
                    name: bypassed ? "Output, unprocessed" : "Cleaned output",
                    level: bridge.outLevel,
                    samples: bridge.outWave,
                    color: bypassed ? Palette.warning : Palette.accent,
                    growsUp: false
                )
            }
            .padding(Space.s4)

            Rectangle()
                .fill(Palette.line)
                .frame(height: 1)

            HStack(spacing: Space.s4) {
                figure("Delay", bridge.isRunning ? String(format: "%.0f ms", daemon.e2eMs) : "about \(idleDelay)")
                figure("Processing", bridge.isRunning ? String(format: "%.1f of 20 ms", daemon.computeMs) : "Idle")
                figure("GPU memory", daemon.mlxMemMB > 0 ? String(format: "%.0f MB", daemon.mlxMemMB) : "Not loaded")
            }
            .padding(.horizontal, Space.s4)
            .padding(.vertical, Space.s3)
        }
        .background(shape.fill(Palette.surface))
        .overlay(shape.strokeBorder(Palette.line, lineWidth: 1))
    }

    private func lane(name: String, level: Float, samples: [Float], color: Color, growsUp: Bool) -> some View {
        HStack(spacing: Space.s3) {
            VStack(alignment: .leading, spacing: 2) {
                Text(name)
                    .font(Typo.label)
                    .foregroundStyle(Palette.ink)
                Text(String(format: "%.0f%%", level * 100))
                    .font(Typo.figure)
                    .foregroundStyle(Palette.muted)
            }
            .frame(width: laneLabelWidth, alignment: .leading)
            .accessibilityElement(children: .combine)

            LevelBars(samples: samples, growsUp: growsUp)
                .fill(color)
                .frame(minHeight: 40, maxHeight: 84)
                .accessibilityHidden(true)
        }
    }

    private func figure(_ label: String, _ value: String) -> some View {
        HStack(spacing: 6) {
            Text(label)
                .font(Typo.caption)
                .foregroundStyle(Palette.muted)
            Text(value)
                .font(Typo.figure)
                .foregroundStyle(Palette.ink)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .combine)
    }
}
