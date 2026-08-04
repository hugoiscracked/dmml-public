"""
W04 — The Kernel Trick

Three-phase animation for Lecture 1 (SVMs).

  Phase 1: 1D scatter — two classes are interleaved (not linearly separable).
           A failed threshold shows that no single cut-point works.
  Phase 2: Feature map φ(x) = (x, x²) lifts each point to the parabola.
           The y-axis appears and dots animate from y=0 to y=x².
  Phase 3: A horizontal separator cleanly splits the classes in 2D.
           Two projection lines drop back to the x-axis to reveal the
           decision boundary in the original 1D space: x = ±1.

Data:  hand-picked 1D positions for maximum visual clarity.

Render:
  ../../env/bin/manim -pql kernel_trick.py KernelTrick
  ../../env/bin/manim -pqh kernel_trick.py KernelTrick
"""

from manim import *
import numpy as np

# ── Palette ───────────────────────────────────────────────────────────────────
C_BG  = "#0d1117"
C_C0  = "#ffa657"   # amber  – outer class (far from origin)
C_C1  = "#2dd4bf"   # teal   – inner class (near origin)
C_PHI = "#bc8cff"   # purple – parabola / feature map curve
C_SEP = "#ffffff"   # white  – 2D separator
C_PRJ = "#e3b341"   # gold   – projection / back-map lines
C_BAD = "#f78166"   # red    – failed threshold
C_DIM = "#8b949e"

_AX = dict(
    axis_config={"color": "#21262d", "stroke_width": 1.2, "include_ticks": False},
    tips=False,
)

# ── Data ──────────────────────────────────────────────────────────────────────
X0_1D = np.array([-2.2, -1.8, -1.6, -1.3,  1.3,  1.6,  1.9,  2.3])  # outer
X1_1D = np.array([-0.70, -0.35, -0.10,  0.20,  0.50,  0.75])          # inner
ALL_X = np.concatenate([X0_1D, X1_1D])
ALL_C = [C_C0] * len(X0_1D) + [C_C1] * len(X1_1D)

THRESHOLD = 1.0           # separator at y = x² = 1.0
THRESH_X  = np.sqrt(THRESHOLD)   # = 1.0  (boundary in 1D)

# ── Axes parameters ───────────────────────────────────────────────────────────
_X_LO, _X_HI = -2.6, 2.6
_Y_LO, _Y_HI = -0.2, 6.2
_X_LEN, _Y_LEN = 7.5, 5.2
# cy=0.438 puts the x-axis (y=0) at screen y ≈ -2.0, well-centred for Phase 1
_CENTER = [0.2, 0.438, 0]


# ── Scene ─────────────────────────────────────────────────────────────────────

class KernelTrick(Scene):
    def construct(self):
        self.camera.background_color = C_BG

        ax = Axes(
            x_range=[_X_LO, _X_HI, 1.0],
            y_range=[_Y_LO, _Y_HI, 1.0],
            x_length=_X_LEN, y_length=_Y_LEN,
            **_AX,
        ).move_to(_CENTER)

        # ════════════════════════════════════════════════════════════════════
        #  Phase 1 — 1D scatter: not linearly separable
        # ════════════════════════════════════════════════════════════════════
        self.play(Create(ax.x_axis), run_time=0.6)

        # Dots at y = 0
        dots = [
            Dot(ax.c2p(float(x), 0.0), radius=0.10, color=c)
            for x, c in zip(ALL_X, ALL_C)
        ]

        legend = VGroup(
            VGroup(
                Dot(radius=0.08, color=C_C0),
                Text("outer class", font_size=12, color=C_C0),
            ).arrange(RIGHT, buff=0.12),
            VGroup(
                Dot(radius=0.08, color=C_C1),
                Text("inner class", font_size=12, color=C_C1),
            ).arrange(RIGHT, buff=0.12),
        ).arrange(DOWN, buff=0.14, aligned_edge=LEFT).to_corner(UL, buff=0.35)

        self.play(
            LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.07),
            run_time=1.1,
        )
        self.play(FadeIn(legend), run_time=0.4)
        self.wait(0.3)

        title1 = (
            Text("1D: no single threshold separates the classes",
                 font_size=16, color=WHITE, weight=BOLD)
            .to_edge(UP, buff=0.35)
        )
        self.play(FadeIn(title1), run_time=0.4)

        # Show a failed threshold at x = 0.9
        fail_line = DashedLine(
            ax.c2p(0.9, -0.35), ax.c2p(0.9, 0.35),
            color=C_BAD, stroke_width=2.2, dash_length=0.11,
        )
        fail_lbl = (
            Text("\u2717", font_size=18, color=C_BAD)   # ✗
            .next_to(ax.c2p(0.9, 0.35), UP, buff=0.10)
        )
        self.play(Create(fail_line), FadeIn(fail_lbl), run_time=0.5)
        self.wait(0.3)
        # Move to x = -0.9 — also fails
        fail_line2 = DashedLine(
            ax.c2p(-0.9, -0.35), ax.c2p(-0.9, 0.35),
            color=C_BAD, stroke_width=2.2, dash_length=0.11,
        )
        fail_lbl2 = (
            Text("\u2717", font_size=18, color=C_BAD)
            .next_to(ax.c2p(-0.9, 0.35), UP, buff=0.10)
        )
        self.play(Create(fail_line2), FadeIn(fail_lbl2), run_time=0.5)
        self.wait(0.8)

        # ════════════════════════════════════════════════════════════════════
        #  Phase 2 — lift to 2D via φ(x) = (x, x²)
        # ════════════════════════════════════════════════════════════════════
        title2 = (
            Text("Feature map  \u03c6(x) = (x, x\u00b2)  lifts points to 2D",
                 font_size=16, color=WHITE, weight=BOLD)
            .to_edge(UP, buff=0.35)
        )
        self.play(
            FadeOut(fail_line), FadeOut(fail_lbl),
            FadeOut(fail_line2), FadeOut(fail_lbl2),
            ReplacementTransform(title1, title2),
            run_time=0.5,
        )

        # Reveal y-axis
        y_lbl = (
            Text("x\u00b2", font_size=14, color=C_DIM)
            .next_to(ax.y_axis, LEFT, buff=0.15)
        )
        self.play(Create(ax.y_axis), FadeIn(y_lbl), run_time=0.7)

        # Parabola curve  y = x²
        parabola = ax.plot(
            lambda x: x ** 2,
            x_range=[-2.45, 2.45, 0.05],
            color=C_PHI, stroke_width=1.8,
            use_smoothing=True,
        )
        parabola_lbl = (
            Text("y = x\u00b2", font_size=12, color=C_PHI)
            .next_to(ax.c2p(2.1, 2.1 ** 2), RIGHT, buff=0.12)
        )
        self.play(Create(parabola), FadeIn(parabola_lbl), run_time=0.9)
        self.wait(0.2)

        # Lift dots from y=0 to y=x²
        lift_anims = [
            dot.animate.move_to(ax.c2p(float(x), float(x) ** 2))
            for dot, x in zip(dots, ALL_X)
        ]
        self.play(
            LaggedStart(*lift_anims, lag_ratio=0.08),
            run_time=1.6,
        )
        self.wait(0.9)

        # ════════════════════════════════════════════════════════════════════
        #  Phase 3 — separate in 2D, project back to 1D
        # ════════════════════════════════════════════════════════════════════
        title3 = (
            Text("Linearly separable in 2D  \u2014  project boundary back to 1D",
                 font_size=16, color=WHITE, weight=BOLD)
            .to_edge(UP, buff=0.35)
        )
        self.play(ReplacementTransform(title2, title3), run_time=0.5)

        # Horizontal separator at y = THRESHOLD = 1.0
        sep_line = DashedLine(
            ax.c2p(_X_LO, THRESHOLD), ax.c2p(_X_HI, THRESHOLD),
            color=C_SEP, stroke_width=2.3, dash_length=0.14,
        )
        sep_lbl = (
            Text("separator  y = 1", font_size=12, color=C_DIM)
            .next_to(ax.c2p(_X_HI, THRESHOLD), RIGHT, buff=0.12)
        )
        self.play(Create(sep_line), FadeIn(sep_lbl), run_time=0.8)
        self.wait(0.5)

        # Light band: inner class region (0 ≤ y ≤ THRESHOLD) highlighted
        band = Polygon(
            ax.c2p(_X_LO, 0), ax.c2p(_X_HI, 0),
            ax.c2p(_X_HI, THRESHOLD), ax.c2p(_X_LO, THRESHOLD),
            fill_color=C_C1, fill_opacity=0.06, stroke_width=0,
        )
        self.play(FadeIn(band), run_time=0.5)
        self.wait(0.3)

        # Projection lines from x = ±THRESH_X on the separator down to the x-axis
        proj_r = DashedLine(
            ax.c2p( THRESH_X, THRESHOLD), ax.c2p( THRESH_X, 0),
            color=C_PRJ, stroke_width=2.0, dash_length=0.10,
        )
        proj_l = DashedLine(
            ax.c2p(-THRESH_X, THRESHOLD), ax.c2p(-THRESH_X, 0),
            color=C_PRJ, stroke_width=2.0, dash_length=0.10,
        )
        self.play(Create(proj_r), Create(proj_l), run_time=0.7)

        # Boundary markers on the x-axis
        mark_r = Dot(ax.c2p( THRESH_X, 0), radius=0.10, color=C_PRJ)
        mark_l = Dot(ax.c2p(-THRESH_X, 0), radius=0.10, color=C_PRJ)
        lbl_r  = (
            Text("+1", font_size=12, color=C_PRJ)
            .next_to(ax.c2p( THRESH_X, 0), DOWN, buff=0.15)
        )
        lbl_l  = (
            Text("\u22121", font_size=12, color=C_PRJ)   # −1
            .next_to(ax.c2p(-THRESH_X, 0), DOWN, buff=0.15)
        )
        self.play(
            GrowFromCenter(mark_r), GrowFromCenter(mark_l),
            FadeIn(lbl_r), FadeIn(lbl_l),
            run_time=0.6,
        )

        # Closing caption
        caption = (
            Text(
                "1D boundary: x < \u22121 or x > +1  \u2192  outer class   |"
                "   \u22121 \u2264 x \u2264 +1  \u2192  inner class",
                font_size=12, color=C_DIM,
            )
            .to_edge(DOWN, buff=0.35)
        )
        self.play(FadeIn(caption, shift=UP * 0.1), run_time=0.5)
        self.wait(2.5)
