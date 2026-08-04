"""
W02 — Stationarity & First Differencing

Shows a drifting random walk, applies first-differencing, then compares
the ACF before and after to demonstrate the stationarity transformation.

Render with:
  ../../env/bin/manim -pql stationarity.py Stationarity
  ../../env/bin/manim -pqh stationarity.py Stationarity
"""

from manim import *
import numpy as np

# ── Palette ───────────────────────────────────────────────────────────────────
C_BG   = "#0d1117"
C_ORIG = "#ffa657"   # amber  – original random walk
C_DIFF = "#2dd4bf"   # teal   – differenced series
C_DIM  = "#8b949e"   # grey
C_WARN = "#f78166"   # red-orange – non-stationary
C_OK   = "#3fb950"   # green – stationary

# ── Signal (fixed seed) ───────────────────────────────────────────────────────
N   = 300
rng = np.random.default_rng(7)

increments = rng.normal(0.02, 0.22, N)
rw   = np.cumsum(increments)
rw  -= rw[0]               # start at zero

diff = np.diff(rw)         # first difference, length N-1

# ── ACF ───────────────────────────────────────────────────────────────────────
NLAGS = 20

def compute_acf(x, nlags=NLAGS):
    x  = x - x.mean()
    n  = len(x)
    c0 = float(np.dot(x, x)) / n
    return np.array(
        [float(np.dot(x[: n - k], x[k:])) / (n * c0) for k in range(nlags + 1)]
    )

acf_rw   = compute_acf(rw)
acf_diff = compute_acf(diff)
conf95   = 1.96 / np.sqrt(N)

# ── Shared axis style ─────────────────────────────────────────────────────────
_AX = dict(
    axis_config={"color": "#21262d", "stroke_width": 1.2, "include_ticks": False},
    tips=False,
)


class Stationarity(Scene):
    def construct(self):
        self.camera.background_color = C_BG

        # ════════════════════════════════════════════════════════════════════
        #  Phase 1 — Random Walk (full-width, centred)
        # ════════════════════════════════════════════════════════════════════
        ax_rw = Axes(
            x_range=[0, N - 1, 50],
            y_range=[float(rw.min()) - 0.3, float(rw.max()) + 0.3, 1.0],
            x_length=9.0, y_length=2.2,
            **_AX,
        ).move_to([0.6, 0.5, 0])

        lbl_rw = (
            Text("Random Walk", font_size=15, color=C_ORIG, weight=BOLD)
            .next_to(ax_rw, LEFT, buff=0.3)
        )

        rw_curve = ax_rw.plot(
            lambda t: float(np.interp(t, np.arange(N), rw)),
            x_range=[0, N - 1], color=C_ORIG, stroke_width=2.0,
        )

        tag_ns = (
            Text("non-stationary", font_size=13, color=C_WARN, weight=BOLD)
            .next_to(ax_rw, DOWN, buff=0.18)
        )

        self.play(Create(ax_rw), FadeIn(lbl_rw), run_time=0.8)
        self.play(Create(rw_curve), run_time=2.0)
        self.play(FadeIn(tag_ns, shift=UP * 0.1), run_time=0.5)
        self.wait(0.8)

        # ════════════════════════════════════════════════════════════════════
        #  Phase 2 — First Differencing
        # ════════════════════════════════════════════════════════════════════

        # Slide random walk to top half
        rw_group = VGroup(ax_rw, lbl_rw, rw_curve, tag_ns)
        self.play(rw_group.animate.move_to([0.6, 2.2, 0]).scale(0.80), run_time=1.0)

        # Differencing formula between the two rows
        formula = Text(
            "First difference:   \u0394y[t]  =  y[t] \u2212 y[t\u22121]",
            font_size=17, color=C_DIM,
        ).move_to([0, 0.1, 0])
        self.play(FadeIn(formula), run_time=0.5)
        self.wait(0.4)

        # Differenced series axis
        ax_diff = Axes(
            x_range=[0, N - 2, 50],
            y_range=[float(diff.min()) - 0.05, float(diff.max()) + 0.05, 0.5],
            x_length=9.0, y_length=2.2,
            **_AX,
        ).move_to([0.6, -2.2, 0])

        lbl_diff = (
            Text("First Difference", font_size=15, color=C_DIFF, weight=BOLD)
            .next_to(ax_diff, LEFT, buff=0.3)
        )

        diff_curve = ax_diff.plot(
            lambda t: float(np.interp(t, np.arange(N - 1), diff)),
            x_range=[0, N - 2], color=C_DIFF, stroke_width=2.0,
        )

        tag_ok = (
            Text("stationary", font_size=13, color=C_OK, weight=BOLD)
            .next_to(ax_diff, DOWN, buff=0.18)
        )

        self.play(FadeOut(formula), Create(ax_diff), FadeIn(lbl_diff), run_time=0.8)
        self.play(Create(diff_curve), run_time=2.0)
        self.play(FadeIn(tag_ok, shift=UP * 0.1), run_time=0.5)
        self.wait(1.0)

        # ════════════════════════════════════════════════════════════════════
        #  Phase 3 — ACF comparison
        # ════════════════════════════════════════════════════════════════════
        diff_group = VGroup(ax_diff, lbl_diff, diff_curve, tag_ok)
        self.play(FadeOut(rw_group), FadeOut(diff_group), run_time=0.7)

        acf_title = (
            Text("Autocorrelation Function (ACF)", font_size=20, color=C_DIM, weight=BOLD)
            .to_edge(UP, buff=0.35)
        )
        self.play(FadeIn(acf_title), run_time=0.4)

        # Two ACF axes, side by side
        _ACF_AX = dict(
            x_range=[0, NLAGS, 5],
            y_range=[-0.35, 1.05, 0.5],
            x_length=5.5, y_length=2.6,
            **_AX,
        )
        ax_acf_rw   = Axes(**_ACF_AX).move_to([-3.1, -0.5, 0])
        ax_acf_diff = Axes(**_ACF_AX).move_to([ 3.1, -0.5, 0])

        lbl_acf_rw = (
            Text("Random Walk", font_size=14, color=C_ORIG, weight=BOLD)
            .next_to(ax_acf_rw, UP, buff=0.2)
        )
        lbl_acf_diff = (
            Text("First Difference", font_size=14, color=C_DIFF, weight=BOLD)
            .next_to(ax_acf_diff, UP, buff=0.2)
        )

        self.play(
            Create(ax_acf_rw), Create(ax_acf_diff),
            FadeIn(lbl_acf_rw), FadeIn(lbl_acf_diff),
            run_time=0.8,
        )

        def make_acf_bars(ax, acf_vals, col):
            bars = VGroup()
            for k in range(1, NLAGS + 1):
                v       = float(acf_vals[k])
                bar_col = col if abs(v) > conf95 else C_DIM
                bars.add(Line(ax.c2p(k, 0), ax.c2p(k, v),
                              color=bar_col, stroke_width=3))
            thresh = VGroup(
                DashedLine(ax.c2p(0,  conf95), ax.c2p(NLAGS,  conf95),
                           color=C_DIM, stroke_width=1.0, dash_length=0.07),
                DashedLine(ax.c2p(0, -conf95), ax.c2p(NLAGS, -conf95),
                           color=C_DIM, stroke_width=1.0, dash_length=0.07),
            )
            return bars, thresh

        bars_rw,   thresh_rw   = make_acf_bars(ax_acf_rw,   acf_rw,   C_ORIG)
        bars_diff, thresh_diff = make_acf_bars(ax_acf_diff, acf_diff, C_DIFF)

        # Confidence threshold lines first, then animated bars
        self.play(Create(thresh_rw), Create(thresh_diff), run_time=0.5)
        self.play(
            LaggedStart(*[Create(b) for b in bars_rw],   lag_ratio=0.06),
            LaggedStart(*[Create(b) for b in bars_diff], lag_ratio=0.06),
            run_time=2.0,
        )

        ann_rw = (
            Text("slowly decaying  \u2192  non-stationary", font_size=11, color=C_WARN)
            .next_to(ax_acf_rw, DOWN, buff=0.25)
        )
        ann_diff = (
            Text("drops to zero  \u2192  stationary", font_size=11, color=C_OK)
            .next_to(ax_acf_diff, DOWN, buff=0.25)
        )
        self.play(FadeIn(ann_rw), FadeIn(ann_diff), run_time=0.6)
        self.wait(2.5)
