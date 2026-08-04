"""
W02 — Time Series Decomposition
Builds a time series by stacking trend + seasonality + noise,
then animates each component being added to the observed panel.

Render with:
  ../../env/bin/manim -pql decomposition.py Decomposition
  ../../env/bin/manim -pqh decomposition.py Decomposition   (high quality)
"""

from manim import *
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d

# ── Palette ───────────────────────────────────────────────────────────────────
C_BG     = "#0d1117"
C_TREND  = "#58a6ff"   # blue
C_SEASON = "#3fb950"   # green
C_NOISE  = "#8b949e"   # grey
C_SUM    = "#ffa657"   # amber – observed series
C_DIM    = "#8b949e"

# ── Signal (fixed seed for reproducibility) ───────────────────────────────────
T0, T1  = 0.0, 4 * np.pi
N       = 350
rng     = np.random.default_rng(42)
t_vals  = np.linspace(T0, T1, N)

def _trend(t):    return 0.35 * t
def _season(t):   return 1.5 * np.sin(t) + 0.4 * np.sin(2 * t)

_noise_arr  = uniform_filter1d(rng.normal(0, 0.28, N), size=6)
_ps2_arr    = _trend(t_vals) + _season(t_vals)
_obs_arr    = _ps2_arr + _noise_arr

# Callable interpolants for ax.plot(lambda t: ...)
_noise_fn   = interp1d(t_vals, _noise_arr,  bounds_error=False, fill_value=0.0)
_ps2_fn     = interp1d(t_vals, _ps2_arr,    bounds_error=False, fill_value=0.0)
_obs_fn     = interp1d(t_vals, _obs_arr,    bounds_error=False, fill_value=0.0)

# Blended colour for the trend+season partial sum
C_BLEND  = interpolate_color(ManimColor(C_TREND), ManimColor(C_SUM), 0.45)


class Decomposition(Scene):
    def construct(self):
        self.camera.background_color = C_BG

        # ── Axes (shifted slightly right to give labels room on the left) ────
        AX_CFG = dict(
            x_range=[T0, T1, np.pi],
            y_range=[-3.5, 3.5, 1],
            x_length=9.0,
            y_length=1.15,
            axis_config={"color": "#21262d", "stroke_width": 1.2,
                         "include_ticks": False},
            tips=False,
        )
        y_pos   = [2.7, 0.9, -0.8, -2.6]
        row_info = [
            ("Trend",       C_TREND),
            ("Seasonality", C_SEASON),
            ("Noise",       C_NOISE),
            ("Observed",    C_SUM),
        ]

        axes_list, labels_list = [], []
        for yc, (name, col) in zip(y_pos, row_info):
            ax  = Axes(**AX_CFG).move_to([0.6, yc, 0])
            lbl = Text(name, font_size=15, color=col, weight=BOLD)\
                      .next_to(ax, LEFT, buff=0.3)
            axes_list.append(ax)
            labels_list.append(lbl)

        ax_t, ax_s, ax_n, ax_o = axes_list

        # ── Operator symbols (+, +, =) between rows ──────────────────────────
        op_x = labels_list[0].get_left()[0] - 0.15
        ops  = []
        for sym, i in zip(["+", "+", "="], range(3)):
            y_mid = (y_pos[i] + y_pos[i + 1]) / 2
            op = Text(sym, font_size=22, color=C_DIM, weight=BOLD)\
                     .move_to([op_x, y_mid, 0])
            ops.append(op)
        op_plus1, op_plus2, op_eq = ops

        # ── Step 0: axes + labels appear ─────────────────────────────────────
        self.play(
            LaggedStart(*[Create(ax) for ax in axes_list], lag_ratio=0.15),
            run_time=1.4,
        )
        self.play(
            LaggedStart(*[FadeIn(l) for l in labels_list], lag_ratio=0.12),
            run_time=0.8,
        )
        self.wait(0.3)

        # ── Step 1: draw Trend ────────────────────────────────────────────────
        trend_curve = ax_t.plot(
            lambda t: _trend(t),
            x_range=[T0, T1], color=C_TREND, stroke_width=2.5,
        )
        self.play(Create(trend_curve), run_time=1.6)
        self.wait(0.3)

        # ── Step 2: draw Seasonality ──────────────────────────────────────────
        season_curve = ax_s.plot(
            lambda t: _season(t),
            x_range=[T0, T1], color=C_SEASON, stroke_width=2.5,
        )
        self.play(Create(season_curve), run_time=1.6)
        self.wait(0.3)

        # ── Step 3: draw Noise ────────────────────────────────────────────────
        noise_curve = ax_n.plot(
            lambda t: float(_noise_fn(t)),
            x_range=[T0, T1], color=C_NOISE, stroke_width=1.8,
        )
        self.play(Create(noise_curve), run_time=1.6)
        self.wait(0.5)

        # ════════════════════════════════════════════════════════════════════
        #  Additive build-up: animate each component flowing into the
        #  observed panel and watch the partial sum evolve.
        # ════════════════════════════════════════════════════════════════════

        # Pre-build the three partial-sum curves (NOT added to scene yet)
        partial_trend = ax_o.plot(
            lambda t: _trend(t),
            x_range=[T0, T1], color=C_TREND, stroke_width=2.3,
        )
        partial_ts = ax_o.plot(
            lambda t: float(_ps2_fn(t)),
            x_range=[T0, T1], color=C_BLEND, stroke_width=2.5,
        )
        partial_full = ax_o.plot(
            lambda t: float(_obs_fn(t)),
            x_range=[T0, T1], color=C_SUM, stroke_width=2.8,
        )

        # ── 3a: Trend flows into observed panel ───────────────────────────────
        self.play(Indicate(trend_curve, color=WHITE, scale_factor=1.08),
                  run_time=0.45)
        # TransformFromCopy: copy of trend_curve morphs into partial_trend
        # → trend_curve stays in place, partial_trend appears in obs panel
        self.play(
            FadeIn(op_plus1, scale=1.5),
            TransformFromCopy(trend_curve, partial_trend),
            run_time=1.1,
        )
        self.wait(0.4)

        # ── 3b: Seasonality flows in, partial morphs to trend+season ─────────
        self.play(Indicate(season_curve, color=WHITE, scale_factor=1.08),
                  run_time=0.45)
        # TransformFromCopy brings a copy of season down; simultaneously
        # partial_trend morphs into partial_ts (the new accumulated curve)
        season_copy_target = partial_ts.copy()   # shape target for the flying copy
        self.play(
            FadeIn(op_plus2, scale=1.5),
            TransformFromCopy(season_curve, season_copy_target),
            Transform(partial_trend, partial_ts),
            run_time=1.2,
        )
        self.remove(season_copy_target)   # merged; keep only partial_trend (now = partial_ts)
        self.wait(0.4)

        # ── 3c: Noise flows in, partial morphs to full observed signal ────────
        self.play(Indicate(noise_curve, color=WHITE, scale_factor=1.08),
                  run_time=0.45)
        noise_copy_target = partial_full.copy()
        self.play(
            FadeIn(op_eq, scale=1.5),
            TransformFromCopy(noise_curve, noise_copy_target),
            Transform(partial_trend, partial_full),
            run_time=1.2,
        )
        self.remove(noise_copy_target)
        self.wait(0.3)

        # Brief flash on the completed observed curve
        self.play(Indicate(partial_trend, color=C_SUM, scale_factor=1.05),
                  run_time=0.6)
        self.wait(0.8)

        # ── Step 4: fade components, zoom observed ────────────────────────────
        components = VGroup(
            trend_curve, season_curve, noise_curve,
            ax_t, ax_s, ax_n,
            labels_list[0], labels_list[1], labels_list[2],
            op_plus1, op_plus2, op_eq,
        )
        self.play(FadeOut(components), run_time=0.9)

        obs_group = VGroup(ax_o, labels_list[3], partial_trend)
        self.play(obs_group.animate.move_to(RIGHT * 0.5).scale(1.4), run_time=1.1)

        title = Text("Observed Time Series", font_size=22,
                     color=C_SUM, weight=BOLD)
        title.next_to(obs_group, UP, buff=0.4)
        sub = Text("Goal: decompose back into trend, seasonality & noise",
                   font_size=14, color=C_DIM)
        sub.next_to(title, DOWN, buff=0.2)

        self.play(FadeIn(title, shift=DOWN * 0.2), run_time=0.5)
        self.play(FadeIn(sub,   shift=DOWN * 0.2), run_time=0.5)
        self.wait(2.0)
