"""
W03 — Bias-Variance Tradeoff

Three-phase animation:
  1. Noisy observations appear, then the hidden true function is revealed.
  2. Three polynomial fits cycle through:
       degree-1  (underfitting / high bias)  →
       degree-4  (good fit)                 →
       degree-12 (overfitting / high variance)
  3. Error U-curve: Bias², Variance, and Total Error drawn in sequence,
     with a sweet-spot marker at the minimum.

Render:
  ../../env/bin/manim -pql bias_variance.py BiasVariance
  ../../env/bin/manim -pqh bias_variance.py BiasVariance
"""

from manim import *
import numpy as np
from scipy.interpolate import interp1d

# ── Palette ───────────────────────────────────────────────────────────────────
C_BG    = "#0d1117"
C_TRUE  = "#58a6ff"   # blue   – ground truth
C_DATA  = "#ffa657"   # amber  – noisy observations
C_FIT1  = "#f78166"   # red    – degree-1  (underfit)
C_FIT4  = "#3fb950"   # green  – degree-4  (good fit)
C_FIT12 = "#bc8cff"   # purple – degree-12 (overfit)
C_BIAS  = "#58a6ff"   # blue   – bias² curve
C_VAR   = "#f78166"   # red    – variance curve
C_TOT   = "#e3b341"   # gold   – total-error curve
C_SWEET = "#3fb950"   # green  – sweet-spot marker
C_DIM   = "#8b949e"

_AX = dict(
    axis_config={"color": "#21262d", "stroke_width": 1.2, "include_ticks": False},
    tips=False,
)

# ── Noisy data  (sin + Gaussian noise) ───────────────────────────────────────
N      = 28
rng    = np.random.default_rng(5)
x_data = np.linspace(0.3, 2 * np.pi - 0.3, N)
y_data = np.sin(x_data) + rng.normal(0, 0.38, N)
X_MIN, X_MAX = float(x_data[0]), float(x_data[-1])

# ── Polynomial fits ───────────────────────────────────────────────────────────
fit1  = np.poly1d(np.polyfit(x_data, y_data,  1))
fit4  = np.poly1d(np.polyfit(x_data, y_data,  4))
fit12 = np.poly1d(np.polyfit(x_data, y_data, 12))

# ── U-curve data  (conceptual — illustrates the general principle) ────────────
_cx  = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=float)
_b2  = np.array([0.76, 0.53, 0.35, 0.22, 0.14, 0.09, 0.06, 0.04, 0.03, 0.02, 0.015, 0.01])
_vr  = np.array([0.01, 0.02, 0.05, 0.09, 0.17, 0.27, 0.40, 0.56, 0.74, 0.95, 1.18,  1.43])
NF   = 0.05   # irreducible noise floor
_tot = _b2 + _vr + NF

b2_fn  = interp1d(_cx, _b2,  kind='cubic', fill_value='extrapolate')
vr_fn  = interp1d(_cx, _vr,  kind='cubic', fill_value='extrapolate')
tot_fn = interp1d(_cx, _tot, kind='cubic', fill_value='extrapolate')

x_sweet = float(_cx[np.argmin(_tot)])   # = 4.0


class BiasVariance(Scene):
    def construct(self):
        self.camera.background_color = C_BG

        # ════════════════════════════════════════════════════════════════════
        #  Phase 1 — Noisy observations + hidden ground truth
        # ════════════════════════════════════════════════════════════════════
        ax = Axes(
            x_range=[0, 2 * np.pi, np.pi],
            y_range=[-2.2, 2.2, 1.0],
            x_length=9.5, y_length=4.0,
            **_AX,
        ).move_to([0.3, 0, 0])

        self.play(Create(ax), run_time=0.6)

        # Noisy data points
        dots = VGroup(*[
            Dot(ax.c2p(float(xi), float(yi)), radius=0.07, color=C_DATA)
            for xi, yi in zip(x_data, y_data)
        ])
        lbl_data = (
            Text("Noisy observations", font_size=13, color=C_DATA)
            .next_to(ax, DOWN, buff=0.25)
        )
        self.play(
            LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.05),
            run_time=1.2,
        )
        self.play(FadeIn(lbl_data), run_time=0.4)
        self.wait(0.5)

        # Reveal the true function
        true_curve = ax.plot(
            lambda x: float(np.sin(x)),
            x_range=[X_MIN, X_MAX], color=C_TRUE, stroke_width=2.5,
        )
        lbl_true = (
            Text("True function", font_size=13, color=C_TRUE)
            .to_corner(UL, buff=0.35)
        )
        self.play(Create(true_curve), run_time=1.2)
        self.play(FadeIn(lbl_true), run_time=0.4)
        self.wait(0.8)

        # ════════════════════════════════════════════════════════════════════
        #  Phase 2 — Three polynomial fits
        # ════════════════════════════════════════════════════════════════════

        # Dim the ground truth so the fit stands out
        self.play(true_curve.animate.set_stroke(opacity=0.22), run_time=0.4)

        # Build all three curves up-front (only fit_curve enters the scene)
        fit_curve = ax.plot(
            lambda x: float(fit1(x)),
            x_range=[X_MIN, X_MAX], color=C_FIT1, stroke_width=2.5,
        )
        target_4 = ax.plot(
            lambda x: float(fit4(x)),
            x_range=[X_MIN, X_MAX], color=C_FIT4, stroke_width=2.5,
        )
        target_12 = ax.plot(
            lambda x: float(fit12(x)),
            x_range=[X_MIN, X_MAX], color=C_FIT12, stroke_width=2.5,
        )

        def fit_label(line1, line2, color):
            return VGroup(
                Text(line1, font_size=16, color=color, weight=BOLD),
                Text(line2, font_size=13, color=color),
            ).arrange(DOWN, buff=0.08, aligned_edge=LEFT).to_corner(UR, buff=0.35)

        lbl_1  = fit_label("Degree 1",  "High Bias",     C_FIT1)
        lbl_4  = fit_label("Degree 4",  "Good Fit",      C_FIT4)
        lbl_12 = fit_label("Degree 12", "High Variance", C_FIT12)

        # Degree-1 fit
        self.play(Create(fit_curve), FadeIn(lbl_1), run_time=0.9)
        self.wait(0.8)

        # Morph → degree-4  (Transform interpolates shape AND colour)
        self.play(
            Transform(fit_curve, target_4),
            FadeOut(lbl_1), FadeIn(lbl_4),
            run_time=1.0,
        )
        self.wait(0.8)

        # Morph → degree-12
        self.play(
            Transform(fit_curve, target_12),
            FadeOut(lbl_4), FadeIn(lbl_12),
            run_time=1.0,
        )
        self.wait(0.8)

        # ════════════════════════════════════════════════════════════════════
        #  Phase 3 — Error U-curve
        # ════════════════════════════════════════════════════════════════════
        phase12 = VGroup(ax, true_curve, dots, fit_curve,
                         lbl_true, lbl_data, lbl_12)
        self.play(FadeOut(phase12), run_time=0.7)

        title3 = (
            Text("Bias\u00b2 + Variance = Total Error",
                 font_size=19, color=WHITE, weight=BOLD)
            .to_edge(UP, buff=0.35)
        )
        self.play(FadeIn(title3), run_time=0.4)

        # New axes
        ax3 = Axes(
            x_range=[1, 12, 3],
            y_range=[0, 1.55, 0.5],
            x_length=9.0, y_length=3.4,
            **_AX,
        ).move_to([0.2, -0.5, 0])

        x_lbl = (
            Text("Model Complexity  \u2192", font_size=13, color=C_DIM)
            .next_to(ax3, DOWN, buff=0.2)
        )
        y_lbl = (
            Text("Error", font_size=13, color=C_DIM)
            .next_to(ax3, LEFT, buff=0.25)
            .rotate(PI / 2)
        )
        # Legend — shown before curves so viewers are pre-keyed
        legend = VGroup(
            Text("Bias\u00b2",      font_size=13, color=C_BIAS, weight=BOLD),
            Text("Variance",        font_size=13, color=C_VAR,  weight=BOLD),
            Text("Total Error",     font_size=13, color=C_TOT,  weight=BOLD),
            Text("noise floor",     font_size=11, color=C_DIM),
        ).arrange(DOWN, buff=0.18, aligned_edge=LEFT)
        legend.to_corner(UR, buff=0.35)

        self.play(Create(ax3), FadeIn(x_lbl), FadeIn(y_lbl), FadeIn(legend), run_time=0.7)

        # Noise floor
        noise_line = DashedLine(
            ax3.c2p(1, NF), ax3.c2p(12, NF),
            color=C_DIM, stroke_width=1.0, dash_length=0.10,
        )
        self.play(Create(noise_line), run_time=0.4)

        # ── Bias² curve  (decreasing) ─────────────────────────────────────
        bias_curve = ax3.plot(
            lambda x: float(b2_fn(x)),
            x_range=[1, 12], color=C_BIAS, stroke_width=2.5,
        )
        self.play(Create(bias_curve), run_time=0.9)
        self.wait(0.25)

        # ── Variance curve  (increasing) ──────────────────────────────────
        var_curve = ax3.plot(
            lambda x: float(vr_fn(x)),
            x_range=[1, 12], color=C_VAR, stroke_width=2.5,
        )
        self.play(Create(var_curve), run_time=0.9)
        self.wait(0.25)

        # ── Total-error curve  (U-shaped) ──────────────────────────────────
        tot_curve = ax3.plot(
            lambda x: float(tot_fn(x)),
            x_range=[1, 12], color=C_TOT, stroke_width=2.5,
        )
        self.play(Create(tot_curve), run_time=0.9)
        self.wait(0.4)

        # ── Sweet-spot marker ─────────────────────────────────────────────
        sw_y = float(tot_fn(x_sweet))
        sweet_line = DashedLine(
            ax3.c2p(x_sweet, 0), ax3.c2p(x_sweet, sw_y),
            color=C_SWEET, stroke_width=1.8, dash_length=0.10,
        )
        sweet_dot = Dot(ax3.c2p(x_sweet, sw_y), color=C_SWEET, radius=0.09)
        sweet_lbl = (
            Text("sweet spot", font_size=11, color=C_SWEET)
            .next_to(sweet_dot, UR, buff=0.10)
        )
        self.play(
            Create(sweet_line), GrowFromCenter(sweet_dot), FadeIn(sweet_lbl),
            run_time=0.6,
        )

        # ── Region annotations ────────────────────────────────────────────
        reg_left = (
            Text("Underfitting", font_size=11, color=C_DIM)
            .move_to(ax3.c2p(2.0, 1.42))
        )
        reg_right = (
            Text("Overfitting", font_size=11, color=C_DIM)
            .move_to(ax3.c2p(10.5, 1.42))
        )
        self.play(FadeIn(reg_left), FadeIn(reg_right), run_time=0.5)
        self.wait(2.5)
