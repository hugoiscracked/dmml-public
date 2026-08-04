"""
W02 — ARIMA Forecast Cone

Builds an AR(1) historical series, draws a point forecast that
mean-reverts toward zero, then reveals 80 % and 95 % confidence bands
growing from the forecast origin to make the widening-uncertainty concept
concrete.

Render with:
  ../../env/bin/manim -pql forecast_cone.py ForecastCone
  ../../env/bin/manim -pqh forecast_cone.py ForecastCone
"""

from manim import *
import numpy as np

# ── Palette ───────────────────────────────────────────────────────────────────
C_BG    = "#0d1117"
C_HIST  = "#ffa657"   # amber  – historical data
C_FCST  = "#58a6ff"   # blue   – point forecast line
C_BAND  = "#58a6ff"   # blue   – confidence bands (two opacities)
C_NOW   = "#8b949e"   # grey   – "now" boundary
C_DIM   = "#8b949e"

_AX = dict(
    axis_config={"color": "#21262d", "stroke_width": 1.2, "include_ticks": False},
    tips=False,
)

# ── Data  (seed 17 ends at y_T ≈ +1.0, clear mean-reversion toward 0) ────────
PHI    = 0.75
SIGMA  = 0.40
N_HIST = 22
H      = 20
T_NOW  = N_HIST - 1   # x-coordinate of the last observed value

rng      = np.random.default_rng(17)
eps_hist = rng.normal(0, SIGMA, N_HIST)
y_hist   = [0.0]
for e in eps_hist:
    y_hist.append(PHI * y_hist[-1] + e)
y_hist = np.array(y_hist[1:])          # shape (N_HIST,)
y_T    = float(y_hist[-1])             # ≈ +1.0

# ── Forecast mean & std for h = 0 … H ────────────────────────────────────────
# h = 0  →  y_T itself  (zero uncertainty)
# h > 0  →  PHI^h * y_T  ±  cumulative noise
h_arr     = np.arange(H + 1)
fcst_mean = PHI ** h_arr * y_T
fcst_std  = np.concatenate([[0.0],
    [SIGMA * np.sqrt((1 - PHI**(2*h)) / (1 - PHI**2)) for h in range(1, H+1)]
])


class ForecastCone(Scene):
    def construct(self):
        self.camera.background_color = C_BG

        # ── Axis spanning history + forecast ─────────────────────────────
        y_lo = min(float(y_hist.min()), float((fcst_mean - 1.96*fcst_std).min())) - 0.35
        y_hi = max(float(y_hist.max()), float((fcst_mean + 1.96*fcst_std).max())) + 0.35

        ax = Axes(
            x_range=[0, T_NOW + H, 5],
            y_range=[y_lo, y_hi, 1.0],
            x_length=11.0, y_length=3.8,
            **_AX,
        ).move_to([0.2, -0.2, 0])

        self.play(Create(ax), run_time=0.6)

        # ── Historical curve ──────────────────────────────────────────────
        hist_curve = ax.plot(
            lambda t: float(np.interp(t, np.arange(N_HIST), y_hist)),
            x_range=[0, T_NOW], color=C_HIST, stroke_width=2.3,
        )
        lbl_hist = Text("Historical", font_size=14, color=C_HIST, weight=BOLD)
        lbl_hist.move_to(ax.c2p(T_NOW // 2, y_hi) + UP * 0.38)

        self.play(Create(hist_curve), run_time=1.8)
        self.play(FadeIn(lbl_hist), run_time=0.4)
        self.wait(0.4)

        # ── "Now" boundary ────────────────────────────────────────────────
        now_x   = ax.c2p(T_NOW, 0)[0]
        now_bot = np.array([now_x, ax.c2p(0, y_lo)[1], 0])
        now_top = np.array([now_x, ax.c2p(0, y_hi)[1], 0])
        now_line = DashedLine(now_bot, now_top,
                              color=C_NOW, stroke_width=1.5, dash_length=0.12)
        lbl_now = Text("now", font_size=12, color=C_NOW)
        lbl_now.next_to(now_line, UP, buff=0.12)

        self.play(Create(now_line), FadeIn(lbl_now), run_time=0.5)
        self.wait(0.35)

        # ── Point forecast ────────────────────────────────────────────────
        fcst_curve = ax.plot(
            lambda t: float(np.interp(t - T_NOW, h_arr, fcst_mean)),
            x_range=[T_NOW, T_NOW + H], color=C_FCST, stroke_width=2.3,
        )
        lbl_fcst = Text("Forecast", font_size=14, color=C_FCST, weight=BOLD)
        lbl_fcst.move_to(ax.c2p(T_NOW + H // 2, y_hi) + UP * 0.38)

        self.play(Create(fcst_curve), run_time=1.2)
        self.play(FadeIn(lbl_fcst), run_time=0.4)
        self.wait(0.5)

        # ── Confidence bands  (grow left → right via UpdateFromAlphaFunc) ─
        def make_cone(alpha, ci_mult, fill_opacity):
            h_end    = max(1, int(np.round(alpha * H)))
            h_slice  = np.arange(h_end + 1)              # 0 … h_end
            t_slice  = T_NOW + h_slice
            means    = fcst_mean[:h_end + 1]
            stds     = fcst_std[:h_end + 1]
            upper    = means + ci_mult * stds
            lower    = means - ci_mult * stds
            up_pts   = [ax.c2p(float(t), float(u)) for t, u in zip(t_slice, upper)]
            dn_pts   = [ax.c2p(float(t), float(d)) for t, d in zip(t_slice, lower)]
            return Polygon(*up_pts, *dn_pts[::-1],
                           fill_color=C_BAND, fill_opacity=fill_opacity,
                           stroke_width=0)

        # 95 % CI  — wider, lighter
        cone_95 = make_cone(0.02, 1.96, 0.15)
        self.add(cone_95)
        self.play(
            UpdateFromAlphaFunc(
                cone_95, lambda m, a: m.become(make_cone(a, 1.96, 0.15))
            ),
            run_time=1.2,
        )

        # 80 % CI  — narrower, darker (draws on top)
        cone_80 = make_cone(0.02, 1.28, 0.28)
        self.add(cone_80)
        self.play(
            UpdateFromAlphaFunc(
                cone_80, lambda m, a: m.become(make_cone(a, 1.28, 0.28))
            ),
            run_time=0.9,
        )

        # ── Legend ────────────────────────────────────────────────────────
        legend = VGroup(
            Text("darker core: 80% CI",  font_size=11, color=C_BAND),
            Text("lighter rim:  95% CI", font_size=11, color=C_BAND),
        ).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
        legend.to_corner(UR, buff=0.35)

        self.play(FadeIn(legend), run_time=0.4)

        # ── Closing caption ───────────────────────────────────────────────
        caption = Text(
            "Uncertainty grows with forecast horizon",
            font_size=14, color=C_DIM,
        ).next_to(ax, DOWN, buff=0.3)

        self.play(FadeIn(caption, shift=UP * 0.1), run_time=0.5)
        self.wait(2.5)
