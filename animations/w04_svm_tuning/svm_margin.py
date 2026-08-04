"""
W04 — SVM Max-Margin Intuition

Three-phase animation for Lecture 1 (SVMs).

  Phase 1: Two-class scatter; three candidate decision boundaries flash in to
           show "many valid hyperplanes exist."
  Phase 2: The maximum-margin hyperplane settles in; a shaded margin band and
           two dashed margin gutters appear; a double-headed arrow labels the
           margin width 2/‖w‖.
  Phase 3: Support vectors are circled with gold rings; a caption closes.

Data:  make_blobs, centers [±1.4, 0], std=0.50, seed=14  → 26 points.
Model: SVC(kernel='linear', C=1e6) — hard-margin SVM.

Render:
  ../../env/bin/manim -pql svm_margin.py SVMMargin
  ../../env/bin/manim -pqh svm_margin.py SVMMargin
"""

from manim import *
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

# ── Palette ───────────────────────────────────────────────────────────────────
C_BG    = "#0d1117"
C_C0    = "#ffa657"   # amber  – class 0
C_C1    = "#2dd4bf"   # teal   – class 1
C_BNDRY = "#ffffff"   # white  – max-margin boundary
C_MARG  = "#58a6ff"   # blue   – margin gutters / band
C_CAND  = "#8b949e"   # grey   – candidate lines
C_SV    = "#e3b341"   # gold   – support vector rings
C_ARR   = "#3fb950"   # green  – margin-width arrow
C_DIM   = "#8b949e"

_AX = dict(
    axis_config={"color": "#21262d", "stroke_width": 1.2, "include_ticks": False},
    tips=False,
)

# ── Data & model ──────────────────────────────────────────────────────────────
X, y = make_blobs(n_samples=26, centers=[[-1.4, 0.0], [1.4, 0.0]],
                  cluster_std=0.50, random_state=14)
X0, X1 = X[y == 0], X[y == 1]

_svm = SVC(kernel="linear", C=1e6)
_svm.fit(X, y)
_w  = _svm.coef_[0]               # (2,)
_b  = float(_svm.intercept_[0])
_sv = _svm.support_vectors_       # (n_sv, 2)

# ── Axes bounds ───────────────────────────────────────────────────────────────
X_LO, X_HI = -2.90, 2.90
Y_LO, Y_HI = -1.40, 1.25

# ── Candidate separating lines for Phase 1: (slope, y-intercept) ──────────────
#   All confirmed to separate the two classes; none is the max-margin line.
CANDIDATES = [
    (1.0,  0.28),   # shallower slope, shifted toward class 1
    (2.8, -0.22),   # much steeper, biased toward class 0
    (1.5,  0.35),   # similar slope but too close to class 1
]

# ── Pre-compute arrow endpoints for the margin-width indicator ────────────────
_w_norm   = _w / np.linalg.norm(_w)
_half_m   = 1.0 / np.linalg.norm(_w)          # half of margin = 1/‖w‖
_mid      = np.array([0.0, -_b / _w[1]])       # midpoint on boundary at x=0
_arr_neg  = _mid - _half_m * _w_norm           # point on w·x+b = -1 (class-0 side)
_arr_pos  = _mid + _half_m * _w_norm           # point on w·x+b = +1 (class-1 side)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clip_line(slope, yint):
    """Return ((x1,y1), (x2,y2)) clipped to the axes viewport."""
    pts = []
    for xv in [X_LO, X_HI]:
        yv = slope * xv + yint
        if Y_LO <= yv <= Y_HI:
            pts.append((xv, yv))
    if abs(slope) > 1e-8:
        for yv in [Y_LO, Y_HI]:
            xv = (yv - yint) / slope
            if X_LO < xv < X_HI:
                pts.append((xv, yv))
    pts = sorted(set([(round(x, 6), round(y, 6)) for x, y in pts]))
    return pts[0], pts[-1]


def _make_line(ax_m, slope, yint, color, sw=2.0, dash=False):
    p1, p2 = _clip_line(slope, yint)
    if dash:
        return DashedLine(
            ax_m.c2p(*p1), ax_m.c2p(*p2),
            color=color, stroke_width=sw, dash_length=0.13,
        )
    return Line(ax_m.c2p(*p1), ax_m.c2p(*p2), color=color, stroke_width=sw)


def _boundary_params(offset=0.0):
    """Return (slope, yint) for the hyperplane w·x + b = offset."""
    return -_w[0] / _w[1], (offset - _b) / _w[1]


# ── Scene ─────────────────────────────────────────────────────────────────────

class SVMMargin(Scene):
    def construct(self):
        self.camera.background_color = C_BG

        ax = Axes(
            x_range=[X_LO, X_HI, 1.0],
            y_range=[Y_LO, Y_HI, 1.0],
            x_length=9.0, y_length=5.0,
            **_AX,
        ).move_to([0.2, 0, 0])
        self.play(Create(ax), run_time=0.6)

        # Scatter
        dots0 = VGroup(*[
            Dot(ax.c2p(float(xi), float(yi)), radius=0.065, color=C_C0)
            for xi, yi in X0
        ])
        dots1 = VGroup(*[
            Dot(ax.c2p(float(xi), float(yi)), radius=0.065, color=C_C1)
            for xi, yi in X1
        ])
        legend = VGroup(
            VGroup(
                Dot(radius=0.07, color=C_C0),
                Text("Class 0", font_size=12, color=C_C0),
            ).arrange(RIGHT, buff=0.12),
            VGroup(
                Dot(radius=0.07, color=C_C1),
                Text("Class 1", font_size=12, color=C_C1),
            ).arrange(RIGHT, buff=0.12),
        ).arrange(DOWN, buff=0.14, aligned_edge=LEFT).to_corner(UL, buff=0.35)

        self.play(
            LaggedStart(*[GrowFromCenter(d) for d in [*dots0, *dots1]], lag_ratio=0.04),
            run_time=1.2,
        )
        self.play(FadeIn(legend), run_time=0.4)
        self.wait(0.3)

        # ════════════════════════════════════════════════════════════════════
        #  Phase 1 — Candidate boundaries
        # ════════════════════════════════════════════════════════════════════
        title1 = (
            Text("Many valid boundaries separate the classes",
                 font_size=16, color=WHITE, weight=BOLD)
            .to_edge(UP, buff=0.35)
        )
        self.play(FadeIn(title1), run_time=0.4)

        cand_lines = []
        for slope, yint in CANDIDATES:
            ln = _make_line(ax, slope, yint, color=C_CAND, sw=1.8)
            cand_lines.append(ln)
            self.play(FadeIn(ln), run_time=0.45)
            self.wait(0.3)
        self.wait(0.7)

        # ════════════════════════════════════════════════════════════════════
        #  Phase 2 — Max-margin boundary + band + arrow
        # ════════════════════════════════════════════════════════════════════
        title2 = (
            Text("The maximum-margin hyperplane maximises the gap",
                 font_size=16, color=WHITE, weight=BOLD)
            .to_edge(UP, buff=0.35)
        )
        self.play(
            *[FadeOut(ln) for ln in cand_lines],
            ReplacementTransform(title1, title2),
            run_time=0.6,
        )

        # Decision boundary (solid white)
        sl_bnd, yi_bnd = _boundary_params(0)
        bnd_line = _make_line(ax, sl_bnd, yi_bnd, color=C_BNDRY, sw=2.4)
        self.play(Create(bnd_line), run_time=0.8)
        self.wait(0.2)

        # Margin band (light fill between the two margin lines)
        sl_m, yi_pos = _boundary_params(+1)
        _,    yi_neg = _boundary_params(-1)
        p1n, p2n = _clip_line(sl_m, yi_neg)   # class-0 side
        p1p, p2p = _clip_line(sl_m, yi_pos)   # class-1 side
        band = Polygon(
            ax.c2p(*p1n), ax.c2p(*p2n), ax.c2p(*p2p), ax.c2p(*p1p),
            fill_color=C_MARG, fill_opacity=0.09, stroke_width=0,
        )
        # Margin gutters (dashed blue)
        marg_neg = _make_line(ax, sl_m, yi_neg, color=C_MARG, sw=1.6, dash=True)
        marg_pos = _make_line(ax, sl_m, yi_pos, color=C_MARG, sw=1.6, dash=True)
        self.play(FadeIn(band), Create(marg_neg), Create(marg_pos), run_time=0.9)
        self.wait(0.3)

        # Double-headed arrow spanning the margin (perpendicular to boundary)
        arr = DoubleArrow(
            ax.c2p(float(_arr_neg[0]), float(_arr_neg[1])),
            ax.c2p(float(_arr_pos[0]), float(_arr_pos[1])),
            color=C_ARR, stroke_width=2.2, tip_length=0.20, buff=0,
        )
        mid_screen = ax.c2p(float(_mid[0]), float(_mid[1]))
        margin_lbl = (
            Text("2 / \u2016w\u2016", font_size=14, color=C_ARR)
            .move_to(mid_screen + RIGHT * 0.75)
        )
        self.play(Create(arr), FadeIn(margin_lbl), run_time=0.8)
        self.wait(1.3)

        # ════════════════════════════════════════════════════════════════════
        #  Phase 3 — Support vectors
        # ════════════════════════════════════════════════════════════════════
        title3 = (
            Text("Support vectors lie on the margin gutters — they define it",
                 font_size=16, color=WHITE, weight=BOLD)
            .to_edge(UP, buff=0.35)
        )
        self.play(ReplacementTransform(title2, title3), run_time=0.5)

        sv_rings = VGroup(*[
            Circle(radius=0.23, color=C_SV, stroke_width=2.8)
            .move_to(ax.c2p(float(sv[0]), float(sv[1])))
            for sv in _sv
        ])
        self.play(
            LaggedStart(*[Create(r) for r in sv_rings], lag_ratio=0.25),
            run_time=0.9,
        )
        sv_lbl = (
            Text("support vectors", font_size=13, color=C_SV)
            .to_corner(UR, buff=0.35)
        )
        self.play(FadeIn(sv_lbl), run_time=0.4)
        self.wait(2.5)
