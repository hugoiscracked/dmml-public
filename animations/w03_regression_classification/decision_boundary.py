"""
W03 — Decision Boundary

Two-phase animation covering classification intuition from Lecture 2.

  Phase 1 — Logistic Regression (linear boundary)
    Scatter of two-class moons data appears, then the logistic-regression
    decision line is drawn with half-plane shading and a "High Bias" note.

  Phase 2 — k-NN boundary sweep  (k = 1 → 5 → 15)
    LR elements fade out; the kNN boundary morphs from k=1 (jagged, high
    variance) through k=5 to k=15 (smooth, lower variance).  A closing
    caption frames the bias-variance message.

Render:
  ../../env/bin/manim -pql decision_boundary.py DecisionBoundary
  ../../env/bin/manim -pqh decision_boundary.py DecisionBoundary
"""

from manim import *
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# ── Palette ───────────────────────────────────────────────────────────────────
C_BG  = "#0d1117"
C_C0  = "#ffa657"   # amber – class 0
C_C1  = "#2dd4bf"   # teal  – class 1
C_LR  = "#ffffff"   # white – LR boundary line
C_K1  = "#f78166"   # red   – kNN k=1 (high variance)
C_K5  = "#e3b341"   # gold  – kNN k=5
C_K15 = "#3fb950"   # green – kNN k=15 (lower variance)
C_DIM = "#8b949e"

_AX = dict(
    axis_config={"color": "#21262d", "stroke_width": 1.2, "include_ticks": False},
    tips=False,
)

# ── Data ──────────────────────────────────────────────────────────────────────
X, y = make_moons(n_samples=120, noise=0.22, random_state=42)
X0 = X[y == 0]
X1 = X[y == 1]

PAD = 0.30
X_LO, X_HI = float(X[:, 0].min()) - PAD, float(X[:, 0].max()) + PAD
Y_LO, Y_HI = float(X[:, 1].min()) - PAD, float(X[:, 1].max()) + PAD

# ── Classifiers ───────────────────────────────────────────────────────────────
lr = LogisticRegression()
lr.fit(X, y)
_a, _b = float(lr.coef_[0, 0]), float(lr.coef_[0, 1])
_c = float(lr.intercept_[0])

K_SWEEP = [1, 5, 15]
knns = {k: KNeighborsClassifier(n_neighbors=k).fit(X, y) for k in K_SWEEP}

# ── Extract kNN decision boundaries via matplotlib contour ────────────────────
_RES = 200
_xx, _yy = np.meshgrid(
    np.linspace(X_LO, X_HI, _RES),
    np.linspace(Y_LO, Y_HI, _RES),
)
_grid = np.c_[_xx.ravel(), _yy.ravel()]
N_PTS = 400   # fixed resample length — required for Transform to work


def _get_boundary_verts(knn, n_pts=N_PTS):
    """Return (n_pts, 2) boundary vertices in data coords, resampled by arc length."""
    Z = knn.predict(_grid).reshape(_xx.shape).astype(float)
    fig, ax_m = plt.subplots()
    cs = ax_m.contour(_xx, _yy, Z, levels=[0.5])
    paths = cs.get_paths()
    plt.close(fig)
    verts = max(paths, key=lambda p: len(p.vertices)).vertices
    # Resample uniformly by arc length
    seg_len = np.linalg.norm(np.diff(verts, axis=0), axis=1)
    cumlen = np.concatenate([[0], np.cumsum(seg_len)])
    t_new = np.linspace(0, cumlen[-1], n_pts)
    return np.column_stack([
        np.interp(t_new, cumlen, verts[:, 0]),
        np.interp(t_new, cumlen, verts[:, 1]),
    ])


boundary_verts = {k: _get_boundary_verts(knns[k]) for k in K_SWEEP}


# ── Manim helpers ─────────────────────────────────────────────────────────────

def _make_dots(ax_m, data, color):
    return VGroup(*[
        Dot(ax_m.c2p(float(xi), float(yi)), radius=0.065, color=color)
        for xi, yi in data
    ])


def _make_lr_line(ax_m):
    """LR decision line clipped to the plot axes."""
    pt_l = (X_LO, np.clip(-((_a * X_LO + _c) / _b), Y_LO, Y_HI))
    pt_r = (X_HI, np.clip(-((_a * X_HI + _c) / _b), Y_LO, Y_HI))
    return Line(ax_m.c2p(*pt_l), ax_m.c2p(*pt_r), color=C_LR, stroke_width=2.3)


def _make_lr_shading(ax_m):
    """Two shaded half-planes separated by the LR boundary."""
    pt_l = (X_LO, np.clip(-((_a * X_LO + _c) / _b), Y_LO, Y_HI))
    pt_r = (X_HI, np.clip(-((_a * X_HI + _c) / _b), Y_LO, Y_HI))

    # TL corner → class 0 (amber); BL corner → class 1 (teal)
    sign_tl = _a * X_LO + _b * Y_HI + _c   # negative → class 0
    upper_col = C_C0 if sign_tl < 0 else C_C1
    lower_col = C_C1 if sign_tl < 0 else C_C0

    upper = Polygon(
        ax_m.c2p(X_LO, Y_HI), ax_m.c2p(X_HI, Y_HI),
        ax_m.c2p(*pt_r),       ax_m.c2p(*pt_l),
        fill_color=upper_col, fill_opacity=0.12, stroke_width=0,
    )
    lower = Polygon(
        ax_m.c2p(*pt_l),       ax_m.c2p(*pt_r),
        ax_m.c2p(X_HI, Y_LO), ax_m.c2p(X_LO, Y_LO),
        fill_color=lower_col, fill_opacity=0.12, stroke_width=0,
    )
    return upper, lower


def _make_knn_boundary(verts_2d, ax_m, color, sw=2.3):
    """Build a VMobject from resampled kNN boundary vertices."""
    pts = [ax_m.c2p(float(v[0]), float(v[1])) for v in verts_2d]
    mob = VMobject(color=color, stroke_width=sw, fill_opacity=0)
    mob.set_points_as_corners(pts)
    return mob


# ── Scene ─────────────────────────────────────────────────────────────────────

class DecisionBoundary(Scene):
    def construct(self):
        self.camera.background_color = C_BG

        # ════════════════════════════════════════════════════════════════════
        #  Phase 1 — Scatter + Logistic Regression boundary
        # ════════════════════════════════════════════════════════════════════
        ax = Axes(
            x_range=[X_LO, X_HI, 1.0],
            y_range=[Y_LO, Y_HI, 1.0],
            x_length=9.0, y_length=5.0,
            **_AX,
        ).move_to([0.2, 0, 0])

        self.play(Create(ax), run_time=0.6)

        # Scatter — both classes in one LaggedStart
        dots0 = _make_dots(ax, X0, C_C0)
        dots1 = _make_dots(ax, X1, C_C1)
        all_dots = [*dots0, *dots1]

        legend_sc = VGroup(
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
            LaggedStart(*[GrowFromCenter(d) for d in all_dots], lag_ratio=0.03),
            run_time=1.4,
        )
        self.play(FadeIn(legend_sc), run_time=0.4)
        self.wait(0.4)

        # LR decision boundary
        lr_line = _make_lr_line(ax)
        upper_sh, lower_sh = _make_lr_shading(ax)

        title1 = (
            Text("Logistic Regression — linear boundary",
                 font_size=16, color=WHITE, weight=BOLD)
            .to_edge(UP, buff=0.35)
        )
        lbl_bias = (
            Text("High Bias:\nlinear model misses\nthe curved structure",
                 font_size=12, color=C_DIM)
            .to_corner(UR, buff=0.35)
        )

        self.play(FadeIn(title1), run_time=0.4)
        self.play(
            FadeIn(upper_sh), FadeIn(lower_sh),
            Create(lr_line),
            run_time=0.9,
        )
        self.play(FadeIn(lbl_bias), run_time=0.4)
        self.wait(1.3)

        # ════════════════════════════════════════════════════════════════════
        #  Phase 2 — k-NN boundary sweep  k = 1 → 5 → 15
        # ════════════════════════════════════════════════════════════════════
        self.play(
            FadeOut(lr_line), FadeOut(upper_sh), FadeOut(lower_sh),
            FadeOut(title1), FadeOut(lbl_bias),
            run_time=0.5,
        )

        title2 = (
            Text("k-NN Classifier — decision boundary",
                 font_size=16, color=WHITE, weight=BOLD)
            .to_edge(UP, buff=0.35)
        )
        self.play(FadeIn(title2), run_time=0.4)

        K_INFO = {
            1:  ("k = 1",  "High Variance",   C_K1),
            5:  ("k = 5",  "Moderate",         C_K5),
            15: ("k = 15", "Lower Variance",   C_K15),
        }

        # k = 1
        k_bnd = _make_knn_boundary(boundary_verts[1], ax, C_K1)
        k_lbl = (
            VGroup(
                Text(K_INFO[1][0], font_size=15, color=C_K1, weight=BOLD),
                Text(K_INFO[1][1], font_size=12, color=C_K1),
            ).arrange(DOWN, buff=0.08, aligned_edge=LEFT)
            .to_corner(UR, buff=0.35)
        )
        self.play(Create(k_bnd), FadeIn(k_lbl), run_time=1.0)
        self.wait(0.9)

        # k = 1 → 5
        tgt_5 = _make_knn_boundary(boundary_verts[5], ax, C_K5)
        lbl_5 = (
            VGroup(
                Text(K_INFO[5][0], font_size=15, color=C_K5, weight=BOLD),
                Text(K_INFO[5][1], font_size=12, color=C_K5),
            ).arrange(DOWN, buff=0.08, aligned_edge=LEFT)
            .to_corner(UR, buff=0.35)
        )
        self.play(
            Transform(k_bnd, tgt_5),
            FadeOut(k_lbl), FadeIn(lbl_5),
            run_time=1.1,
        )
        self.wait(0.9)

        # k = 5 → 15
        tgt_15 = _make_knn_boundary(boundary_verts[15], ax, C_K15)
        lbl_15 = (
            VGroup(
                Text(K_INFO[15][0], font_size=15, color=C_K15, weight=BOLD),
                Text(K_INFO[15][1], font_size=12, color=C_K15),
            ).arrange(DOWN, buff=0.08, aligned_edge=LEFT)
            .to_corner(UR, buff=0.35)
        )
        self.play(
            Transform(k_bnd, tgt_15),
            FadeOut(lbl_5), FadeIn(lbl_15),
            run_time=1.1,
        )
        self.wait(0.9)

        # Closing caption
        caption = (
            Text(
                "small k  →  high variance     |     large k  →  high bias",
                font_size=13, color=C_DIM,
            )
            .to_edge(DOWN, buff=0.35)
        )
        self.play(FadeIn(caption, shift=UP * 0.1), run_time=0.5)
        self.wait(2.5)
