"""
W06 — Gradient Boosting: Sequential Residual Fitting

Three-phase animation for Lecture 1 (Boosting concepts).

  Phase 1: Initial model F₀ = mean(y); large residuals highlighted in red.
  Phase 2: Three boosting rounds — each round flashes the weak tree h_k
           that fits the current residuals, then transforms the ensemble
           prediction to F_k; residuals visibly shrink each round; a
           running formula accumulates on the right.
  Phase 3: AdaBoost vs Gradient Boosting schematic comparison — left panel
           shows re-weighting of misclassified points; right panel shows
           the residual-fitting loop in miniature.

Render:
  ../../env/bin/manim -pql gradient_boosting.py GradientBoosting
  ../../env/bin/manim -pqh gradient_boosting.py GradientBoosting
"""

from manim import *
import numpy as np
from sklearn.tree import DecisionTreeRegressor

# ── Palette ───────────────────────────────────────────────────────────────────
C_BG   = "#0d1117"
C_WHITE = "#ffffff"
C_DIM  = "#8b949e"
C_DATA = "#58a6ff"   # blue  – data points
C_F0   = "#8b949e"   # grey  – F₀ mean line
C_RES  = "#f78166"   # red   – residuals
C_TREE = "#e3b341"   # gold  – weak tree fit
C_C0   = "#ffa657"   # amber
C_C1   = "#2dd4bf"   # teal
C_GREEN = "#3fb950"  # green

ROUND_COLS = ["#ffa657", "#2dd4bf", "#3fb950"]   # colour per boosting round

_AX = dict(
    axis_config={"color": "#21262d", "stroke_width": 1.2, "include_ticks": False},
    tips=False,
)

# ── Pre-compute boosting ──────────────────────────────────────────────────────
np.random.seed(7)
_N     = 14
_X     = np.linspace(0.4, 5.9, _N)
_Y     = np.sin(_X) + np.random.default_rng(7).normal(0, 0.22, _N)
_YMEAN = float(_Y.mean())

ETA    = 0.65
ROUNDS = 3

_F_pts = [np.full(_N, _YMEAN)]   # ensemble predictions on training pts
_trees = []
for _ in range(ROUNDS):
    r = _Y - _F_pts[-1]
    t = DecisionTreeRegressor(max_depth=2, random_state=42)
    t.fit(_X.reshape(-1, 1), r)
    _trees.append(t)
    _F_pts.append(_F_pts[-1] + ETA * t.predict(_X.reshape(-1, 1)))

_SUBS  = ["\u2081", "\u2082", "\u2083"]   # ₁ ₂ ₃
_F_LBL = [
    "F\u2080(x)  =  mean(y)",
    "F\u2081(x)  =  F\u2080  +  \u03b7\u00b7h\u2081",
    "F\u2082(x)  =  F\u2081  +  \u03b7\u00b7h\u2082",
    "F\u2083(x)  =  F\u2082  +  \u03b7\u00b7h\u2083",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pred_func(k):
    """Return a callable for the stage-k ensemble prediction."""
    m = _YMEAN
    ts = _trees[:k]
    def f(x):
        v = m
        for tr in ts:
            v += ETA * float(tr.predict([[x]])[0])
        return v
    return f


def _res_vgroup(ax, stage, min_len=0.04):
    """VGroup of red DashedLines for residuals at the given stage."""
    pred = _F_pts[stage]
    grp  = VGroup()
    for xi, yi, pi in zip(_X, _Y, pred):
        if abs(yi - pi) > min_len:
            grp.add(DashedLine(
                ax.c2p(float(xi), float(pi)),
                ax.c2p(float(xi), float(yi)),
                color=C_RES, stroke_width=1.8, dash_length=0.09,
            ))
    return grp


# ── Scene ─────────────────────────────────────────────────────────────────────

class GradientBoosting(Scene):
    def construct(self):
        self.camera.background_color = C_BG

        ax = Axes(
            x_range=[0, 6.3, 1],
            y_range=[-1.6, 1.6, 0.5],
            x_length=8.2, y_length=4.0,
            **_AX,
        ).move_to([-0.5, 0.5, 0])

        # ════════════════════════════════════════════════════════════════════
        #  Phase 1 — F₀ and residuals
        # ════════════════════════════════════════════════════════════════════
        title = (
            Text("Gradient Boosting — Sequential Residual Fitting",
                 font_size=17, color=C_WHITE, weight=BOLD)
            .to_edge(UP, buff=0.35)
        )
        self.play(FadeIn(title), Create(ax), run_time=0.7)

        dots = VGroup(*[
            Dot(ax.c2p(float(xi), float(yi)), radius=0.07, color=C_DATA)
            for xi, yi in zip(_X, _Y)
        ])
        self.play(
            LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.06),
            run_time=0.9,
        )

        # F₀ = mean(y)
        cur_pred = ax.plot(_pred_func(0), color=C_F0, stroke_width=2.4)
        f0_lbl = (
            Text("F\u2080 = mean(y)", font_size=12, color=C_F0)
            .next_to(ax.c2p(5.9, _YMEAN), RIGHT, buff=0.15)
            .shift(UP * 0.28)
        )
        self.play(Create(cur_pred), FadeIn(f0_lbl), run_time=0.6)
        self.wait(0.3)

        # Residuals
        cur_res = _res_vgroup(ax, 0)
        res_lbl = (
            Text("residuals  r = y \u2212 F\u2080  (red lines)", font_size=12, color=C_RES)
            .to_edge(DOWN, buff=0.42)
        )
        self.play(Create(cur_res), FadeIn(res_lbl), run_time=0.65)
        self.wait(1.0)

        # ════════════════════════════════════════════════════════════════════
        #  Phase 2 — Three boosting rounds
        # ════════════════════════════════════════════════════════════════════
        title2 = (
            Text("Each round: fit a weak tree to the residuals",
                 font_size=17, color=C_WHITE, weight=BOLD)
            .to_edge(UP, buff=0.35)
        )
        self.play(
            FadeOut(f0_lbl), FadeOut(res_lbl),
            ReplacementTransform(title, title2),
            run_time=0.5,
        )

        # Running formula (right margin)
        FMT_X  = 5.55
        FMT_Y0 = 0.8
        FMT_DY = 0.52

        f0_fml = (
            Text(_F_LBL[0], font_size=11, color=C_DIM)
            .move_to([FMT_X, FMT_Y0, 0])
        )
        self.play(FadeIn(f0_fml), run_time=0.3)
        formula_objs = [f0_fml]

        for k in range(ROUNDS):
            col = ROUND_COLS[k]

            # Flash weak tree h_{k+1}
            tree_line = ax.plot(
                lambda x, _k=k: float(_trees[_k].predict([[x]])[0]),
                x_range=[0.4, 5.9, 0.03],
                color=C_TREE, stroke_width=1.8,
            )
            rnd_hdr = (
                Text(f"Round {k+1}", font_size=14, color=C_WHITE, weight=BOLD)
                .to_edge(DOWN, buff=1.05)
            )
            tree_lbl = (
                Text(f"h{_SUBS[k]}(x) fits residuals", font_size=11, color=C_TREE)
                .to_edge(DOWN, buff=0.42)
            )
            self.play(FadeIn(rnd_hdr), Create(tree_line), FadeIn(tree_lbl), run_time=0.55)
            self.wait(0.30)

            # Update ensemble prediction
            new_pred = ax.plot(_pred_func(k + 1), color=col, stroke_width=2.4)
            new_res  = _res_vgroup(ax, k + 1)

            self.play(
                Transform(cur_pred, new_pred),
                FadeOut(tree_line),
                FadeOut(tree_lbl),
                FadeOut(rnd_hdr),
                FadeOut(cur_res),
                run_time=0.65,
            )
            if len(new_res) > 0:
                self.play(Create(new_res), run_time=0.35)
            cur_res = new_res

            # Append formula line
            fml = (
                Text(_F_LBL[k + 1], font_size=11, color=col)
                .move_to([FMT_X, FMT_Y0 - (k + 1) * FMT_DY, 0])
            )
            self.play(FadeIn(fml), run_time=0.30)
            formula_objs.append(fml)
            self.wait(0.55)

        conv_lbl = (
            Text("residuals shrink each round  \u2192  prediction converges",
                 font_size=12, color=C_DIM)
            .to_edge(DOWN, buff=0.42)
        )
        self.play(FadeIn(conv_lbl), run_time=0.4)
        self.wait(1.2)

        # ════════════════════════════════════════════════════════════════════
        #  Phase 3 — AdaBoost vs Gradient Boosting
        # ════════════════════════════════════════════════════════════════════
        title3 = (
            Text("AdaBoost vs Gradient Boosting",
                 font_size=17, color=C_WHITE, weight=BOLD)
            .to_edge(UP, buff=0.35)
        )
        p2_all = VGroup(ax, dots, cur_pred, cur_res, *formula_objs, conv_lbl)
        self.play(FadeOut(p2_all), ReplacementTransform(title2, title3), run_time=0.7)

        divider = Line([0, 2.6, 0], [0, -2.6, 0], color=C_DIM, stroke_width=0.8)
        self.play(Create(divider), run_time=0.3)

        # ── Left: AdaBoost ────────────────────────────────────────────────
        LX = -3.2   # panel center x

        adb_hdr = (
            Text("AdaBoost", font_size=15, color=C_WHITE, weight=BOLD)
            .move_to([LX, 2.15, 0])
        )
        adb_axis = Line([LX - 2.3, 0.3, 0], [LX + 2.3, 0.3, 0],
                        color=C_DIM, stroke_width=1.2)

        # 10 classification points — 4 amber left, 1 amber near boundary,
        # 1 teal near boundary, 4 teal right
        ADB_XS = [-2.0, -1.5, -1.1, -0.7, -0.2, 0.2, 0.7, 1.1, 1.5, 2.0]
        ADB_CS = [C_C0]*5 + [C_C1]*5
        MISCL  = {4, 5}   # near-boundary points: misclassified by threshold at x=LX

        adb_dots = VGroup(*[
            Dot([LX + ADB_XS[i] * 0.72, 0.3, 0], radius=0.10, color=ADB_CS[i])
            for i in range(10)
        ])
        adb_thresh = DashedLine(
            [LX, -0.25, 0], [LX, 0.90, 0],
            color=C_DIM, stroke_width=1.8, dash_length=0.10,
        )
        adb_t1_lbl = (
            Text("round 1\nthreshold", font_size=9, color=C_DIM)
            .next_to(adb_thresh, RIGHT, buff=0.08)
        )

        self.play(FadeIn(adb_hdr), Create(adb_axis), run_time=0.4)
        self.play(
            LaggedStart(*[GrowFromCenter(d) for d in adb_dots], lag_ratio=0.05),
            run_time=0.6,
        )
        self.play(Create(adb_thresh), FadeIn(adb_t1_lbl), run_time=0.4)
        self.wait(0.3)

        # Up-weight misclassified points
        self.play(
            *[adb_dots[i].animate.scale(2.3).set_color(C_TREE) for i in MISCL],
            run_time=0.55,
        )
        adb_cap = (
            Text("up-weight misclassified samples\n\u2192 next tree focuses on them",
                 font_size=11, color=C_DIM)
            .move_to([LX, -2.55, 0])
        )
        self.play(FadeIn(adb_cap), run_time=0.40)

        # ── Right: Gradient Boosting mini demo ───────────────────────────
        RX = 3.0

        gb_hdr = (
            Text("Gradient Boosting", font_size=15, color=C_WHITE, weight=BOLD)
            .move_to([RX, 2.15, 0])
        )

        mini_ax = Axes(
            x_range=[0, 6.3, 1],
            y_range=[-1.6, 1.6, 1],
            x_length=4.0, y_length=2.2,
            **_AX,
        ).move_to([RX, 0.45, 0])

        mini_dots = VGroup(*[
            Dot(mini_ax.c2p(float(xi), float(yi)), radius=0.055, color=C_DATA)
            for xi, yi in zip(_X, _Y)
        ])
        mini_f0 = mini_ax.plot(_pred_func(0), color=C_F0, stroke_width=1.6)
        mini_res = _res_vgroup(mini_ax, 0, min_len=0.06)

        self.play(FadeIn(gb_hdr), Create(mini_ax), run_time=0.5)
        self.play(
            LaggedStart(*[GrowFromCenter(d) for d in mini_dots], lag_ratio=0.04),
            Create(mini_f0),
            run_time=0.6,
        )
        self.play(Create(mini_res), run_time=0.35)
        self.wait(0.3)

        mini_f3 = mini_ax.plot(_pred_func(ROUNDS), color=C_GREEN, stroke_width=2.0)
        self.play(
            Transform(mini_f0, mini_f3),
            FadeOut(mini_res),
            run_time=0.85,
        )

        gb_cap = (
            Text("fit residuals directly\n\u2192 prediction improves each round",
                 font_size=11, color=C_DIM)
            .move_to([RX, -2.55, 0])
        )
        self.play(FadeIn(gb_cap), run_time=0.40)

        # ── Shared caption ────────────────────────────────────────────────
        bottom = (
            Text(
                "Both: combine many weak learners into one strong predictor",
                font_size=12, color=C_DIM,
            )
            .to_edge(DOWN, buff=0.55)
        )
        self.play(FadeIn(bottom), run_time=0.5)
        self.wait(2.5)
