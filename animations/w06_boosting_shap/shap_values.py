"""
W06 — SHAP Values: Global Importance, Waterfall, and Beeswarm

  Phase 1: Global feature importance — horizontal mean|SHAP| bar chart.
  Phase 2: Waterfall — how a single prediction is built from the base value.
  Phase 3: Beeswarm summary — distribution of SHAP values across all samples,
           coloured by feature value.

Render:
  ../../env/bin/manim -pql shap_values.py SHAPValues
  ../../env/bin/manim -pqh shap_values.py SHAPValues
"""

from manim import *
import numpy as np
from manim.utils.color import ManimColor

# ── Palette ───────────────────────────────────────────────────────────────────
C_BG    = "#0d1117"
C_WHITE = "#ffffff"
C_DIM   = "#8b949e"
C_POS   = "#f78166"   # red  – positive SHAP (pushes output up)
C_NEG   = "#58a6ff"   # blue – negative SHAP (pushes output down)
C_GOLD  = "#e3b341"   # gold – final prediction
C_TEAL  = "#2dd4bf"   # teal – importance bars

# ── Fixed data ────────────────────────────────────────────────────────────────
FEATURES  = ["rooms", "distance", "age", "crime rate"]
MEAN_ABS  = [0.42, 0.31, 0.18, 0.09]       # mean |SHAP| across all samples

BASE_VALUE  = 2.20                           # E[f(x)]  (e.g. house price, $100k)
SHAP_SINGLE = [+0.85, -0.38, -0.19, -0.07]  # one sample's feature contributions

# Beeswarm data: 16 samples × 4 features
np.random.seed(3)
_N = 16
_BEE_SHAP = [
    np.random.normal( 0.40, 0.26, _N),
    np.random.normal(-0.22, 0.18, _N),
    np.random.normal(-0.09, 0.12, _N),
    np.random.normal(-0.03, 0.06, _N),
]
_BEE_FVAL = [np.clip(np.random.normal(0.55, 0.28, _N), 0, 1) for _ in range(4)]

# Pre-compute beeswarm jitter (seed fixed for reproducibility)
np.random.seed(99)
_BEE_JITTER = [np.random.uniform(-0.14, 0.14, _N) for _ in range(4)]


def _fval_col(v):
    """Blue (low feature value) → red (high feature value)."""
    return interpolate_color(ManimColor("#58a6ff"), ManimColor("#f78166"), float(v))


# ── Scene ─────────────────────────────────────────────────────────────────────

class SHAPValues(Scene):

    def construct(self):
        self.camera.background_color = C_BG
        self._phase1()
        self._phase2()
        self._phase3()

    # ── Phase 1: Global importance bar chart ──────────────────────────────────
    def _phase1(self):
        title = (
            Text("SHAP — Global Feature Importance",
                 font_size=18, color=C_WHITE, weight=BOLD)
            .to_edge(UP, buff=0.35)
        )
        sub = (
            Text("mean |SHAP value| averaged across all predictions",
                 font_size=12, color=C_DIM)
            .next_to(title, DOWN, buff=0.12)
        )
        self.play(FadeIn(title), FadeIn(sub), run_time=0.6)

        ZERO_X = -2.2
        MAX_W  = 5.0
        BAR_H  = 0.38
        ROW_Y  = [1.4, 0.55, -0.30, -1.15]

        zero_line = DashedLine(
            [ZERO_X, ROW_Y[-1] - 0.55, 0],
            [ZERO_X, ROW_Y[0]  + 0.55, 0],
            color=C_DIM, stroke_width=0.9, dash_length=0.09,
        )
        x_lbl = (
            Text("mean |SHAP|", font_size=11, color=C_DIM)
            .move_to([ZERO_X + MAX_W / 2, ROW_Y[-1] - 0.95, 0])
        )
        self.play(Create(zero_line), FadeIn(x_lbl), run_time=0.35)

        all_rows = VGroup()
        for feat, val, y in zip(FEATURES, MEAN_ABS, ROW_Y):
            w   = val / max(MEAN_ABS) * MAX_W
            bar = (
                Rectangle(width=w, height=BAR_H, color=C_TEAL,
                          fill_opacity=0.85, stroke_width=0)
                .move_to([ZERO_X + w / 2, y, 0])
            )
            feat_lbl = (
                Text(feat, font_size=14, color=C_WHITE)
                .move_to([-5.2, y, 0])
            )
            val_lbl = (
                Text(f"{val:.2f}", font_size=12, color=C_TEAL)
                .next_to(bar, RIGHT, buff=0.14)
            )
            all_rows.add(VGroup(feat_lbl, bar, val_lbl))
            self.play(FadeIn(feat_lbl), GrowFromEdge(bar, LEFT), run_time=0.45)
            self.play(FadeIn(val_lbl), run_time=0.2)

        caption = (
            Text('"rooms" dominates — highest global feature importance',
                 font_size=12, color=C_DIM)
            .to_edge(DOWN, buff=0.42)
        )
        self.play(FadeIn(caption), run_time=0.4)
        self.wait(1.5)
        self.play(
            FadeOut(VGroup(title, sub, zero_line, x_lbl, all_rows, caption)),
            run_time=0.5,
        )

    # ── Phase 2: Waterfall plot ───────────────────────────────────────────────
    def _phase2(self):
        title = (
            Text("Waterfall — One Prediction Explained",
                 font_size=18, color=C_WHITE, weight=BOLD)
            .to_edge(UP, buff=0.35)
        )
        sub = (
            Text("each bar shows how one feature pushes the output from the base value",
                 font_size=12, color=C_DIM)
            .next_to(title, DOWN, buff=0.12)
        )
        self.play(FadeIn(title), FadeIn(sub), run_time=0.5)

        nl = (
            NumberLine(
                x_range=[1.2, 3.9, 0.5],
                length=8.8,
                color="#21262d",
                stroke_width=1.2,
                include_numbers=False,
            )
            .move_to([0.1, -2.0, 0])
        )
        nl_lbls = VGroup(*[
            Text(f"{v:.1f}", font_size=13, color=C_DIM).next_to(nl.n2p(v), DOWN, buff=0.15)
            for v in np.arange(1.5, 3.9, 0.5)
        ])
        self.play(Create(nl), FadeIn(nl_lbls), run_time=0.4)

        # Base value vertical marker
        bp = nl.n2p(BASE_VALUE)
        base_line = DashedLine(
            bp + UP * 0.15, bp + UP * 4.1,
            color=C_DIM, stroke_width=1.2, dash_length=0.09,
        )
        base_lbl = (
            Text(f"base = {BASE_VALUE:.2f}", font_size=12, color=C_DIM)
            .next_to(bp + UP * 4.1, UP, buff=0.1)
            .shift(LEFT * 0.6)
        )
        self.play(Create(base_line), FadeIn(base_lbl), run_time=0.4)

        ROW_Y = [1.5, 0.7, -0.1, -0.9]
        BAR_H = 0.38
        cumul = BASE_VALUE
        all_p2 = VGroup()

        for i, (feat, shap) in enumerate(zip(FEATURES, SHAP_SINGLE)):
            col  = C_POS if shap > 0 else C_NEG
            sign = "+" if shap > 0 else ""
            x0   = min(cumul, cumul + shap)
            x1   = max(cumul, cumul + shap)
            w    = nl.n2p(x1)[0] - nl.n2p(x0)[0]
            cx   = (nl.n2p(x0)[0] + nl.n2p(x1)[0]) / 2
            y    = ROW_Y[i]

            bar = (
                Rectangle(width=w, height=BAR_H, color=col,
                          fill_opacity=0.85, stroke_width=0)
                .move_to([cx, y, 0])
            )
            feat_lbl = Text(feat, font_size=13, color=C_WHITE).move_to([-4.5, y, 0])
            val_lbl  = (
                Text(f"{sign}{shap:.2f}", font_size=13, color=col)
                .next_to(bar, RIGHT, buff=0.12)
            )

            edge = LEFT if shap > 0 else RIGHT
            self.play(FadeIn(feat_lbl), GrowFromEdge(bar, edge), run_time=0.5)

            # Vertical connector to next row
            conn_x = nl.n2p(cumul + shap)[0]
            if i < len(FEATURES) - 1:
                conn = DashedLine(
                    [conn_x, y - BAR_H / 2, 0],
                    [conn_x, ROW_Y[i + 1] + BAR_H / 2, 0],
                    color=C_DIM, stroke_width=0.8, dash_length=0.06,
                )
                self.play(FadeIn(val_lbl), Create(conn), run_time=0.3)
                all_p2.add(conn)
            else:
                self.play(FadeIn(val_lbl), run_time=0.3)

            all_p2.add(bar, feat_lbl, val_lbl)
            cumul += shap

        # Final prediction vertical marker
        fp    = nl.n2p(cumul)
        fline = Line(
            fp + UP * 0.15, fp + UP * 4.1,
            color=C_GOLD, stroke_width=2.2,
        )
        flbl = (
            Text(f"f(x) = {cumul:.2f}", font_size=13, color=C_GOLD, weight=BOLD)
            .next_to(fp + UP * 4.1, UP, buff=0.1)
            .shift(RIGHT * 0.6)
        )
        self.play(Create(fline), FadeIn(flbl), run_time=0.5)
        self.wait(1.8)

        self.play(
            FadeOut(VGroup(title, sub, nl, nl_lbls, base_line, base_lbl, all_p2, fline, flbl)),
            run_time=0.6,
        )

    # ── Phase 3: Beeswarm summary plot ────────────────────────────────────────
    def _phase3(self):
        title = (
            Text("Beeswarm — SHAP Value Distribution",
                 font_size=18, color=C_WHITE, weight=BOLD)
            .to_edge(UP, buff=0.35)
        )
        sub = (
            Text(
                "each dot = one sample  ·  colour = feature value  "
                "(blue = low,  red = high)",
                font_size=12, color=C_DIM,
            )
            .next_to(title, DOWN, buff=0.12)
        )
        self.play(FadeIn(title), FadeIn(sub), run_time=0.5)

        nl = (
            NumberLine(
                x_range=[-0.95, 1.15, 0.2],
                length=8.8,
                color="#21262d",
                stroke_width=1.2,
                include_numbers=False,
            )
            .move_to([0.1, -2.7, 0])
        )
        nl_lbls = VGroup(*[
            Text(f"{v:.1f}", font_size=12, color=C_DIM).next_to(nl.n2p(v), DOWN, buff=0.15)
            for v in np.arange(-0.8, 1.15, 0.4)
        ])
        zero_vline = DashedLine(
            nl.n2p(0) + DOWN * 0.2,
            nl.n2p(0) + UP  * 4.8,
            color=C_DIM, stroke_width=0.8, dash_length=0.08,
        )
        ax_lbl = (
            Text("SHAP value", font_size=12, color=C_DIM)
            .next_to(nl, DOWN, buff=0.3)
        )
        self.play(Create(nl), FadeIn(nl_lbls), Create(zero_vline), FadeIn(ax_lbl), run_time=0.5)

        ROW_Y = [1.7, 0.7, -0.3, -1.3]

        for i, (feat, shaps, fvals, jitters) in enumerate(
            zip(FEATURES, _BEE_SHAP, _BEE_FVAL, _BEE_JITTER)
        ):
            y = ROW_Y[i]
            feat_lbl = Text(feat, font_size=13, color=C_WHITE).move_to([-4.8, y, 0])
            self.play(FadeIn(feat_lbl), run_time=0.2)

            dots = VGroup(*[
                Dot(
                    [nl.n2p(float(sv))[0], y + float(jit), 0],
                    radius=0.09,
                    color=_fval_col(fv),
                ).set_stroke(color=C_BG, width=0.8)
                for sv, fv, jit in zip(shaps, fvals, jitters)
            ])
            self.play(
                LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.04),
                run_time=0.55,
            )

        # Colour legend
        LX, LY = 3.6, -2.1
        grad = VGroup(*[
            Rectangle(
                width=0.12, height=0.22,
                fill_color=_fval_col(t / 20), fill_opacity=1.0, stroke_width=0,
            ).move_to([LX + t * 0.065, LY, 0])
            for t in range(21)
        ])
        low_lbl  = Text("Low",  font_size=10, color=C_DIM).next_to(grad, LEFT,  buff=0.1)
        high_lbl = Text("High", font_size=10, color=C_DIM).next_to(grad, RIGHT, buff=0.1)
        leg_hdr  = (
            Text("Feature value", font_size=10, color=C_DIM)
            .next_to(grad, UP, buff=0.12)
        )
        self.play(FadeIn(VGroup(grad, low_lbl, high_lbl, leg_hdr)), run_time=0.4)

        caption = (
            Text(
                '"rooms" has the widest SHAP spread — dominant driver of predictions',
                font_size=12, color=C_DIM,
            )
            .to_edge(DOWN, buff=0.42)
        )
        self.play(FadeIn(caption), run_time=0.4)
        self.wait(2.5)
