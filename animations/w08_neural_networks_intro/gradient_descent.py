"""
W08 — Gradient Descent

  Phase 1: 1D loss surface — a ball (parameter w) rolls downhill guided by
           the rotating tangent line (gradient).  7 steps shown explicitly
           with the update formula w ← w − η·∇L(w).
  Phase 2: Three mini-plots side by side — the same ball on the same curve
           for η = 0.07 (too small), 0.60 (converges), 3.50 (oscillates) —
           all animating simultaneously so the contrast is immediate.
  Phase 3: The resulting training-loss curve (Loss vs Epoch) drawn stroke-
           by-stroke, connecting the per-step updates to the familiar plot
           students see in PyTorch training loops.

Render:
  ../../env/bin/manim -pql gradient_descent.py GradientDescent
  ../../env/bin/manim -pqh gradient_descent.py GradientDescent
"""

from manim import *
import numpy as np

# ── Palette ───────────────────────────────────────────────────────────────────
C_BG     = "#0d1117"
C_WHITE  = "#ffffff"
C_DIM    = "#8b949e"
C_GRID   = "#21262d"

C_CURVE  = "#58a6ff"   # blue   — loss curve
C_DOT    = "#ffa657"   # amber  — the parameter ball
C_TAN    = "#e3b341"   # gold   — tangent line (gradient)
C_MIN    = "#3fb950"   # green  — minimum marker / "good" LR
C_SLOW   = "#8b949e"   # grey   — "too slow" LR
C_FAST   = "#f78166"   # red    — "too large" LR
C_TRAIN  = "#2dd4bf"   # teal   — training loss curve

# ── Phase 1 loss function ─────────────────────────────────────────────────────
# Slightly wavy parabola so it looks more realistic than a pure quadratic.
# Minimum near w ≈ 2.5.  x-domain: [-1, 5].
def _L(w):
    return 0.22 * (w - 2.5) ** 2 + 0.05 * np.sin(2 * w) + 0.08

def _dL(w):
    return 0.44 * (w - 2.5) + 0.10 * np.cos(2 * w)

ETA_MAIN = 0.50
W_START  = -0.50

# Pre-compute 7-step path at module level
_W_PATH = [W_START]
for _i in range(7):
    _W_PATH.append(_W_PATH[-1] - ETA_MAIN * _dL(_W_PATH[-1]))

# ── Phase 2: simple parabola for three-LR comparison ─────────────────────────
def _L2(w):  return 0.25 * (w - 3.0) ** 2
def _dL2(w): return 0.50 * (w - 3.0)

W2_START = 0.50
N2_STEPS = 6

LR_CASES = [
    (0.07, C_SLOW, "eta = 0.07\n(too small)"),
    (0.60, C_MIN,  "eta = 0.60\n(converges)"),
    (3.50, C_FAST, "eta = 3.50\n(oscillates)"),
]

def _run_gd2(eta, n=N2_STEPS):
    ws = [W2_START]
    for _ in range(n):
        ws.append(ws[-1] - eta * _dL2(ws[-1]))
    return ws

_W2_PATHS = [_run_gd2(lr) for lr, _, _ in LR_CASES]

# ── Phase 3: synthetic training-loss curve ────────────────────────────────────
np.random.seed(42)
_EPOCHS = np.arange(0, 51)
_noise  = np.random.randn(51) * 0.05
_TRAIN_LOSS = 2.2 * np.exp(-0.09 * _EPOCHS) + 0.18 + _noise.clip(-0.07, 0.07)

# ── Shared axes config ────────────────────────────────────────────────────────
_AX_CFG = dict(
    axis_config={"color": C_GRID, "stroke_width": 1.0, "include_ticks": False},
    tips=False,
)


def _make_tangent(ax, w, half_span=0.65):
    """Short tangent-line segment at w, clipped to x_range [-1, 5]."""
    g   = _dL(w)
    L_w = _L(w)
    w_lo = max(w - half_span, -1.0)
    w_hi = min(w + half_span,  5.0)
    return Line(
        ax.c2p(w_lo, L_w + g * (w_lo - w)),
        ax.c2p(w_hi, L_w + g * (w_hi - w)),
        stroke_color=C_TAN,
        stroke_width=2.2,
    )


class GradientDescent(Scene):

    def construct(self):
        self.camera.background_color = C_BG
        self._phase1()
        self._phase2()
        self._phase3()

    # ── Phase 1: ball on loss surface ─────────────────────────────────────────

    def _phase1(self):
        title = (
            Text("Gradient Descent — Rolling Down the Loss Surface",
                 font_size=19, color=C_WHITE, weight=BOLD)
            .to_edge(UP, buff=0.30)
        )
        self.play(FadeIn(title), run_time=0.5)

        ax = Axes(
            x_range=[-1, 5, 1],
            y_range=[ 0, 2.5, 0.5],
            x_length=7.8, y_length=4.6,
            **_AX_CFG,
        ).move_to([0.2, -0.35, 0])

        # Axis tick labels
        x_ticks = VGroup(*[
            Text(str(v), font_size=11, color=C_DIM)
            .next_to(ax.c2p(v, 0), DOWN, buff=0.16)
            for v in range(-1, 6)
        ])
        x_lbl = Text("w  (parameter)", font_size=12, color=C_DIM).next_to(ax, DOWN, buff=0.48)
        y_lbl = (
            Text("L(w)  (loss)", font_size=12, color=C_DIM)
            .rotate(PI / 2)
            .next_to(ax, LEFT, buff=0.45)
        )

        self.play(Create(ax), FadeIn(x_ticks), FadeIn(x_lbl), FadeIn(y_lbl), run_time=0.5)

        # Draw loss curve
        curve = ax.plot(_L, x_range=[-0.9, 4.9], color=C_CURVE, stroke_width=2.5)
        self.play(Create(curve), run_time=0.8)

        # Formula label (top-right of axes)
        formula = (
            Text("w  ←  w  −  η · ∇L(w)     η = 0.5",
                 font_size=13, color=C_TAN)
            .to_corner(UR, buff=0.4)
        )
        self.play(FadeIn(formula), run_time=0.4)

        # Place starting dot
        dot = Dot(ax.c2p(W_START, _L(W_START)), radius=0.14,
                  color=C_DOT, fill_opacity=1.0)
        dot.set_stroke(color=C_BG, width=1.2)
        self.play(GrowFromCenter(dot), run_time=0.4)
        self.wait(0.3)

        # First tangent line
        tangent = _make_tangent(ax, _W_PATH[0])
        self.play(Create(tangent), run_time=0.4)
        self.wait(0.2)

        # 7 gradient steps
        step_lbl = None
        for t in range(7):
            w_old = _W_PATH[t]
            w_new = _W_PATH[t + 1]
            new_tangent = _make_tangent(ax, w_new)

            new_lbl = (
                Text(
                    f"step {t+1}:  w = {w_old:.3f}  →  {w_new:.3f}",
                    font_size=12, color=C_DIM,
                )
                .to_edge(DOWN, buff=0.42)
            )

            if step_lbl is None:
                self.play(FadeIn(new_lbl), run_time=0.2)
            else:
                self.play(ReplacementTransform(step_lbl, new_lbl), run_time=0.2)
            step_lbl = new_lbl

            self.play(
                dot.animate.move_to(ax.c2p(w_new, _L(w_new))),
                ReplacementTransform(tangent, new_tangent),
                run_time=0.55,
            )
            tangent = new_tangent
            self.wait(0.15)

        # Converged annotation
        w_final = _W_PATH[-1]
        conv_line = DashedLine(
            ax.c2p(w_final, 0), ax.c2p(w_final, _L(w_final)),
            color=C_MIN, stroke_width=1.5, dash_length=0.12,
        )
        conv_lbl = (
            Text("converged", font_size=13, color=C_MIN, weight=BOLD)
            .next_to(ax.c2p(w_final, _L(w_final)), UR, buff=0.15)
        )
        self.play(Create(conv_line), FadeIn(conv_lbl), run_time=0.5)
        self.wait(1.5)

        self.play(
            FadeOut(VGroup(
                title, ax, curve, dot, tangent, formula,
                x_ticks, x_lbl, y_lbl,
                conv_line, conv_lbl, step_lbl,
            )),
            run_time=0.5,
        )

    # ── Phase 2: three learning rates compared ────────────────────────────────

    def _phase2(self):
        title = (
            Text("Learning Rate — Too Small, Just Right, Too Large",
                 font_size=19, color=C_WHITE, weight=BOLD)
            .to_edge(UP, buff=0.30)
        )
        self.play(FadeIn(title), run_time=0.4)

        # Three mini-axes, centred at x = -4.0 / 0.0 / +4.0
        AX_KW = dict(
            x_range=[0, 6, 1],
            y_range=[0, 2.5, 0.5],
            x_length=3.5, y_length=2.6,
            **_AX_CFG,
        )
        X_CENTS = [-4.0, 0.0, 4.0]
        Y_CENT  = -0.30

        axes   = [Axes(**AX_KW).move_to([xc, Y_CENT, 0]) for xc in X_CENTS]
        colors = [lr_col for _, lr_col, _ in LR_CASES]
        labels = [lr_lbl for _, _, lr_lbl in LR_CASES]

        # Draw parabolas
        curves = [
            ax.plot(_L2, x_range=[0.05, 5.95], color=C_CURVE, stroke_width=2.0)
            for ax in axes
        ]
        # Minimum markers
        min_lines = [
            DashedLine(
                ax.c2p(3.0, 0), ax.c2p(3.0, _L2(0.05)),
                color=C_DIM, stroke_width=1.0, dash_length=0.10,
            )
            for ax in axes
        ]
        # Minimum labels
        min_lbls = [
            Text("w*", font_size=10, color=C_DIM).next_to(ax.c2p(3.0, 0), DOWN, buff=0.12)
            for ax in axes
        ]

        self.play(
            LaggedStart(*[Create(ax) for ax in axes], lag_ratio=0.1),
            run_time=0.5,
        )
        self.play(
            *[Create(c) for c in curves],
            *[Create(ml) for ml in min_lines],
            *[FadeIn(ml) for ml in min_lbls],
            run_time=0.6,
        )

        # Panel labels (η value + description) above each axis
        panel_hdrs = [
            Text(lbl, font_size=12, color=col)
            .next_to(axes[k], UP, buff=0.18)
            for k, (col, lbl) in enumerate(zip(colors, labels))
        ]
        self.play(LaggedStart(*[FadeIn(h) for h in panel_hdrs], lag_ratio=0.15), run_time=0.4)

        # Initial dots
        dots = [
            Dot(axes[k].c2p(W2_START, _L2(W2_START)),
                radius=0.13, color=colors[k], fill_opacity=1.0)
            .set_stroke(color=C_BG, width=1.0)
            for k in range(3)
        ]
        self.play(*[GrowFromCenter(d) for d in dots], run_time=0.4)
        self.wait(0.3)

        # Animate N2_STEPS steps simultaneously
        for t in range(N2_STEPS):
            anims = []
            for k in range(3):
                w_new = _W2_PATHS[k][t + 1]
                anims.append(
                    dots[k].animate.move_to(axes[k].c2p(w_new, _L2(w_new)))
                )
            self.play(*anims, run_time=0.50)
            self.wait(0.18)

        # Final captions
        outcome_txts = [
            "barely moved",
            "reached minimum",
            "jumped back and forth",
        ]
        captions = [
            Text(txt, font_size=11, color=colors[k])
            .next_to(axes[k], DOWN, buff=0.28)
            for k, txt in enumerate(outcome_txts)
        ]
        self.play(LaggedStart(*[FadeIn(c) for c in captions], lag_ratio=0.15), run_time=0.5)
        self.wait(1.8)

        self.play(
            FadeOut(VGroup(
                title,
                *axes, *curves, *min_lines, *min_lbls,
                *panel_hdrs, *dots, *captions,
            )),
            run_time=0.5,
        )

    # ── Phase 3: training loss curve ─────────────────────────────────────────

    def _phase3(self):
        title = (
            Text("The Training Loss Curve",
                 font_size=20, color=C_WHITE, weight=BOLD)
            .to_edge(UP, buff=0.30)
        )
        sub = (
            Text("each epoch runs gradient descent over the full dataset",
                 font_size=12, color=C_DIM)
            .next_to(title, DOWN, buff=0.12)
        )
        self.play(FadeIn(title), FadeIn(sub), run_time=0.5)

        ax = Axes(
            x_range=[0, 50, 10],
            y_range=[0, 2.5, 0.5],
            x_length=8.0, y_length=4.6,
            **_AX_CFG,
        ).move_to([0.3, -0.40, 0])

        # Manual tick labels
        x_ticks = VGroup(*[
            Text(str(v), font_size=11, color=C_DIM)
            .next_to(ax.c2p(v, 0), DOWN, buff=0.16)
            for v in range(0, 51, 10)
        ])
        y_ticks = VGroup(*[
            Text(f"{v:.1f}", font_size=11, color=C_DIM)
            .next_to(ax.c2p(0, v), LEFT, buff=0.16)
            for v in [0.5, 1.0, 1.5, 2.0, 2.5]
        ])
        x_lbl = Text("Epoch", font_size=12, color=C_DIM).next_to(ax, DOWN, buff=0.48)
        y_lbl = (
            Text("Loss", font_size=12, color=C_DIM)
            .rotate(PI / 2)
            .next_to(ax, LEFT, buff=0.45)
        )

        self.play(
            Create(ax), FadeIn(x_ticks), FadeIn(y_ticks),
            FadeIn(x_lbl), FadeIn(y_lbl),
            run_time=0.6,
        )

        # Draw training loss curve stroke-by-stroke
        loss_curve = VMobject(stroke_color=C_TRAIN, stroke_width=2.5, fill_opacity=0)
        self.add(loss_curve)

        def _update_curve(mob, alpha):
            n = min(len(_EPOCHS), max(2, int(alpha * len(_EPOCHS)) + 1))
            pts = [ax.c2p(int(_EPOCHS[i]), float(_TRAIN_LOSS[i])) for i in range(n)]
            mob.set_points_as_corners(pts)

        self.play(
            UpdateFromAlphaFunc(loss_curve, _update_curve),
            run_time=3.0,
            rate_func=linear,
        )
        self.wait(0.3)

        # Annotate: rapid drop + plateau
        # "rapid drop" arrow near epoch 5
        drop_pt = ax.c2p(5, float(_TRAIN_LOSS[5]))
        drop_lbl = (
            Text("rapid drop\n(large gradient steps)", font_size=11, color=C_TAN)
            .next_to(drop_pt, UR, buff=0.3)
        )
        drop_arrow = Arrow(
            drop_lbl.get_bottom() + DOWN * 0.05,
            drop_pt + UR * 0.06,
            color=C_TAN, stroke_width=1.6, buff=0.05,
            max_tip_length_to_length_ratio=0.25,
        )
        # "plateau" annotation near epoch 40
        plat_pt = ax.c2p(40, float(_TRAIN_LOSS[40]))
        plat_lbl = (
            Text("plateau\n(small gradients\nnear minimum)", font_size=11, color=C_MIN)
            .next_to(plat_pt, UR, buff=0.3)
        )
        plat_arrow = Arrow(
            plat_lbl.get_left() + LEFT * 0.05,
            plat_pt + RIGHT * 0.08,
            color=C_MIN, stroke_width=1.6, buff=0.05,
            max_tip_length_to_length_ratio=0.25,
        )

        self.play(FadeIn(drop_lbl), Create(drop_arrow), run_time=0.5)
        self.play(FadeIn(plat_lbl), Create(plat_arrow), run_time=0.5)

        caption = (
            Text(
                "gradient descent turns the loss surface into the training curve — "
                "choose η so the curve drops smoothly",
                font_size=11, color=C_DIM,
            )
            .to_edge(DOWN, buff=0.42)
        )
        self.play(FadeIn(caption), run_time=0.4)
        self.wait(2.5)
