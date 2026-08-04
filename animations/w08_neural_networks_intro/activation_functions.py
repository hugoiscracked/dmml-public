"""
W08 — Activation Functions

  Phase 1: Draw Sigmoid, Tanh, and ReLU on shared axes — key properties
           annotated (output range, linearity for positives).

  Phase 2: Three side-by-side derivative plots on the same y-scale [0, 1.2].
           Sigmoid's peak of 0.25 looks tiny next to ReLU's flat 1.0 line —
           immediately showing why gradients shrink through sigmoid layers.

  Phase 3: Vanishing-gradient schematic — 5-layer network shown as two rows
           (Sigmoid vs ReLU).  Gradient propagates backward step-by-step;
           sigmoid values dim and redden toward near-zero while the ReLU row
           stays bright green throughout.

Text uses LaTeX (Tex/MathTex). Labels are width-capped / fit-to-box so the LaTeX
metrics cannot overflow. Captions are wrapped in \\mbox to stay single-line.

Render:
  ../../env/bin/manim -pql activation_functions.py ActivationFunctions
  ../../env/bin/manim -qk  activation_functions.py ActivationFunctions   # 4K master
"""

from manim import *
import numpy as np
from manim.utils.color import ManimColor

# ── Palette ───────────────────────────────────────────────────────────────────
C_BG    = "#0d1117"
C_WHITE = "#ffffff"
C_DIM   = "#8b949e"
C_GRID  = "#21262d"

C_SIG   = "#58a6ff"   # blue  — Sigmoid
C_TANH  = "#2dd4bf"   # teal  — Tanh
C_RELU  = "#ffa657"   # amber — ReLU

C_ALIVE = "#3fb950"   # green — "gradient survives"
C_DEAD  = "#f78166"   # red   — "gradient dies"

# ── Type sizes ─────────────────────────────────────────────────────────────────
FS_TITLE = 46
FS_SUB   = 28
FS_CLBL  = 26   # curve labels
FS_RANGE = 21
FS_TICK  = 21
FS_DLBL  = 28   # derivative-plot labels
FS_MAX   = 21
FS_INS   = 23
FS_ROW   = 26
FS_SMALL = 21
FS_VAL   = 22   # values inside circles
FS_MULT  = 19
FS_VERD  = 25
FS_CAP   = 26


def _tx(s, fs, color=C_DIM, bold=False, math=False, w=None, h=None):
    if math:
        m = MathTex(s, font_size=fs, color=color)
    elif bold:
        m = Tex(r"\textbf{%s}" % s, font_size=fs, color=color)
    else:
        m = Tex(s, font_size=fs, color=color)
    if w is not None and m.width > w:
        m.scale_to_fit_width(w)
    if h is not None and m.height > h:
        m.scale_to_fit_height(h)
    return m


def _stack(lines, fs, colors, math=False, buff=0.14, w=None):
    grp = VGroup()
    for i, ln in enumerate(lines):
        col = colors[i] if isinstance(colors, list) else colors
        grp.add(_tx(ln, fs, color=col, math=math, w=w))
    grp.arrange(DOWN, buff=buff)
    return grp


# ── Activation functions (safe clip for sigmoid) ──────────────────────────────
def _sig(z):  return 1 / (1 + np.exp(-np.clip(z, -30, 30)))
def _tanh(z): return np.tanh(z)
def _relu(z): return np.maximum(0, z)

def _d_sig(z):  return _sig(z) * (1 - _sig(z))
def _d_tanh(z): return 1 - np.tanh(z) ** 2

# ── Phase 3 constants ─────────────────────────────────────────────────────────
LAYER_X = [3.5, 1.75, 0.0, -1.75, -3.5]
ROW_SIG  =  0.85
ROW_RELU = -0.85
CIRC_R   =  0.36

SIG_GRADS  = [1.0, 0.25, 0.0625, 0.0156, 0.0039]
RELU_GRADS = [1.0, 1.0,  1.0,    1.0,    1.0   ]

SIG_COLS  = ["#ffffff", "#e3b341", "#ffa657", "#f78166", "#3d444d"]
RELU_COLS = ["#3fb950"] * 5

_AX_CFG = dict(
    axis_config={"color": C_GRID, "stroke_width": 1.4, "include_ticks": False},
    tips=False,
)


class ActivationFunctions(Scene):

    def construct(self):
        self.camera.background_color = C_BG
        self._phase1()
        self._phase2()
        self._phase3()

    # ── Phase 1: the three functions on shared axes ───────────────────────────

    def _phase1(self):
        title = _tx("Activation Functions", FS_TITLE, color=C_WHITE, bold=True, w=11).to_edge(UP, buff=0.30)
        sub = _tx("non-linearity lets networks learn complex patterns",
                  FS_SUB, color=C_DIM, w=12.0).next_to(title, DOWN, buff=0.14)
        self.play(FadeIn(title), FadeIn(sub), run_time=0.5)

        ax = Axes(
            x_range=[-3, 3, 1], y_range=[-1.5, 3.5, 1],
            x_length=7.8, y_length=5.0, **_AX_CFG,
        ).move_to([0.3, -0.30, 0])

        x_ticks = VGroup(*[
            _tx(str(v), FS_TICK, color=C_DIM, w=0.5).next_to(ax.c2p(v, 0), DOWN, buff=0.16)
            for v in range(-3, 4)
        ])
        y_ticks = VGroup(*[
            _tx(str(v), FS_TICK, color=C_DIM, w=0.5).next_to(ax.c2p(0, v), LEFT, buff=0.16)
            for v in [-1, 0, 1, 2, 3]
        ])
        x_lbl = _tx(r"$z$  (pre-activation)", FS_RANGE, color=C_DIM, w=3.0).next_to(ax, DOWN, buff=0.50)

        x0_line = DashedLine(ax.c2p(-3, 0), ax.c2p(3, 0), color=C_GRID, stroke_width=1.4, dash_length=0.12)
        y0_line = DashedLine(ax.c2p(0, -1.5), ax.c2p(0, 3.5), color=C_GRID, stroke_width=1.4, dash_length=0.12)

        self.play(Create(ax), FadeIn(x_ticks), FadeIn(y_ticks),
                  FadeIn(x_lbl), Create(x0_line), Create(y0_line), run_time=0.5)

        # ── Sigmoid ──────────────────────────────────────────────────────────
        sig_curve = ax.plot(_sig, x_range=[-3, 3], color=C_SIG, stroke_width=3.2)
        sig_lbl = _tx(r"Sigmoid  $\sigma(z) = 1/(1+e^{-z})$", FS_CLBL, color=C_SIG, w=4.3)\
            .next_to(ax.c2p(1.0, _sig(1.0)), DR, buff=0.18)
        sig_range = _tx(r"output: $(0, 1)$", FS_RANGE, color=C_SIG, w=2.2)\
            .next_to(sig_lbl, DOWN, buff=0.08).align_to(sig_lbl, LEFT)
        self.play(Create(sig_curve), run_time=0.7)
        self.play(FadeIn(sig_lbl), FadeIn(sig_range), run_time=0.3)
        self.wait(0.2)

        # ── Tanh ─────────────────────────────────────────────────────────────
        tanh_curve = ax.plot(_tanh, x_range=[-3, 3], color=C_TANH, stroke_width=3.2)
        tanh_lbl = _tx(r"Tanh  $\tanh(z)$", FS_CLBL, color=C_TANH, w=2.8)\
            .next_to(ax.c2p(-1.6, _tanh(-1.6)), DL, buff=0.18)
        tanh_range = _tx(r"output: $(-1, 1)$", FS_RANGE, color=C_TANH, w=2.4)\
            .next_to(tanh_lbl, DOWN, buff=0.08).align_to(tanh_lbl, LEFT)
        self.play(Create(tanh_curve), run_time=0.7)
        self.play(FadeIn(tanh_lbl), FadeIn(tanh_range), run_time=0.3)
        self.wait(0.2)

        # ── ReLU ─────────────────────────────────────────────────────────────
        relu_curve = ax.plot(_relu, x_range=[-3, 3], color=C_RELU, stroke_width=3.2)
        # Placed in the empty upper-left of the plot so the rising ReLU line
        # never crosses the text.
        relu_lbl = _tx(r"ReLU  $\max(0, z)$", FS_CLBL, color=C_RELU, w=2.9)\
            .move_to(ax.c2p(-1.15, 3.05))
        relu_range = _tx(r"output: $[0, \infty)$ --- linear for $z>0$", FS_RANGE, color=C_RELU, w=4.4)\
            .next_to(relu_lbl, DOWN, buff=0.10).align_to(relu_lbl, LEFT)
        self.play(Create(relu_curve), run_time=0.7)
        self.play(FadeIn(relu_lbl), FadeIn(relu_range), run_time=0.3)
        self.wait(1.2)

        self._p1_group = VGroup(
            title, sub, ax, x_ticks, y_ticks, x_lbl, x0_line, y0_line,
            sig_curve, sig_lbl, sig_range,
            tanh_curve, tanh_lbl, tanh_range,
            relu_curve, relu_lbl, relu_range,
        )

    # ── Phase 2: derivative plots ─────────────────────────────────────────────

    def _phase2(self):
        self.play(FadeOut(self._p1_group), run_time=0.4)

        title = _tx("Derivatives --- How Much Gradient Passes Through?",
                    FS_TITLE, color=C_WHITE, bold=True, w=13.0).to_edge(UP, buff=0.30)
        sub = _tx("backprop multiplies the gradient by the activation derivative at each layer",
                  FS_SUB, color=C_DIM, w=13.0).next_to(title, DOWN, buff=0.14)
        self.play(FadeIn(title), FadeIn(sub), run_time=0.4)

        AX_KW = dict(
            x_range=[-3, 3, 1], y_range=[0, 1.2, 0.5],
            x_length=3.5, y_length=2.8, **_AX_CFG,
        )
        X_CENTS = [-4.0, 0.0, 4.0]
        Y_CENT  = -0.55

        ax_s = Axes(**AX_KW).move_to([X_CENTS[0], Y_CENT, 0])
        ax_t = Axes(**AX_KW).move_to([X_CENTS[1], Y_CENT, 0])
        ax_r = Axes(**AX_KW).move_to([X_CENTS[2], Y_CENT, 0])

        y1_lbls = VGroup(*[
            _tx("1.0", FS_MAX, color=C_DIM, w=0.55).next_to(ax.c2p(-3, 1.0), LEFT, buff=0.14)
            for ax in (ax_s, ax_t, ax_r)
        ])
        y0_lbls = VGroup(*[
            _tx("0", FS_MAX, color=C_DIM, w=0.4).next_to(ax.c2p(-3, 0), LEFT, buff=0.14)
            for ax in (ax_s, ax_t, ax_r)
        ])

        self.play(
            Create(ax_s), Create(ax_t), Create(ax_r),
            FadeIn(y1_lbls), FadeIn(y0_lbls), run_time=0.5,
        )

        # ── σ'(z) ─────────────────────────────────────────────────────────────
        sig_d_curve = ax_s.plot(_d_sig, x_range=[-3, 3], color=C_SIG, stroke_width=3.2)
        sig_d_lbl = _tx(r"\sigma'(z)", FS_DLBL, color=C_SIG, math=True).next_to(ax_s, UP, buff=0.22)
        max_sig_line = DashedLine(ax_s.c2p(-3, 0.25), ax_s.c2p(3, 0.25), color=C_SIG, stroke_width=1.8, dash_length=0.10)
        max_sig_lbl = _tx(r"max $= 0.25$", FS_MAX, color=C_SIG, w=1.7).next_to(ax_s.c2p(0, 0.25), UR, buff=0.10)

        # ── tanh'(z) ──────────────────────────────────────────────────────────
        tanh_d_curve = ax_t.plot(_d_tanh, x_range=[-3, 3], color=C_TANH, stroke_width=3.2)
        tanh_d_lbl = _tx(r"\tanh'(z)", FS_DLBL, color=C_TANH, math=True).next_to(ax_t, UP, buff=0.22)
        max_tanh_line = DashedLine(ax_t.c2p(-3, 1.0), ax_t.c2p(3, 1.0), color=C_TANH, stroke_width=1.8, dash_length=0.10)
        max_tanh_lbl = _tx(r"max $= 1.0$", FS_MAX, color=C_TANH, w=1.7).next_to(ax_t.c2p(0, 1.0), UR, buff=0.10)

        # ── ReLU'(z) ──────────────────────────────────────────────────────────
        relu_d_neg = ax_r.plot(lambda z: 0.0, x_range=[-3, -0.01], color=C_RELU, stroke_width=3.2)
        relu_d_pos = ax_r.plot(lambda z: 1.0, x_range=[ 0.01,  3], color=C_RELU, stroke_width=3.2)
        relu_d_vert = DashedLine(ax_r.c2p(0, 0), ax_r.c2p(0, 1.0), color=C_RELU, stroke_width=1.8, dash_length=0.10)
        relu_d_lbl = _tx(r"\mathrm{ReLU}'(z)", FS_DLBL, color=C_RELU, math=True).next_to(ax_r, UP, buff=0.22)
        relu_one_lbl = _tx(r"$= 1$ for $z>0$", FS_MAX, color=C_RELU, w=2.0).next_to(ax_r.c2p(1.5, 1.0), UR, buff=0.10)

        self.play(
            Create(sig_d_curve), Create(tanh_d_curve),
            Create(relu_d_neg), Create(relu_d_pos), Create(relu_d_vert), run_time=0.8,
        )
        self.play(FadeIn(sig_d_lbl), FadeIn(tanh_d_lbl), FadeIn(relu_d_lbl), run_time=0.3)
        self.play(
            Create(max_sig_line), FadeIn(max_sig_lbl),
            Create(max_tanh_line), FadeIn(max_tanh_lbl),
            FadeIn(relu_one_lbl), run_time=0.5,
        )

        insight = _stack(
            [r"Sigmoid's max derivative is 0.25 --- each layer shrinks gradients by 4$\times$",
             r"ReLU passes gradient through unchanged for active neurons"],
            FS_INS, C_DIM, buff=0.14, w=12.5,
        ).to_edge(DOWN, buff=0.42)
        self.play(FadeIn(insight), run_time=0.4)
        self.wait(1.8)

        self._p2_group = VGroup(
            title, sub, ax_s, ax_t, ax_r, y1_lbls, y0_lbls,
            sig_d_curve, sig_d_lbl, max_sig_line, max_sig_lbl,
            tanh_d_curve, tanh_d_lbl, max_tanh_line, max_tanh_lbl,
            relu_d_neg, relu_d_pos, relu_d_vert, relu_d_lbl, relu_one_lbl,
            insight,
        )

    # ── Phase 3: vanishing gradient schematic ────────────────────────────────

    def _phase3(self):
        self.play(FadeOut(self._p2_group), run_time=0.4)

        title = _tx("Vanishing Gradient --- 5-Layer Network",
                    FS_TITLE, color=C_WHITE, bold=True, w=12.0).to_edge(UP, buff=0.30)
        sub = _tx(r"gradient $\times$ activation derivative at each layer "
                  r"(right $\to$ left $=$ output $\to$ input)",
                  FS_SUB, color=C_DIM, w=13.0).next_to(title, DOWN, buff=0.14)
        self.play(FadeIn(title), FadeIn(sub), run_time=0.4)

        sig_circles  = VGroup()
        relu_circles = VGroup()
        for k in range(5):
            sig_circles.add(Circle(radius=CIRC_R, color=C_DIM, fill_opacity=0.15, stroke_width=2.2)
                            .move_to([LAYER_X[k], ROW_SIG, 0]))
            relu_circles.add(Circle(radius=CIRC_R, color=C_DIM, fill_opacity=0.15, stroke_width=2.2)
                             .move_to([LAYER_X[k], ROW_RELU, 0]))

        sig_row_lbl  = _tx("Sigmoid", FS_ROW, color=C_SIG, bold=True, w=1.8).next_to(sig_circles[4], LEFT, buff=0.38)
        relu_row_lbl = _tx("ReLU", FS_ROW, color=C_RELU, bold=True, w=1.4).next_to(relu_circles[4], LEFT, buff=0.38)

        hdr_lbls = VGroup(
            _tx("output", FS_SMALL, color=C_DIM, w=1.2).next_to(sig_circles[0], UP, buff=0.32),
            _tx("input", FS_SMALL, color=C_DIM, w=1.0).next_to(sig_circles[4], UP, buff=0.32),
        )
        flow_lbl = _tx(r"gradient flows $\leftarrow$", FS_SMALL, color=C_DIM, w=2.6).move_to([0, 0, 0])

        self.play(
            LaggedStart(*[GrowFromCenter(c) for c in sig_circles], lag_ratio=0.08),
            LaggedStart(*[GrowFromCenter(c) for c in relu_circles], lag_ratio=0.08),
            run_time=0.6,
        )
        self.play(FadeIn(sig_row_lbl), FadeIn(relu_row_lbl), FadeIn(hdr_lbls), FadeIn(flow_lbl), run_time=0.3)
        self.wait(0.3)

        sig_val_mobs, relu_val_mobs, arrow_mobs = [], [], []

        for k in range(5):
            sg, rg = SIG_GRADS[k], RELU_GRADS[k]
            sc_col, rc_col = SIG_COLS[k], RELU_COLS[k]

            self.play(
                sig_circles[k].animate.set_color(sc_col).set_fill(sc_col, opacity=0.35),
                relu_circles[k].animate.set_color(rc_col).set_fill(rc_col, opacity=0.35),
                run_time=0.30,
            )

            sg_txt = _tx(f"{sg:.3f}" if sg < 0.1 else f"{sg:.2f}", FS_VAL,
                         color=C_WHITE, bold=True, w=CIRC_R * 1.55).move_to(sig_circles[k])
            rg_txt = _tx(f"{rg:.1f}", FS_VAL, color=C_WHITE, bold=True, w=CIRC_R * 1.3).move_to(relu_circles[k])
            self.play(FadeIn(sg_txt), FadeIn(rg_txt), run_time=0.22)
            sig_val_mobs.append(sg_txt); relu_val_mobs.append(rg_txt)

            if k < 4:
                s_start = np.array([LAYER_X[k]     - CIRC_R - 0.05, ROW_SIG,  0])
                s_end   = np.array([LAYER_X[k + 1] + CIRC_R + 0.05, ROW_SIG,  0])
                r_start = np.array([LAYER_X[k]     - CIRC_R - 0.05, ROW_RELU, 0])
                r_end   = np.array([LAYER_X[k + 1] + CIRC_R + 0.05, ROW_RELU, 0])

                s_arr = Arrow(s_start, s_end, color=C_SIG,  stroke_width=2.2, buff=0, max_tip_length_to_length_ratio=0.22)
                r_arr = Arrow(r_start, r_end, color=C_RELU, stroke_width=2.2, buff=0, max_tip_length_to_length_ratio=0.22)
                s_lbl = _tx(r"$\times 0.25$", FS_MULT, color=C_SIG, w=1.1).next_to(s_arr, UP, buff=0.08)
                r_lbl = _tx(r"$\times 1.0$", FS_MULT, color=C_RELU, w=1.0).next_to(r_arr, DOWN, buff=0.08)

                self.play(Create(s_arr), Create(r_arr), FadeIn(s_lbl), FadeIn(r_lbl), run_time=0.30)
                arrow_mobs.extend([s_arr, r_arr, s_lbl, r_lbl])

            self.wait(0.12)

        sig_verdict = _tx(r"Input gradient: $%.4f$  (vanished)" % SIG_GRADS[-1], FS_VERD,
                          color=C_DEAD, bold=True, w=4.6).next_to(sig_circles[4], DOWN, buff=0.55)
        relu_verdict = _tx(r"Input gradient: $%.1f$  (survives)" % RELU_GRADS[-1], FS_VERD,
                           color=C_ALIVE, bold=True, w=4.6).next_to(relu_circles[4], DOWN, buff=0.55)
        self.play(FadeIn(sig_verdict), FadeIn(relu_verdict), run_time=0.5)

        caption = _tx(
            r"\mbox{Deep networks need ReLU (or variants) to keep gradients alive during training}",
            FS_CAP, color=C_DIM, w=13.0,
        ).to_edge(DOWN, buff=0.42)
        self.play(FadeIn(caption), run_time=0.4)
        self.wait(2.5)
