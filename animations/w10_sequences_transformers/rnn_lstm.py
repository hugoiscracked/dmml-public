"""
W10 — RNN vs LSTM

  Phase 1: RNN unrolled across 5 time steps.
           Hidden state h_t flows right across steps; x_t feeds in
           from below. Vanishing gradient shown as magnitude bars
           shrinking during backprop.

  Phase 2: LSTM cell internals.
           Forget / input / cell-update / output gates labelled.
           Cell state C_t flows as a straight "highway" across the top.
           Contrast with RNN: short memory vs long-term cell state.

Text uses LaTeX (Tex/MathTex) — subscripts (h_t, C_{t-1}, x_t) now typeset
properly. Every label is fit-to-box; captions use \\mbox.

Render:
  ../../env/bin/manim -pql rnn_lstm.py RNNvsLSTM
  ../../env/bin/manim -qk  rnn_lstm.py RNNvsLSTM   # 4K master
"""

from manim import *
import numpy as np

# ── Palette ───────────────────────────────────────────────────────────────────
C_BG    = "#0d1117"
C_WHITE = "#ffffff"
C_DIM   = "#8b949e"
C_BLUE  = "#58a6ff"
C_AMBER = "#ffa657"
C_GREEN = "#3fb950"
C_RED   = "#f78166"
C_GOLD  = "#e3b341"
C_TEAL  = "#2dd4bf"
C_PURP  = "#bc8cff"
C_DEAD  = "#3d444d"

# ── Type sizes ─────────────────────────────────────────────────────────────────
FS_TITLE = 46
FS_SUB   = 28
FS_CELL  = 22
FS_TOKEN = 22
FS_H0    = 20
FS_HLBL  = 24
FS_VGT   = 23
FS_BAR   = 18
FS_NOTE  = 24
FS_GATE  = 20
FS_HW    = 23
FS_END   = 23
FS_IO    = 25
FS_ANN   = 18
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


def _stack(lines, fs, colors, buff=0.07, w=None):
    grp = VGroup()
    for i, ln in enumerate(lines):
        col = colors[i] if isinstance(colors, list) else colors
        grp.add(_tx(ln, fs, color=col, w=w))
    grp.arrange(DOWN, buff=buff)
    return grp


def _box(label, pos, color, w=0.90, h=0.58, fs=FS_CELL):
    rect = RoundedRectangle(
        width=w, height=h, corner_radius=0.08,
        fill_color=color, fill_opacity=0.18, stroke_color=color, stroke_width=2.1,
    ).move_to(pos)
    if "\n" in label:
        txt = _stack(label.split("\n"), fs, color, buff=0.05, w=w * 0.86)
    else:
        txt = _tx(label, fs, color=color, w=w * 0.86)
    if txt.height > h * 0.82:
        txt.scale_to_fit_height(h * 0.82)
    txt.move_to(pos)
    return VGroup(rect, txt)


def _arrow(start, end, color=C_DIM, stroke_w=2.2):
    return Arrow(start, end, buff=0.08, color=color, stroke_width=stroke_w,
                 max_tip_length_to_length_ratio=0.22)


class RNNvsLSTM(Scene):

    def construct(self):
        self.camera.background_color = C_BG
        self._phase1()
        self._phase2()

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 1 — Unrolled RNN + vanishing gradient
    # ══════════════════════════════════════════════════════════════════════════

    def _phase1(self):
        title = _tx("RNN  vs  LSTM", FS_TITLE, color=C_WHITE, bold=True, w=8).to_edge(UP, buff=0.28)
        sub = _tx("RNN --- one cell reused each time step; the hidden state carries memory",
                  FS_SUB, color=C_DIM, w=13.2).next_to(title, DOWN, buff=0.14)
        self.play(FadeIn(title), FadeIn(sub), run_time=0.5)

        T = 5
        xs_tok = ["The", "cat", "sat", "on", "mat"]
        CX0    = -4.8
        STEP   = 2.20
        Y_CELL =  0.20
        Y_IN   = -1.10
        Y_H    =  1.20

        cells, x_nodes = [], []
        for t in range(T):
            cx = CX0 + t * STEP
            cells.append(_box("RNN\n$t=%d$" % t, [cx, Y_CELL, 0], C_BLUE, w=0.94, h=0.66, fs=FS_CELL))
            x_nodes.append(_box(xs_tok[t], [cx, Y_IN, 0], C_AMBER, w=0.80, h=0.42, fs=FS_TOKEN))

        h0 = _box(r"$h_0=0$", [CX0 - STEP * 0.75, Y_CELL, 0], C_DIM, w=0.82, h=0.42, fs=FS_H0)

        self.play(FadeIn(h0), run_time=0.25)
        for t in range(T):
            self.play(FadeIn(cells[t]), FadeIn(x_nodes[t]), run_time=0.28)

        arr_x, arr_hh, h_lbls = [], [], []
        for t in range(T):
            cx = CX0 + t * STEP
            arr_x.append(_arrow([cx, Y_IN + 0.22, 0], [cx, Y_CELL - 0.34, 0], color=C_AMBER))
            src_x = (CX0 - STEP * 0.75 + 0.40) if t == 0 else (CX0 + (t - 1) * STEP + 0.47)
            arr_hh.append(_arrow([src_x, Y_CELL, 0], [cx - 0.47, Y_CELL, 0], color=C_TEAL))
            h_lbls.append(_tx(r"h_{%d}" % (t + 1), FS_HLBL, color=C_TEAL, math=True).move_to([cx + 0.74, Y_H, 0]))

        arr_hout = []
        for t in range(T):
            cx = CX0 + t * STEP
            arr_hout.append(_arrow([cx + 0.47, Y_CELL, 0],
                                   [cx + STEP - 0.47 if t < T - 1 else cx + 0.9, Y_CELL, 0], color=C_TEAL))

        self.play(
            LaggedStart(
                *[AnimationGroup(GrowArrow(arr_x[t]), GrowArrow(arr_hh[t]),
                                 GrowArrow(arr_hout[t]), FadeIn(h_lbls[t])) for t in range(T)],
                lag_ratio=0.30),
            run_time=1.8,
        )
        self.wait(0.4)

        vg_title = _tx(r"Backprop Through Time $\to$ vanishing gradient", FS_VGT, color=C_RED, w=6.5)\
            .to_edge(DOWN, buff=1.05)
        self.play(FadeIn(vg_title), run_time=0.28)

        grad_mags  = [0.80, 0.52, 0.28, 0.12, 0.04]
        bar_w      = 0.30
        BASELINE_Y = Y_IN - 1.10
        grad_bars  = []
        for t in range(T):
            cx  = CX0 + t * STEP
            mag = grad_mags[T - 1 - t]
            col = interpolate_color(ManimColor(C_DEAD), ManimColor(C_RED), mag)
            h_b = max(mag * 0.80, 0.03)
            bar = Rectangle(width=bar_w, height=h_b, fill_color=col, fill_opacity=0.85,
                            stroke_color=col, stroke_width=1.0).move_to([cx, BASELINE_Y + h_b / 2, 0])
            lbl = _tx(f"{mag:.2f}", FS_BAR, color=col, w=0.6).next_to(bar, DOWN, buff=0.06)
            grad_bars.append(VGroup(bar, lbl))

        self.play(
            LaggedStart(*[GrowFromEdge(vg[0], DOWN) for vg in grad_bars], lag_ratio=0.15),
            LaggedStart(*[FadeIn(vg[1]) for vg in grad_bars], lag_ratio=0.15),
            run_time=1.0,
        )

        vanish_note = _tx(
            r"\mbox{early steps receive near-zero gradient $\to$ cannot learn long-range dependencies}",
            FS_NOTE, color=C_DIM, w=13.0).to_edge(DOWN, buff=0.34)
        self.play(FadeIn(vanish_note), run_time=0.30)
        self.wait(1.8)

        self._title  = title
        self._p1_all = VGroup(
            sub, h0, *cells, *x_nodes, *arr_x, *arr_hh, *arr_hout, *h_lbls,
            *grad_bars, vg_title, vanish_note,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 2 — LSTM cell internals
    # ══════════════════════════════════════════════════════════════════════════

    def _phase2(self):
        self.play(FadeOut(self._p1_all), run_time=0.40)

        sub = _tx("LSTM --- gates separate long-term cell state from short-term hidden state",
                  FS_SUB, color=C_DIM, w=13.2).next_to(self._title, DOWN, buff=0.14)
        self.play(FadeIn(sub), run_time=0.28)

        CX      = 0.0
        CY      = -0.20
        GATE_W  = 1.10
        GATE_H  = 0.60
        GAP     = 1.62

        g_xs = [CX + (i - 1.5) * GAP for i in range(4)]
        gate_defs = [
            ("forget\ngate $\\sigma$",   C_RED,   g_xs[0]),
            ("input\ngate $\\sigma$",    C_BLUE,  g_xs[1]),
            ("cell\nupdate $\\tanh$",    C_PURP,  g_xs[2]),
            ("output\ngate $\\sigma$",   C_GREEN, g_xs[3]),
        ]

        Y_HIGHWAY = CY + 1.30
        Y_GATE    = CY
        Y_INPUT   = CY - 1.25

        highway_line = DashedLine([-5.5, Y_HIGHWAY, 0], [5.5, Y_HIGHWAY, 0],
                                  dash_length=0.18, stroke_color=C_GOLD, stroke_width=3.0)
        hw_lbl = _tx(r"Cell state $C_t$ (long-term memory)", FS_HW, color=C_GOLD, w=5.0)\
            .move_to([0.0, Y_HIGHWAY + 0.36, 0])
        c_in  = _tx(r"C_{t-1}", FS_END, color=C_GOLD, math=True).move_to([-4.7, Y_HIGHWAY - 0.34, 0])
        c_out = _tx(r"C_t",     FS_END, color=C_GOLD, math=True).move_to([ 4.7, Y_HIGHWAY - 0.34, 0])

        self.play(Create(highway_line), FadeIn(hw_lbl), FadeIn(c_in), FadeIn(c_out), run_time=0.55)

        gate_mobs = [_box(label, [gx, Y_GATE, 0], color, w=GATE_W, h=GATE_H, fs=FS_GATE)
                     for label, color, gx in gate_defs]
        self.play(LaggedStart(*[FadeIn(g) for g in gate_mobs], lag_ratio=0.20), run_time=0.70)

        x_in_lbl  = _tx(r"x_t",     FS_IO, color=C_AMBER, math=True).move_to([-1.25, Y_INPUT, 0])
        h_in_lbl  = _tx(r"h_{t-1}", FS_IO, color=C_TEAL,  math=True).move_to([ 1.25, Y_INPUT, 0])
        h_out_lbl = _tx(r"h_t",     FS_IO, color=C_TEAL,  math=True).move_to([ 5.1,  Y_GATE,  0])
        self.play(FadeIn(x_in_lbl), FadeIn(h_in_lbl), FadeIn(h_out_lbl), run_time=0.28)

        in_arrows = VGroup()
        for _, _, gx in gate_defs:
            in_arrows.add(_arrow([gx - 0.22, Y_INPUT + 0.28, 0], [gx - 0.22, Y_GATE - 0.32, 0], color=C_AMBER))
            in_arrows.add(_arrow([gx + 0.22, Y_INPUT + 0.28, 0], [gx + 0.22, Y_GATE - 0.32, 0], color=C_TEAL))
        self.play(LaggedStart(*[GrowArrow(a) for a in in_arrows], lag_ratio=0.08), run_time=0.70)

        hw_arrows = VGroup()
        for _, color, gx in gate_defs:
            hw_arrows.add(_arrow([gx, Y_GATE + GATE_H / 2 + 0.05, 0], [gx, Y_HIGHWAY - 0.05, 0], color=color))
        self.play(LaggedStart(*[GrowArrow(a) for a in hw_arrows], lag_ratio=0.20), run_time=0.60)

        ann_y = Y_INPUT - 0.60
        ann_forget = _stack(["what to forget", r"from $C_{t-1}$"], FS_ANN, C_RED,   buff=0.06, w=1.45).move_to([g_xs[0], ann_y, 0])
        ann_input  = _stack(["what new info", "to write"],          FS_ANN, C_BLUE,  buff=0.06, w=1.45).move_to([g_xs[1], ann_y, 0])
        ann_cell   = _stack(["candidate", "values"],                FS_ANN, C_PURP,  buff=0.06, w=1.45).move_to([g_xs[2], ann_y, 0])
        ann_out    = _stack(["what to", r"expose as $h_t$"],        FS_ANN, C_GREEN, buff=0.06, w=1.45).move_to([g_xs[3], ann_y, 0])
        self.play(
            LaggedStart(FadeIn(ann_forget), FadeIn(ann_input), FadeIn(ann_cell), FadeIn(ann_out), lag_ratio=0.20),
            run_time=0.65,
        )

        caption = _tx(
            r"\mbox{cell state flows with minimal modification "
            r"$\;\cdot\;$ gates learn when to remember, forget, and output}",
            FS_CAP, color=C_DIM, w=13.2,
        ).to_edge(DOWN, buff=0.34)
        self.play(FadeIn(caption), run_time=0.35)
        self.wait(2.5)
