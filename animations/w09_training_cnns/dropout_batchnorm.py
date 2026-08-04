"""
W09 — Dropout & Batch Normalization

  Phase 1: Dropout — a 4-node hidden layer with neurons randomly dropped
           during training (red X marks). Side-by-side: training (sparse)
           vs inference (all active, weights scaled by keep-prob).
  Phase 2: Overfitting context — two loss curve pairs showing train vs
           validation: without dropout (val diverges) and with dropout
           (both converge). Illustrates *why* dropout helps.
  Phase 3: Batch Normalization — a column of raw activations (spread,
           shifted) is normalised to mean≈0 / std≈1, then scaled and
           shifted by learned gamma/beta. Before/after bar distributions.

Text uses LaTeX (Tex/MathTex) — μ, σ, γ, β and x̂ now typeset properly.
Every label is fit-to-box; captions use \\mbox.

Render:
  ../../env/bin/manim -pql dropout_batchnorm.py DropoutBatchNorm
  ../../env/bin/manim -qk  dropout_batchnorm.py DropoutBatchNorm   # 4K master
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
C_DEAD  = "#3d444d"

NODE_R  = 0.34

# ── Type sizes ─────────────────────────────────────────────────────────────────
FS_TITLE = 44
FS_SUB   = 28
FS_PLBL  = 24
FS_NOTE  = 20
FS_CAP   = 26
FS_AXLBL = 19
FS_LEG   = 19
FS_PANEL = 24
FS_GAP   = 22
FS_CHART = 22
FS_STEP  = 21
FS_FORM  = 21


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


# ── Shared node / edge helpers ─────────────────────────────────────────────────

def _node(pos, color, fill_op=0.18, r=NODE_R):
    return Circle(radius=r, color=color, fill_color=color,
                  fill_opacity=fill_op, stroke_width=2.4).move_to(pos)


def _edge(p1, p2, color=C_DIM, width=1.2):
    return Line(np.array(p1), np.array(p2), stroke_color=color, stroke_width=width)


def _cross(pos, color=C_RED, scale=0.28):
    a = Line([-scale, -scale, 0], [scale,  scale, 0], stroke_color=color, stroke_width=4.0)
    b = Line([-scale,  scale, 0], [scale, -scale, 0], stroke_color=color, stroke_width=4.0)
    return VGroup(a, b).move_to(pos)


class DropoutBatchNorm(Scene):

    def construct(self):
        self.camera.background_color = C_BG
        self._phase1()
        self._phase2()
        self._phase3()

    # ── Phase 1: Dropout ──────────────────────────────────────────────────────

    def _phase1(self):
        title = _tx(r"Dropout \& Batch Normalization", FS_TITLE, color=C_WHITE, bold=True, w=12.0).to_edge(UP, buff=0.28)
        sub = _tx("Dropout --- randomly deactivate neurons to prevent co-adaptation",
                  FS_SUB, color=C_DIM, w=13.2).next_to(title, DOWN, buff=0.14)
        self.play(FadeIn(title), FadeIn(sub), run_time=0.5)

        L_CX = -3.3
        R_CX =  3.3

        def _panel(cx, dropped=None):
            x_in, x_hid, x_out = cx - 1.8, cx, cx + 1.8
            y_in  = [0.8, 0.0, -0.8]
            y_hid = [1.2, 0.4, -0.4, -1.2]
            y_out = [0.0]

            in_nodes  = VGroup(*[_node([x_in, y, 0], C_BLUE) for y in y_in])
            hid_nodes = VGroup()
            for i, y in enumerate(y_hid):
                col = C_DEAD if (dropped and i in dropped) else C_AMBER
                fo  = 0.55   if (dropped and i in dropped) else 0.20
                hid_nodes.add(_node([x_hid, y, 0], col, fill_op=fo))
            out_nodes = VGroup(*[_node([x_out, y, 0], C_GREEN) for y in y_out])

            edges_ih = VGroup()
            for i, yi in enumerate(y_in):
                for j, yj in enumerate(y_hid):
                    if dropped and j in dropped:
                        continue
                    edges_ih.add(_edge([x_in + NODE_R, yi, 0], [x_hid - NODE_R, yj, 0], color=C_DIM, width=0.9))
            edges_ho = VGroup()
            for j, yj in enumerate(y_hid):
                if dropped and j in dropped:
                    continue
                edges_ho.add(_edge([x_hid + NODE_R, yj, 0], [x_out - NODE_R, y_out[0], 0], color=C_DIM, width=0.9))

            return dict(in_nodes=in_nodes, hid_nodes=hid_nodes, out_nodes=out_nodes,
                        edges_ih=edges_ih, edges_ho=edges_ho,
                        positions=dict(hid=[[x_hid, y, 0] for y in y_hid]))

        DROPPED = [1, 3]
        train_p = _panel(L_CX, dropped=DROPPED)
        infer_p = _panel(R_CX, dropped=None)

        divider = DashedLine([0, 2.8, 0], [0, -2.8, 0], dash_length=0.12, color=C_DIM, stroke_width=1.2)

        lbl_train = _tx(r"Training  ($p = 0.5$)", FS_PLBL, color=C_AMBER, bold=True, w=3.4).move_to([L_CX, 2.35, 0])
        lbl_infer = _tx(r"Inference  (all active)", FS_PLBL, color=C_GREEN, bold=True, w=3.6).move_to([R_CX, 2.35, 0])

        self.play(FadeIn(divider), FadeIn(lbl_train), FadeIn(lbl_infer), run_time=0.40)

        self.play(LaggedStart(*[GrowFromCenter(n) for n in train_p["in_nodes"]], lag_ratio=0.15), run_time=0.45)
        self.play(LaggedStart(*[Create(e) for e in train_p["edges_ih"]], lag_ratio=0.05), run_time=0.50)
        self.play(LaggedStart(*[GrowFromCenter(n) for n in train_p["hid_nodes"]], lag_ratio=0.12), run_time=0.45)
        self.play(LaggedStart(*[Create(e) for e in train_p["edges_ho"]], lag_ratio=0.07), run_time=0.35)
        self.play(GrowFromCenter(train_p["out_nodes"][0]), run_time=0.28)

        crosses = VGroup(*[_cross(train_p["positions"]["hid"][i]) for i in DROPPED])
        self.play(LaggedStart(*[GrowFromCenter(x) for x in crosses], lag_ratio=0.3), run_time=0.45)

        drop_note = _tx(r"dropped  (output $= 0$)", FS_NOTE, color=C_RED, w=3.2).move_to([L_CX, -1.95, 0])
        self.play(FadeIn(drop_note), run_time=0.22)
        self.wait(0.4)

        self.play(LaggedStart(*[GrowFromCenter(n) for n in infer_p["in_nodes"]], lag_ratio=0.12), run_time=0.40)
        self.play(LaggedStart(*[Create(e) for e in infer_p["edges_ih"]], lag_ratio=0.04), run_time=0.45)
        self.play(LaggedStart(*[GrowFromCenter(n) for n in infer_p["hid_nodes"]], lag_ratio=0.12), run_time=0.40)
        self.play(LaggedStart(*[Create(e) for e in infer_p["edges_ho"]], lag_ratio=0.06), run_time=0.30)
        self.play(GrowFromCenter(infer_p["out_nodes"][0]), run_time=0.25)

        scale_note = _tx(r"weights scaled by $(1-p)$", FS_NOTE, color=C_BLUE, w=3.4).move_to([R_CX, -1.95, 0])
        self.play(FadeIn(scale_note), run_time=0.22)

        caption = _tx(
            r"\mbox{each training step uses a different random sub-network "
            r"$\to$ ensemble effect at test time}",
            FS_CAP, color=C_DIM, w=13.0,
        ).to_edge(DOWN, buff=0.32)
        self.play(FadeIn(caption), run_time=0.35)
        self.wait(1.8)

        self._title  = title
        self._p1_all = VGroup(
            sub, divider, lbl_train, lbl_infer,
            train_p["in_nodes"], train_p["hid_nodes"], train_p["out_nodes"],
            train_p["edges_ih"], train_p["edges_ho"], crosses, drop_note,
            infer_p["in_nodes"], infer_p["hid_nodes"], infer_p["out_nodes"],
            infer_p["edges_ih"], infer_p["edges_ho"], scale_note, caption,
        )

    # ── Phase 2: loss curves ──────────────────────────────────────────────────

    def _phase2(self):
        self.play(FadeOut(self._p1_all), run_time=0.40)

        sub = _tx("effect on training: dropout reduces the train/validation gap",
                  FS_SUB, color=C_DIM, w=13.0).next_to(self._title, DOWN, buff=0.14)
        self.play(FadeIn(sub), run_time=0.28)

        epochs = np.arange(1, 31)

        def _loss(base, decay, noise_amp, seed):
            rng = np.random.default_rng(seed)
            return base * np.exp(-decay * epochs) + noise_amp * rng.random(len(epochs))

        train_nd = np.clip(_loss(1.8, 0.14, 0.04, 1),        0.04, 2.0)
        val_nd   = np.clip(_loss(1.6, 0.06, 0.05, 2) + 0.25, 0.30, 2.0)
        train_do = np.clip(_loss(1.8, 0.10, 0.05, 3) + 0.05, 0.18, 2.0)
        val_do   = np.clip(_loss(1.6, 0.09, 0.04, 4) + 0.05, 0.20, 2.0)

        panel_specs = [
            ("Without Dropout", train_nd, val_nd, C_RED,   -3.4),
            ("With Dropout",    train_do, val_do, C_GREEN,  3.4),
        ]

        ax_w, ax_h = 4.4, 2.4
        y_top = 1.0
        all_mobs = VGroup()

        for title_txt, t_loss, v_loss, col, cx in panel_specs:
            ax = Axes(
                x_range=[0, 31, 10], y_range=[0, 2.2, 0.5], x_length=ax_w, y_length=ax_h,
                axis_config={"color": C_DIM, "stroke_width": 1.4, "include_ticks": False},
            ).move_to([cx, y_top - ax_h / 2, 0])

            x_lbl = _tx("Epochs", FS_AXLBL, color=C_DIM, w=1.4).next_to(ax, DOWN, buff=0.10)
            y_lbl = _tx("Loss", FS_AXLBL, color=C_DIM, w=1.2).rotate(PI / 2).next_to(ax, LEFT, buff=0.10)
            panel_lbl = _tx(title_txt, FS_PANEL, color=col, bold=True, w=3.4).next_to(ax, UP, buff=0.14)

            self.play(Create(ax), FadeIn(x_lbl), FadeIn(y_lbl), FadeIn(panel_lbl), run_time=0.45)

            def _pts(arr):
                return [ax.c2p(e, arr[i]) for i, e in enumerate(epochs)]

            train_path = VMobject(stroke_color=C_BLUE,  stroke_width=2.8)
            val_path   = VMobject(stroke_color=C_AMBER, stroke_width=2.8)
            train_path.set_points_as_corners(_pts(t_loss))
            val_path.set_points_as_corners(_pts(v_loss))

            self.play(Create(train_path), run_time=0.7)
            self.play(Create(val_path), run_time=0.7)

            leg_train = VGroup(
                Line([0, 0, 0], [0.3, 0, 0], stroke_color=C_BLUE, stroke_width=2.8),
                _tx("train", FS_LEG, color=C_BLUE, w=1.0),
            ).arrange(RIGHT, buff=0.10).next_to(ax, DOWN, buff=0.30).shift(LEFT * 0.5)
            leg_val = VGroup(
                Line([0, 0, 0], [0.3, 0, 0], stroke_color=C_AMBER, stroke_width=2.8),
                _tx("val", FS_LEG, color=C_AMBER, w=0.8),
            ).arrange(RIGHT, buff=0.10).next_to(leg_train, RIGHT, buff=0.40)

            self.play(FadeIn(leg_train), FadeIn(leg_val), run_time=0.22)
            all_mobs.add(ax, x_lbl, y_lbl, panel_lbl, train_path, val_path, leg_train, leg_val)

        gap_note = _tx(r"large gap $=$ overfitting", FS_GAP, color=C_RED, w=3.6).move_to([-3.4, -2.55, 0])
        close_note = _tx("gap closed by dropout", FS_GAP, color=C_GREEN, w=3.6).move_to([3.4, -2.55, 0])
        self.play(FadeIn(gap_note), FadeIn(close_note), run_time=0.30)
        self.wait(2.0)

        self._p2_all = VGroup(all_mobs, sub, gap_note, close_note)

    # ── Phase 3: Batch Normalization ──────────────────────────────────────────

    def _phase3(self):
        self.play(FadeOut(self._p2_all), run_time=0.40)

        sub = _tx("Batch Normalization --- normalise activations within each mini-batch",
                  FS_SUB, color=C_DIM, w=13.0).next_to(self._title, DOWN, buff=0.14)
        self.play(FadeIn(sub), run_time=0.28)

        raw_vals = np.array([8.5, 2.1, 11.3, -1.4, 6.0, 14.2, 0.3, 4.8])
        mu  = raw_vals.mean()
        sig = raw_vals.std() + 1e-5
        norm_vals = (raw_vals - mu) / sig
        gamma, beta = 1.5, 0.5
        out_vals = gamma * norm_vals + beta

        N       = len(raw_vals)
        bar_w   = 0.30
        bar_gap = 0.12
        step    = bar_w + bar_gap

        def _bar_chart(values, origin, color, y_scale=0.12, label=None):
            bars = VGroup()
            for i, v in enumerate(values):
                h = max(abs(v) * y_scale, 0.04)
                bar = Rectangle(width=bar_w, height=h, fill_color=color, fill_opacity=0.75,
                                stroke_color=color, stroke_width=1.0)
                bx = origin[0] + i * step
                by = origin[1] + (h / 2 if v >= 0 else -h / 2)
                bar.move_to([bx, by, 0])
                bars.add(bar)
            grp = VGroup(bars)
            if label is not None:
                lbl = _tx(label, FS_CHART, color=color, w=3.0).next_to(bars, UP, buff=0.14)
                grp.add(lbl)
            return grp

        def _cx_origin(cx, n):
            return cx - (n - 1) * step / 2

        raw_origin  = [_cx_origin(-4.6, N), -0.5, 0.0]
        norm_origin = [_cx_origin( 0.0, N), -0.5, 0.0]
        out_origin  = [_cx_origin( 4.6, N), -0.5, 0.0]

        raw_chart  = _bar_chart(raw_vals,  raw_origin,  C_RED,   y_scale=0.10, label="Raw activations")
        norm_chart = _bar_chart(norm_vals, norm_origin, C_BLUE,  y_scale=0.45, label="After normalise")
        out_chart  = _bar_chart(out_vals,  out_origin,  C_GREEN, y_scale=0.30, label=r"After $\gamma x + \beta$")

        def _zero_line(origin, n, color):
            x0 = origin[0] - step / 2
            x1 = origin[0] + (n - 1) * step + step / 2
            return DashedLine([x0, origin[1], 0], [x1, origin[1], 0],
                              dash_length=0.10, stroke_color=color, stroke_width=1.0)

        z0_raw  = _zero_line(raw_origin,  N, C_DIM)
        z0_norm = _zero_line(norm_origin, N, C_DIM)
        z0_out  = _zero_line(out_origin,  N, C_DIM)

        arr1 = Arrow([-2.85, -0.5, 0], [-1.80, -0.5, 0], buff=0.05, color=C_DIM, stroke_width=2.2, max_tip_length_to_length_ratio=0.22)
        arr2 = Arrow([ 1.80, -0.5, 0], [ 2.85, -0.5, 0], buff=0.05, color=C_DIM, stroke_width=2.2, max_tip_length_to_length_ratio=0.22)

        def _step_lbl(text, cx, color, math=False):
            return _tx(text, FS_STEP, color=color, math=math, w=3.0).move_to([cx, -2.15, 0])

        s1 = _step_lbl(r"$x$  (any scale)", -4.6, C_RED)
        s2 = _step_lbl(r"\hat{x} = (x-\mu)/\sigma", 0.0, C_BLUE, math=True)
        s3 = _step_lbl(r"y = \gamma\,\hat{x} + \beta", 4.6, C_GREEN, math=True)

        f_norm = _tx(r"$\mu = %.1f,\ \ \sigma = %.1f$" % (mu, sig), FS_FORM, color=C_DIM, w=3.0).move_to([0.0, -2.60, 0])
        f_out  = _tx(r"$\gamma = %.1f,\ \ \beta = %.1f$  (learned)" % (gamma, beta), FS_FORM, color=C_DIM, w=3.2).move_to([4.6, -2.60, 0])

        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in raw_chart[0]], lag_ratio=0.07), FadeIn(z0_raw), run_time=0.65)
        self.play(FadeIn(raw_chart[1]), FadeIn(s1), run_time=0.25)
        self.play(GrowArrow(arr1), run_time=0.25)
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in norm_chart[0]], lag_ratio=0.07), FadeIn(z0_norm), run_time=0.65)
        self.play(FadeIn(norm_chart[1]), FadeIn(s2), FadeIn(f_norm), run_time=0.28)
        self.wait(0.35)
        self.play(GrowArrow(arr2), run_time=0.25)
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in out_chart[0]], lag_ratio=0.07), FadeIn(z0_out), run_time=0.65)
        self.play(FadeIn(out_chart[1]), FadeIn(s3), FadeIn(f_out), run_time=0.28)

        caption = _tx(
            r"\mbox{BatchNorm stabilises training $\;\cdot\;$ allows higher learning rates "
            r"$\;\cdot\;$ mild regularisation effect}",
            FS_CAP, color=C_DIM, w=13.0,
        ).to_edge(DOWN, buff=0.30)
        self.play(FadeIn(caption), run_time=0.35)
        self.wait(2.5)
