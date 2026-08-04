"""
W10 — Practical Fine-Tuning

  Phase 1: Pre-train vs fine-tune cost.
           Timeline: huge corpus → long pre-train → general model →
           small task dataset → fast fine-tune.

  Phase 2: Tokenization & input pipeline.
           Raw sentence → WordPiece tokens → IDs → [CLS]/[SEP] →
           embeddings + positional encoding.

  Phase 3: Fine-tuning strategies side by side.
           Feature extraction / Full fine-tune / Gradual unfreeze —
           training loss + val accuracy curves for each.

  Phase 4: Learning rate sensitivity.
           Too-high LR → catastrophic forgetting (val loss spikes).
           Sweet spot lr ≈ 2e-5 → smooth convergence.

Text uses LaTeX (Tex/MathTex); every label is fit-to-box and captions use \\mbox.

Render:
  ../../env/bin/manim -pql finetune.py FineTuning
  ../../env/bin/manim -qk  finetune.py FineTuning   # 4K master
"""

from manim import *
import numpy as np

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
FS_BOX   = 22
FS_TOKEN = 22
FS_ID    = 20
FS_STEP  = 20
FS_COST  = 22
FS_CAP   = 26
FS_HDR   = 24
FS_AXLBL = 19
FS_LEG   = 19
FS_NOTE  = 19


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


def _stack(lines, fs, colors, math=False, buff=0.10, w=None):
    grp = VGroup()
    for i, ln in enumerate(lines):
        col = colors[i] if isinstance(colors, list) else colors
        grp.add(_tx(ln, fs, color=col, math=math, w=w))
    grp.arrange(DOWN, buff=buff)
    return grp


def _box(label, pos, color, w=1.60, h=0.48, fs=FS_BOX, bold=False):
    rect = RoundedRectangle(
        width=w, height=h, corner_radius=0.07,
        fill_color=color, fill_opacity=0.16, stroke_color=color, stroke_width=1.9,
    ).move_to(pos)
    if "\n" in label:
        txt = _stack(label.split("\n"), fs, color, buff=0.06, w=w * 0.88)
    else:
        txt = _tx(label, fs, color=color, bold=bold, w=w * 0.88)
    if txt.height > h * 0.82:
        txt.scale_to_fit_height(h * 0.82)
    txt.move_to(pos)
    return VGroup(rect, txt)


def _arrow(start, end, color=C_DIM, sw=2.0):
    return Arrow(start, end, buff=0.07, color=color, stroke_width=sw, max_tip_length_to_length_ratio=0.22)


class FineTuning(Scene):

    def construct(self):
        self.camera.background_color = C_BG
        self._phase1()
        self._phase2()
        self._phase3()
        self._phase4()

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 1 — Pre-train vs fine-tune cost
    # ══════════════════════════════════════════════════════════════════════════

    def _phase1(self):
        title = _tx("Practical Fine-Tuning", FS_TITLE, color=C_WHITE, bold=True, w=11).to_edge(UP, buff=0.28)
        sub = _tx("Transfer learning: pay the pre-training cost once, reuse for many tasks cheaply",
                  FS_SUB, color=C_DIM, w=13.0).next_to(title, DOWN, buff=0.14)
        self.play(FadeIn(title), FadeIn(sub), run_time=0.5)

        Y = 0.10
        nodes = [
            (-5.2, "Huge corpus\n(web-scale)",             C_DIM,   1.60),
            (-2.2, "Pre-training\n(days / GPU-months)",    C_BLUE,  2.05),
            ( 0.8, "General\nmodel",                       C_TEAL,  1.55),
            ( 3.0, "Task dataset\n(hundreds of examples)", C_AMBER, 2.05),
            ( 5.8, "Fine-tuned\nmodel",                    C_GREEN, 1.55),
        ]

        node_mobs = [_box(lbl, [x, Y, 0], col, w=w, h=0.78, fs=FS_BOX) for x, lbl, col, w in nodes]

        arrs = []
        for i in range(len(nodes) - 1):
            x0 = nodes[i][0]   + nodes[i][3]   / 2 + 0.06
            x1 = nodes[i+1][0] - nodes[i+1][3] / 2 - 0.06
            arrs.append(_arrow([x0, Y, 0], [x1, Y, 0], color=C_DIM))

        self.play(LaggedStart(*[FadeIn(m) for m in node_mobs], lag_ratio=0.15), run_time=0.65)
        self.play(LaggedStart(*[GrowArrow(a) for a in arrs], lag_ratio=0.15), run_time=0.55)

        cost_pre = _tx("millions of GPU-hours", FS_COST, color=C_RED, w=2.9).next_to(arrs[1], UP, buff=0.70)
        cost_ft = _tx("minutes on a laptop", FS_COST, color=C_GREEN, w=2.7).next_to(arrs[3], UP, buff=0.70)
        self.play(FadeIn(cost_pre), FadeIn(cost_ft), run_time=0.30)

        caption = _tx(r"\mbox{BERT, GPT, ViT --- pre-trained once, fine-tuned thousands of times}",
                      FS_CAP, color=C_DIM, w=12.5).to_edge(DOWN, buff=0.34)
        self.play(FadeIn(caption), run_time=0.30)
        self.wait(1.8)

        self._title  = title
        self._p1_all = VGroup(sub, *node_mobs, *arrs, cost_pre, cost_ft, caption)

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 2 — Tokenization & input pipeline
    # ══════════════════════════════════════════════════════════════════════════

    def _phase2(self):
        self.play(FadeOut(self._p1_all), run_time=0.40)

        sub = _tx(r"Input pipeline: raw text $\to$ tokens $\to$ IDs $\to$ embeddings $+$ PE",
                  FS_SUB, color=C_DIM, w=13.2).next_to(self._title, DOWN, buff=0.14)
        self.play(FadeIn(sub), run_time=0.28)

        tokens = ["[CLS]", "great", "film", "!", "[SEP]"]
        ids    = [101,      2307,    2143,  999,  102]
        N      = len(tokens)

        Y_RAW  =  1.40
        Y_TOK  =  0.55
        Y_ID   = -0.30
        Y_EMB  = -1.10

        tok_xs = np.linspace(-3.8, 3.8, N)
        tok_cols = [C_GOLD, C_BLUE, C_BLUE, C_BLUE, C_GOLD]

        raw_box = _box(r"``great film\,!''", [0.0, Y_RAW, 0], C_DIM, w=2.40, h=0.52, fs=FS_TOKEN)
        self.play(FadeIn(raw_box), run_time=0.25)

        tok_arrow = _arrow([0.0, Y_RAW - 0.27, 0], [0.0, Y_TOK + 0.28, 0], color=C_DIM)
        tok_step  = _tx("WordPiece tokeniser", FS_STEP, color=C_DIM, w=2.9).next_to(tok_arrow, RIGHT, buff=0.14)
        self.play(GrowArrow(tok_arrow), FadeIn(tok_step), run_time=0.28)

        tok_boxes = [
            _box(tokens[i].replace("[", r"[").replace("]", r"]"),
                 [tok_xs[i], Y_TOK, 0], tok_cols[i], w=0.92, h=0.46, fs=FS_TOKEN)
            for i in range(N)
        ]
        self.play(LaggedStart(*[FadeIn(t) for t in tok_boxes], lag_ratio=0.12), run_time=0.50)

        id_arrow = _arrow([0.0, Y_TOK - 0.25, 0], [0.0, Y_ID + 0.24, 0], color=C_DIM)
        id_lbl   = _tx("vocabulary lookup", FS_STEP, color=C_DIM, w=2.7).next_to(id_arrow, RIGHT, buff=0.14)
        self.play(GrowArrow(id_arrow), FadeIn(id_lbl), run_time=0.25)

        id_boxes = [
            _box(str(ids[i]), [tok_xs[i], Y_ID, 0], C_DIM, w=0.82, h=0.42, fs=FS_ID)
            for i in range(N)
        ]
        self.play(LaggedStart(*[FadeIn(b) for b in id_boxes], lag_ratio=0.10), run_time=0.40)

        emb_arrow = _arrow([0.0, Y_ID - 0.23, 0], [0.0, Y_EMB + 0.26, 0], color=C_DIM)
        emb_lbl   = _tx(r"embedding table $+$ PE", FS_STEP, color=C_DIM, w=3.0).next_to(emb_arrow, RIGHT, buff=0.14)
        self.play(GrowArrow(emb_arrow), FadeIn(emb_lbl), run_time=0.25)

        CELL = 0.15
        D    = 8
        emb_grids = []
        for i in range(N):
            np.random.seed(ids[i] % 50)
            vals = np.random.randn(D)
            row  = VGroup()
            for d in range(D):
                v   = float(np.tanh(vals[d]))
                col = interpolate_color(ManimColor(C_PURP), ManimColor(C_TEAL), (v + 1) / 2)
                sq  = Square(side_length=CELL, fill_color=col, fill_opacity=0.82,
                             stroke_color=C_DIM, stroke_width=0.3
                             ).move_to([tok_xs[i] + (d - D / 2 + 0.5) * CELL, Y_EMB, 0])
                row.add(sq)
            emb_grids.append(row)

        self.play(LaggedStart(*[FadeIn(g) for g in emb_grids], lag_ratio=0.10), run_time=0.50)

        caption = _tx(
            r"\mbox{[CLS] and [SEP] are special tokens added by the tokeniser "
            r"$\;\cdot\;$ [CLS] output is used for classification}",
            FS_CAP, color=C_DIM, w=13.2,
        ).to_edge(DOWN, buff=0.34)
        self.play(FadeIn(caption), run_time=0.30)
        self.wait(1.8)

        self._p2_all = VGroup(
            sub, raw_box, tok_arrow, tok_step, *tok_boxes,
            id_arrow, id_lbl, *id_boxes,
            emb_arrow, emb_lbl, *emb_grids, caption,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 3 — Fine-tuning strategies
    # ══════════════════════════════════════════════════════════════════════════

    def _phase3(self):
        self.play(FadeOut(self._p2_all), run_time=0.40)

        sub = _tx("Fine-tuning strategies --- how much of the backbone to update",
                  FS_SUB, color=C_DIM, w=13.0).next_to(self._title, DOWN, buff=0.14)
        self.play(FadeIn(sub), run_time=0.28)

        epochs = np.arange(1, 21)

        def _exp(base, decay, noise, seed, offset=0.0):
            rng = np.random.default_rng(seed)
            return np.clip(base * np.exp(-decay * epochs) + noise * rng.random(20) + offset, 0.05, 2.0)

        strategies = [
            {"title": "Feature Extraction", "color": C_BLUE,  "cx": -4.0,
             "train": _exp(1.6, 0.08, 0.05, 1, 0.25), "val": _exp(1.5, 0.07, 0.06, 2, 0.30),
             "note": "backbone frozen\nonly head trains"},
            {"title": "Full Fine-Tune", "color": C_AMBER, "cx": 0.0,
             "train": _exp(1.6, 0.13, 0.04, 3), "val": _exp(1.5, 0.09, 0.06, 4, 0.18),
             "note": "all layers update\nhigh LR risk"},
            {"title": "Gradual Unfreeze", "color": C_GREEN, "cx": 4.0,
             "train": _exp(1.6, 0.12, 0.04, 5, 0.05), "val": _exp(1.5, 0.11, 0.04, 6, 0.06),
             "note": "unfreeze layer by layer\nbest for small data"},
        ]

        AX_W = 2.80
        AX_H = 1.80
        Y_TOP = 0.70
        all_mobs = VGroup()

        for s in strategies:
            cx = s["cx"]
            ax = Axes(
                x_range=[0, 21, 5], y_range=[0, 2.2, 0.5], x_length=AX_W, y_length=AX_H,
                axis_config={"color": C_DIM, "stroke_width": 1.4, "include_ticks": False},
            ).move_to([cx, Y_TOP - AX_H / 2, 0])

            hdr = _tx(s["title"], FS_HDR, color=s["color"], bold=True, w=3.4).next_to(ax, UP, buff=0.14)
            x_lbl = _tx("Epochs", FS_AXLBL, color=C_DIM, w=1.4).next_to(ax, DOWN, buff=0.08)
            y_lbl = _tx("Loss", FS_AXLBL, color=C_DIM, w=1.2).rotate(PI / 2).next_to(ax, LEFT, buff=0.06)

            self.play(Create(ax), FadeIn(hdr), FadeIn(x_lbl), FadeIn(y_lbl), run_time=0.35)

            def _pts(arr):
                return [ax.c2p(e, arr[i]) for i, e in enumerate(epochs)]

            t_path = VMobject(stroke_color=C_BLUE, stroke_width=2.6)
            v_path = VMobject(stroke_color=C_RED,  stroke_width=2.6)
            t_path.set_points_as_corners(_pts(s["train"]))
            v_path.set_points_as_corners(_pts(s["val"]))
            self.play(Create(t_path), Create(v_path), run_time=0.60)

            leg = VGroup(
                VGroup(Line([0,0,0],[0.28,0,0], stroke_color=C_BLUE, stroke_width=2.6),
                       _tx("train", FS_LEG, color=C_BLUE, w=1.0)).arrange(RIGHT, buff=0.09),
                VGroup(Line([0,0,0],[0.28,0,0], stroke_color=C_RED,  stroke_width=2.6),
                       _tx("val",   FS_LEG, color=C_RED, w=0.8)).arrange(RIGHT, buff=0.09),
            ).arrange(RIGHT, buff=0.24).next_to(ax, DOWN, buff=0.34)
            note = _stack(s["note"].split("\n"), FS_NOTE, s["color"], buff=0.08, w=3.2).next_to(leg, DOWN, buff=0.14)

            self.play(FadeIn(leg), FadeIn(note), run_time=0.25)
            all_mobs.add(ax, hdr, x_lbl, y_lbl, t_path, v_path, leg, note)

        caption = _tx(
            r"\mbox{gradual unfreeze maintains pre-trained representations while adapting to the new task}",
            FS_CAP, color=C_DIM, w=13.2,
        ).to_edge(DOWN, buff=0.30)
        self.play(FadeIn(caption), run_time=0.30)
        self.wait(2.0)

        self._p3_all = VGroup(sub, all_mobs, caption)

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 4 — Learning rate sensitivity
    # ══════════════════════════════════════════════════════════════════════════

    def _phase4(self):
        self.play(FadeOut(self._p3_all), run_time=0.40)

        sub = _tx("Learning rate sensitivity --- too high causes catastrophic forgetting",
                  FS_SUB, color=C_DIM, w=13.2).next_to(self._title, DOWN, buff=0.14)
        self.play(FadeIn(sub), run_time=0.28)

        epochs = np.arange(1, 31)

        def _loss(base, decay, noise, seed, offset=0.0):
            rng = np.random.default_rng(seed)
            return np.clip(base * np.exp(-decay * epochs) + noise * rng.random(30) + offset, 0.04, 2.5)

        rng = np.random.default_rng(20)
        val_high = np.concatenate([
            _loss(1.5, 0.18, 0.04, 10)[:8],
            np.clip(1.2 + 0.06 * np.arange(22) + 0.05 * rng.random(22), 0.8, 2.5),
        ])
        train_high = _loss(1.6, 0.15, 0.04, 11)
        train_sweet = _loss(1.6, 0.11, 0.04, 12, 0.04)
        val_sweet   = _loss(1.5, 0.10, 0.04, 13, 0.05)
        train_low = _loss(1.6, 0.03, 0.03, 14, 0.30)
        val_low   = _loss(1.5, 0.025, 0.03, 15, 0.35)

        lr_specs = [
            (r"lr $= 10^{-3}$ (too high)",         C_RED,   -4.0, train_high, val_high),
            (r"lr $= 2\times10^{-5}$ (sweet spot)", C_GREEN,  0.0, train_sweet, val_sweet),
            (r"lr $= 10^{-7}$ (too low)",          C_AMBER,   4.0, train_low,  val_low),
        ]

        AX_W = 2.80
        AX_H = 1.80
        Y_TOP = 0.70
        all_mobs = VGroup()

        for lbl, col, cx, t_loss, v_loss in lr_specs:
            ax = Axes(
                x_range=[0, 31, 10], y_range=[0, 2.5, 0.5], x_length=AX_W, y_length=AX_H,
                axis_config={"color": C_DIM, "stroke_width": 1.4, "include_ticks": False},
            ).move_to([cx, Y_TOP - AX_H / 2, 0])

            hdr   = _tx(lbl, FS_HDR, color=col, bold=True, w=3.5).next_to(ax, UP, buff=0.14)
            x_lbl = _tx("Epochs", FS_AXLBL, color=C_DIM, w=1.4).next_to(ax, DOWN, buff=0.08)
            y_lbl = _tx("Val Loss", FS_AXLBL, color=C_DIM, w=1.5).rotate(PI / 2).next_to(ax, LEFT, buff=0.06)

            self.play(Create(ax), FadeIn(hdr), FadeIn(x_lbl), FadeIn(y_lbl), run_time=0.32)

            def _pts(arr):
                return [ax.c2p(e, arr[i]) for i, e in enumerate(epochs)]

            t_path = VMobject(stroke_color=C_BLUE, stroke_width=2.6)
            v_path = VMobject(stroke_color=col,    stroke_width=2.6)
            t_path.set_points_as_corners(_pts(t_loss))
            v_path.set_points_as_corners(_pts(v_loss))
            self.play(Create(t_path), Create(v_path), run_time=0.65)

            leg = VGroup(
                VGroup(Line([0,0,0],[0.28,0,0], stroke_color=C_BLUE, stroke_width=2.6),
                       _tx("train", FS_LEG, color=C_BLUE, w=1.0)).arrange(RIGHT, buff=0.09),
                VGroup(Line([0,0,0],[0.28,0,0], stroke_color=col, stroke_width=2.6),
                       _tx("val",   FS_LEG, color=col, w=0.8)).arrange(RIGHT, buff=0.09),
            ).arrange(RIGHT, buff=0.24).next_to(ax, DOWN, buff=0.34)
            self.play(FadeIn(leg), run_time=0.22)
            all_mobs.add(ax, hdr, x_lbl, y_lbl, t_path, v_path, leg)

        forget_note = _stack(["catastrophic", "forgetting"], FS_NOTE, C_RED, buff=0.08, w=2.4)\
            .move_to([-4.0, -2.05, 0])
        self.play(FadeIn(forget_note), run_time=0.25)

        caption = _tx(
            r"\mbox{rule of thumb: lr $\approx 2\times10^{-5}$ for BERT "
            r"$\;\cdot\;$ use a scheduler (linear warmup $+$ decay) for best results}",
            FS_CAP, color=C_DIM, w=13.2,
        ).to_edge(DOWN, buff=0.30)
        self.play(FadeIn(caption), run_time=0.30)
        self.wait(2.5)
