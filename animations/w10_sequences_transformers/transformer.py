"""
W10 — Transformer Architecture

  Phase 1: Positional encoding.
           Token embeddings appear; sine/cosine positional encodings
           are added. Encoding pattern shown as a colour grid.

  Phase 2: One encoder layer.
           QKV projection → 2 parallel attention heads → concat →
           linear → residual add & norm → feed-forward → residual
           add & norm.

  Phase 3: Encoder stack → classification head.
           3 stacked encoder layers; [CLS] token flows out the top
           → linear → softmax → sentiment label (positive/negative).

Render:
  ../../env/bin/manim -pql transformer.py Transformer
  ../../env/bin/manim -pqh transformer.py Transformer
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

WORDS = ["[CLS]", "great", "film", "!"]
N     = len(WORDS)
D_EMB = 8   # embedding dimension (for display grid)


def _block(label, pos, color, w=2.20, h=0.50, font_size=12):
    rect = RoundedRectangle(
        width=w, height=h, corner_radius=0.07,
        fill_color=color, fill_opacity=0.16,
        stroke_color=color, stroke_width=1.6,
    ).move_to(pos)
    txt = Text(label, font_size=font_size, color=color).move_to(pos)
    return VGroup(rect, txt)


def _arrow(start, end, color=C_DIM, sw=1.4):
    return Arrow(start, end, buff=0.06, color=color,
                 stroke_width=sw, max_tip_length_to_length_ratio=0.22)


class Transformer(Scene):

    def construct(self):
        self.camera.background_color = C_BG
        self._phase1()
        self._phase2()
        self._phase3()

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 1 — Positional encoding
    # ══════════════════════════════════════════════════════════════════════════

    def _phase1(self):
        title = (
            Text("Transformer Architecture", font_size=22, color=C_WHITE, weight=BOLD)
            .to_edge(UP, buff=0.28)
        )
        sub = (
            Text(
                "Positional encoding — inject word order into "
                "position-agnostic embeddings",
                font_size=13, color=C_DIM,
            )
            .next_to(title, DOWN, buff=0.12)
        )
        self.play(FadeIn(title), FadeIn(sub), run_time=0.5)

        tok_xs  = np.linspace(-3.2, 3.2, N)
        Y_TOK   =  1.30
        Y_EMB   =  0.30
        Y_PE    = -0.60
        Y_SUM   = -1.55

        # ── Token boxes ───────────────────────────────────────────────────────
        tok_cols = [C_GOLD, C_BLUE, C_BLUE, C_BLUE]
        tok_boxes = [
            _block(WORDS[i], [tok_xs[i], Y_TOK, 0], tok_cols[i],
                   w=0.88, h=0.42, font_size=12)
            for i in range(N)
        ]
        self.play(
            LaggedStart(*[FadeIn(t) for t in tok_boxes], lag_ratio=0.15),
            run_time=0.50,
        )

        # ── Embedding vectors (small grids) ───────────────────────────────────
        CELL  = 0.16
        emb_grids = []
        for i in range(N):
            np.random.seed(i + 10)
            vals = np.random.randn(D_EMB)
            cols_row = VGroup()
            for d in range(D_EMB):
                v   = float(np.tanh(vals[d]))
                col = interpolate_color(ManimColor(C_RED), ManimColor(C_BLUE),
                                        (v + 1) / 2)
                sq  = Square(side_length=CELL,
                             fill_color=col, fill_opacity=0.80,
                             stroke_color=C_DIM, stroke_width=0.3
                             ).move_to([tok_xs[i] + (d - D_EMB / 2 + 0.5) * CELL,
                                        Y_EMB, 0])
                cols_row.add(sq)
            emb_grids.append(cols_row)

        arr_te = [_arrow([tok_xs[i], Y_TOK - 0.22, 0],
                         [tok_xs[i], Y_EMB + 0.14, 0], color=C_DIM)
                  for i in range(N)]
        self.play(
            LaggedStart(*[AnimationGroup(GrowArrow(arr_te[i]), FadeIn(emb_grids[i]))
                          for i in range(N)],
                        lag_ratio=0.15),
            run_time=0.65,
        )

        emb_lbl = Text("token embeddings  E", font_size=10, color=C_DIM
                       ).next_to(VGroup(*emb_grids), RIGHT, buff=0.30)
        self.play(FadeIn(emb_lbl), run_time=0.20)

        # ── Positional encoding grid ──────────────────────────────────────────
        pe_grids = []
        for i in range(N):
            cols_row = VGroup()
            for d in range(D_EMB):
                if d % 2 == 0:
                    v = float(np.sin(i / (10000 ** (d / D_EMB))))
                else:
                    v = float(np.cos(i / (10000 ** ((d - 1) / D_EMB))))
                col = interpolate_color(ManimColor(C_DEAD), ManimColor(C_AMBER),
                                        (v + 1) / 2)
                sq  = Square(side_length=CELL,
                             fill_color=col, fill_opacity=0.80,
                             stroke_color=C_DIM, stroke_width=0.3
                             ).move_to([tok_xs[i] + (d - D_EMB / 2 + 0.5) * CELL,
                                        Y_PE, 0])
                cols_row.add(sq)
            pe_grids.append(cols_row)

        pe_lbl = Text("positional encoding  PE", font_size=10, color=C_AMBER
                      ).next_to(VGroup(*pe_grids), RIGHT, buff=0.30)
        self.play(
            LaggedStart(*[FadeIn(g) for g in pe_grids], lag_ratio=0.12),
            FadeIn(pe_lbl),
            run_time=0.55,
        )

        # ── Plus signs between E and PE ───────────────────────────────────────
        plus_signs = [
            Text("+", font_size=18, color=C_DIM).move_to([tok_xs[i], (Y_EMB + Y_PE) / 2, 0])
            for i in range(N)
        ]
        self.play(
            LaggedStart(*[FadeIn(p) for p in plus_signs], lag_ratio=0.10),
            run_time=0.30,
        )

        # ── Summed vectors ────────────────────────────────────────────────────
        sum_grids = []
        for i in range(N):
            cols_row = VGroup()
            np.random.seed(i + 10)
            e_vals = np.random.randn(D_EMB)
            for d in range(D_EMB):
                ev = float(np.tanh(e_vals[d]))
                pv = float(np.sin(i / (10000 ** (d / D_EMB))) if d % 2 == 0
                           else np.cos(i / (10000 ** ((d - 1) / D_EMB))))
                v  = np.tanh(ev + pv)
                col = interpolate_color(ManimColor(C_PURP), ManimColor(C_GREEN),
                                        (v + 1) / 2)
                sq  = Square(side_length=CELL,
                             fill_color=col, fill_opacity=0.80,
                             stroke_color=C_DIM, stroke_width=0.3
                             ).move_to([tok_xs[i] + (d - D_EMB / 2 + 0.5) * CELL,
                                        Y_SUM, 0])
                cols_row.add(sq)
            sum_grids.append(cols_row)

        eq_signs = [
            Text("=", font_size=18, color=C_DIM).move_to([tok_xs[i], (Y_PE + Y_SUM) / 2, 0])
            for i in range(N)
        ]
        self.play(
            LaggedStart(*[FadeIn(e) for e in eq_signs], lag_ratio=0.10),
            run_time=0.25,
        )
        self.play(
            LaggedStart(*[FadeIn(g) for g in sum_grids], lag_ratio=0.12),
            run_time=0.45,
        )
        sum_lbl = Text("E + PE  (input to encoder)", font_size=10, color=C_DIM
                       ).next_to(VGroup(*sum_grids), RIGHT, buff=0.30)
        self.play(FadeIn(sum_lbl), run_time=0.20)

        caption = (
            Text(
                "sin/cos frequencies encode relative position  ·  "
                "allows the model to generalise to unseen sequence lengths",
                font_size=12, color=C_DIM,
            )
            .to_edge(DOWN, buff=0.32)
        )
        self.play(FadeIn(caption), run_time=0.30)
        self.wait(1.8)

        self._title  = title
        self._p1_all = VGroup(
            sub, *tok_boxes, *arr_te, *emb_grids, emb_lbl,
            *pe_grids, pe_lbl, *plus_signs, *eq_signs,
            *sum_grids, sum_lbl, caption,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 2 — One encoder layer
    # ══════════════════════════════════════════════════════════════════════════

    def _phase2(self):
        self.play(FadeOut(self._p1_all), run_time=0.40)

        sub = (
            Text(
                "Encoder layer — multi-head self-attention + "
                "feed-forward, each with residual & norm",
                font_size=13, color=C_DIM,
            )
            .next_to(self._title, DOWN, buff=0.12)
        )
        self.play(FadeIn(sub), run_time=0.28)

        CX     = 0.0
        BW     = 2.60
        BH     = 0.50
        VSTEP  = 0.80
        Y_TOP  = 1.60

        def _b(label, y, color, w=BW, h=BH):
            return _block(label, [CX, y, 0], color, w=w, h=h)

        def _up(y_from, y_to, color=C_DIM):
            # y_from is the upper block (larger y), y_to is below — arrow goes downward
            return _arrow([CX, y_from - BH / 2, 0],
                          [CX, y_to   + BH / 2, 0], color=color)

        # Blocks top → bottom
        y_input  = Y_TOP - 0 * VSTEP
        y_qkv    = Y_TOP - 1 * VSTEP
        y_head1  = Y_TOP - 2 * VSTEP
        y_head2  = Y_TOP - 2 * VSTEP
        y_concat = Y_TOP - 3 * VSTEP
        y_add1   = Y_TOP - 4 * VSTEP
        y_ff     = Y_TOP - 5 * VSTEP
        y_add2   = Y_TOP - 6 * VSTEP

        # Two heads side by side
        H1_CX = CX - 1.45
        H2_CX = CX + 1.45
        HEAD_W = 1.10

        b_input  = _b("Input  (E + PE)",         y_input,  C_DIM,   w=BW)
        b_qkv    = _b("QKV Projection  (3×linear)", y_qkv,   C_BLUE,  w=BW)
        b_h1     = _block("Head 1",  [H1_CX, y_head1, 0], C_TEAL, w=HEAD_W, h=BH)
        b_h2     = _block("Head 2",  [H2_CX, y_head2, 0], C_PURP, w=HEAD_W, h=BH)
        b_concat = _b("Concat + Linear",          y_concat, C_BLUE,  w=BW)
        b_add1   = _b("Add & Norm",               y_add1,   C_TEAL,  w=BW)
        b_ff     = _block("Feed-Forward  (2×linear + ReLU)", [CX, y_ff, 0], C_AMBER, w=BW, h=BH, font_size=10)
        b_add2   = _b("Add & Norm",               y_add2,   C_TEAL,  w=BW)

        # Arrows (main spine)
        a_in_qkv    = _up(y_input,  y_qkv,    C_DIM)
        a_qkv_h1    = _arrow([CX, y_qkv - BH / 2, 0],
                             [H1_CX, y_head1 + BH / 2, 0], color=C_TEAL)
        a_qkv_h2    = _arrow([CX, y_qkv - BH / 2, 0],
                             [H2_CX, y_head2 + BH / 2, 0], color=C_PURP)
        a_h1_cat    = _arrow([H1_CX, y_head1 - BH / 2, 0],
                             [CX, y_concat + BH / 2, 0], color=C_TEAL)
        a_h2_cat    = _arrow([H2_CX, y_head2 - BH / 2, 0],
                             [CX, y_concat + BH / 2, 0], color=C_PURP)
        a_cat_add1  = _up(y_concat, y_add1,   C_DIM)
        a_add1_ff   = _up(y_add1,   y_ff,     C_DIM)
        a_ff_add2   = _up(y_ff,     y_add2,   C_DIM)

        # Residual skip connections — elbow: right → down → left-arrow
        RES_X = CX + BW / 2 + 0.85
        def _elbow(y_from, y_to):
            right_x = CX + BW / 2
            return VGroup(
                Line([right_x, y_from, 0], [RES_X, y_from, 0],
                     stroke_color=C_GOLD, stroke_width=1.4),
                Line([RES_X, y_from, 0], [RES_X, y_to, 0],
                     stroke_color=C_GOLD, stroke_width=1.4),
                Arrow([RES_X, y_to, 0], [right_x + 0.06, y_to, 0],
                      buff=0.0, color=C_GOLD,
                      stroke_width=1.4, max_tip_length_to_length_ratio=0.22),
            )
        res1 = _elbow(y_input, y_add1)
        res2 = _elbow(y_add1,  y_add2)
        res1_lbl = Text("residual", font_size=9, color=C_GOLD).next_to(res1, RIGHT, buff=0.06)
        res2_lbl = Text("residual", font_size=9, color=C_GOLD).next_to(res2, RIGHT, buff=0.06)

        # Animate layer by layer
        blocks_and_arrows = [
            (b_input,),
            (a_in_qkv, b_qkv),
            (a_qkv_h1, b_h1, a_qkv_h2, b_h2),
            (a_h1_cat, a_h2_cat, b_concat),
            (a_cat_add1, b_add1),
            (res1, res1_lbl),
            (a_add1_ff, b_ff),
            (a_ff_add2, b_add2),
            (res2, res2_lbl),
        ]
        for group in blocks_and_arrows:
            anims = []
            for m in group:
                if isinstance(m, Arrow):
                    anims.append(GrowArrow(m))
                elif m in (res1, res2):
                    anims.append(Create(m))
                else:
                    anims.append(FadeIn(m))
            self.play(*anims, run_time=0.32)

        caption = (
            Text(
                "multi-head attention lets the model jointly attend to "
                "different representation subspaces",
                font_size=12, color=C_DIM,
            )
            .to_edge(DOWN, buff=0.32)
        )
        self.play(FadeIn(caption), run_time=0.30)
        self.wait(1.8)

        self._p2_all = VGroup(
            sub, b_input, b_qkv, b_h1, b_h2, b_concat,
            b_add1, b_ff, b_add2,
            a_in_qkv, a_qkv_h1, a_qkv_h2,
            a_h1_cat, a_h2_cat, a_cat_add1, a_add1_ff, a_ff_add2,
            res1, res2, res1_lbl, res2_lbl, caption,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 3 — Encoder stack → classification head
    # ══════════════════════════════════════════════════════════════════════════

    def _phase3(self):
        self.play(FadeOut(self._p2_all), run_time=0.40)

        sub = (
            Text(
                "Stacked encoders + classification head — "
                "[CLS] token aggregates the full sequence",
                font_size=13, color=C_DIM,
            )
            .next_to(self._title, DOWN, buff=0.12)
        )
        self.play(FadeIn(sub), run_time=0.28)

        CX     = 0.0
        BW     = 3.20
        BH     = 0.50
        VSTEP  = 0.65
        Y_TOP  = 2.20   # top of stack, close to subtitle to fill blank space

        def _b(label, y, color, w=BW):
            return _block(label, [CX, y, 0], color, w=w, h=BH)

        def _dn(y_from, y_to, color=C_DIM):
            # arrow from bottom of upper block down to top of lower block
            return _arrow([CX, y_from - BH / 2, 0],
                          [CX, y_to   + BH / 2, 0], color=color)

        layers = [
            ("Input tokens  +  PE",      C_DIM,   0),
            ("Encoder layer 1",           C_BLUE,  1),
            ("Encoder layer 2",           C_BLUE,  2),
            ("Encoder layer 3",           C_BLUE,  3),
            ("[CLS] representation",      C_GOLD,  4),
            ("Linear classifier",         C_AMBER, 5),
            ("Softmax",                   C_GREEN, 6),
        ]

        blocks = []
        arrows = []
        for label, color, idx in layers:
            y = Y_TOP - idx * VSTEP
            blocks.append(_b(label, y, color))
            if idx > 0:
                arrows.append(_dn(Y_TOP - (idx - 1) * VSTEP,
                                  Y_TOP - idx * VSTEP, color))

        # Animate top → bottom
        self.play(FadeIn(blocks[0]), run_time=0.28)
        for i in range(1, len(blocks)):
            self.play(
                GrowArrow(arrows[i - 1]),
                FadeIn(blocks[i]),
                run_time=0.32,
            )

        # Output tokens below Softmax
        y_soft = Y_TOP - 6 * VSTEP
        pos_tok = _block("POSITIVE", [CX - 1.30, y_soft - VSTEP * 0.90, 0],
                         C_GREEN, w=1.20, h=0.42, font_size=12)
        neg_tok = _block("negative", [CX + 1.30, y_soft - VSTEP * 0.90, 0],
                         C_DEAD,  w=1.20, h=0.42, font_size=12)
        a_pos = _arrow([CX, y_soft - BH / 2, 0],
                       [CX - 0.80, y_soft - VSTEP * 0.90 + 0.22, 0], color=C_GREEN)
        a_neg = _arrow([CX, y_soft - BH / 2, 0],
                       [CX + 0.80, y_soft - VSTEP * 0.90 + 0.22, 0], color=C_DEAD)

        self.play(
            GrowArrow(a_pos), GrowArrow(a_neg),
            FadeIn(pos_tok), FadeIn(neg_tok),
            run_time=0.45,
        )

        # Brace on encoder layers 1-3
        enc_group = VGroup(blocks[1], blocks[2], blocks[3])
        brace = Brace(enc_group, direction=RIGHT, color=C_BLUE, buff=0.12)
        brace_lbl = Text("N = 3\nlayers", font_size=11, color=C_BLUE
                         ).next_to(brace, RIGHT, buff=0.10)
        self.play(GrowFromCenter(brace), FadeIn(brace_lbl), run_time=0.35)

        caption = (
            Text(
                "[CLS] token attends to the whole sequence  ·  "
                "fine-tune the head for any classification task",
                font_size=12, color=C_DIM,
            )
            .to_edge(DOWN, buff=0.32)
        )
        self.play(FadeIn(caption), run_time=0.30)
        self.wait(2.5)
