"""
W10 — Attention Mechanism

  Phase 1: Seq2seq bottleneck.
           Encoder compresses a sentence into one fixed vector;
           decoder struggles — information bottleneck visualised.

  Phase 2: Attention mechanism.
           Decoder query attends to all encoder keys; dot-product
           scores → softmax weights (bar chart) → weighted sum of
           values. Source tokens highlighted by attention weight.

  Phase 3: Self-attention score matrix.
           Each token attends to every other token in the same
           sentence; score grid fills in, then output shown as
           a weighted blend row.

Render:
  ../../env/bin/manim -pql attention.py Attention
  ../../env/bin/manim -pqh attention.py Attention
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

SRC_WORDS = ["The", "cat", "sat", "on", "mat"]
TGT_WORDS = ["Le",  "chat", "s'assit"]


def _tok_box(word, pos, color, font_size=12, w=0.78, h=0.42):
    rect = RoundedRectangle(
        width=w, height=h, corner_radius=0.07,
        fill_color=color, fill_opacity=0.16,
        stroke_color=color, stroke_width=1.6,
    ).move_to(pos)
    lbl = Text(word, font_size=font_size, color=color).move_to(pos)
    return VGroup(rect, lbl)


def _arrow(start, end, color=C_DIM, sw=1.4):
    return Arrow(start, end, buff=0.08, color=color,
                 stroke_width=sw, max_tip_length_to_length_ratio=0.22)


class Attention(Scene):

    def construct(self):
        self.camera.background_color = C_BG
        self._phase1()
        self._phase2()
        self._phase3()

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 1 — Seq2seq bottleneck
    # ══════════════════════════════════════════════════════════════════════════

    def _phase1(self):
        title = (
            Text("Attention Mechanism", font_size=22, color=C_WHITE, weight=BOLD)
            .to_edge(UP, buff=0.28)
        )
        sub = (
            Text(
                "Seq2seq without attention — entire input compressed "
                "into one fixed-size vector",
                font_size=13, color=C_DIM,
            )
            .next_to(title, DOWN, buff=0.12)
        )
        self.play(FadeIn(title), FadeIn(sub), run_time=0.5)

        # ── Layout: source tokens stacked vertically on left so arrows fan in
        # to the encoder without crossing each other; same for decoder→target.
        N   = len(SRC_WORDS)
        M   = len(TGT_WORDS)
        CY  = 0.10   # vertical centre of the whole diagram

        src_x   = -5.4
        enc_x   = -2.9
        ctx_x   = -1.1
        dec_x   =  0.8
        tgt_x   =  3.4

        src_step = 0.54
        src_ys   = [CY + src_step * (N / 2 - 0.5 - i) for i in range(N)]

        tgt_step = 0.60
        tgt_ys   = [CY + tgt_step * (M / 2 - 0.5 - i) for i in range(M)]

        # Source tokens (vertical stack)
        src_toks = [_tok_box(SRC_WORDS[i], [src_x, src_ys[i], 0], C_BLUE)
                    for i in range(N)]
        self.play(
            LaggedStart(*[FadeIn(t) for t in src_toks], lag_ratio=0.15),
            run_time=0.55,
        )

        # Encoder box — arrows fan in from different y → same enc_y
        enc_box  = _tok_box("Encoder", [enc_x, CY, 0], C_TEAL, w=1.10, h=0.50)
        enc_arrs = [_arrow([src_x + 0.39, src_ys[i], 0],
                           [enc_x - 0.55,  CY,        0], color=C_DIM)
                    for i in range(N)]
        self.play(FadeIn(enc_box), run_time=0.25)
        self.play(
            LaggedStart(*[GrowArrow(a) for a in enc_arrs], lag_ratio=0.10),
            run_time=0.55,
        )

        # Context vector (narrow tall box — the bottleneck)
        ctx_box   = RoundedRectangle(
            width=0.48, height=1.10, corner_radius=0.07,
            fill_color=C_GOLD, fill_opacity=0.22,
            stroke_color=C_GOLD, stroke_width=1.8,
        ).move_to([ctx_x, CY, 0])
        ctx_lbl   = Text("c", font_size=14, color=C_GOLD, weight=BOLD).move_to([ctx_x, CY, 0])
        ctx_arrow = _arrow([enc_x + 0.55, CY, 0], [ctx_x - 0.25, CY, 0], color=C_GOLD)
        self.play(GrowArrow(ctx_arrow), FadeIn(ctx_box), FadeIn(ctx_lbl), run_time=0.40)

        # Bottleneck annotation
        btn_brace = Brace(ctx_box, direction=DOWN, color=C_RED, buff=0.12)
        btn_lbl   = Text("information\nbottleneck", font_size=10, color=C_RED
                         ).next_to(btn_brace, DOWN, buff=0.08)
        self.play(GrowFromCenter(btn_brace), FadeIn(btn_lbl), run_time=0.35)

        # Decoder
        dec_box   = _tok_box("Decoder", [dec_x, CY, 0], C_PURP, w=1.10, h=0.50)
        dec_arrow = _arrow([ctx_x + 0.25, CY, 0], [dec_x - 0.55, CY, 0], color=C_GOLD)
        self.play(GrowArrow(dec_arrow), FadeIn(dec_box), run_time=0.30)

        # Target tokens (vertical stack) — arrows fan out from same dec_y → different y
        tgt_toks = [_tok_box(TGT_WORDS[i], [tgt_x, tgt_ys[i], 0], C_AMBER)
                    for i in range(M)]
        tgt_arrs = [_arrow([dec_x + 0.55, CY,        0],
                           [tgt_x  - 0.39, tgt_ys[i], 0], color=C_DIM)
                    for i in range(M)]
        self.play(
            LaggedStart(*[AnimationGroup(GrowArrow(tgt_arrs[i]), FadeIn(tgt_toks[i]))
                          for i in range(M)],
                        lag_ratio=0.25),
            run_time=0.65,
        )

        caption = (
            Text(
                "for long sentences, a single vector cannot retain all information",
                font_size=12, color=C_DIM,
            )
            .to_edge(DOWN, buff=0.32)
        )
        self.play(FadeIn(caption), run_time=0.30)
        self.wait(1.6)

        self._title  = title
        self._p1_all = VGroup(
            sub, *src_toks, enc_box, *enc_arrs,
            ctx_box, ctx_lbl, ctx_arrow, btn_brace, btn_lbl,
            dec_box, dec_arrow, *tgt_toks, *tgt_arrs, caption,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 2 — Attention weights
    # ══════════════════════════════════════════════════════════════════════════

    def _phase2(self):
        self.play(FadeOut(self._p1_all), run_time=0.40)

        sub = (
            Text(
                "With attention — decoder query looks at all encoder "
                "states and weighs them dynamically",
                font_size=13, color=C_DIM,
            )
            .next_to(self._title, DOWN, buff=0.12)
        )
        self.play(FadeIn(sub), run_time=0.28)

        N = len(SRC_WORDS)
        SRC_Y  =  0.90
        BAR_Y0 = -0.10
        DEC_Y  = -1.50

        src_xs = np.linspace(-4.2, 4.2, N)

        # ── Source token boxes ────────────────────────────────────────────────
        src_toks = [_tok_box(SRC_WORDS[i], [src_xs[i], SRC_Y, 0], C_BLUE)
                    for i in range(N)]
        self.play(
            LaggedStart(*[FadeIn(t) for t in src_toks], lag_ratio=0.12),
            run_time=0.45,
        )

        # ── Decoder query token ───────────────────────────────────────────────
        query_tok = _tok_box("chat  (query)", [0.0, DEC_Y, 0], C_PURP,
                             w=1.50, h=0.46)
        self.play(FadeIn(query_tok), run_time=0.28)

        # ── Attention scores (hand-crafted for "chat" → "cat") ────────────────
        # Higher score for "cat" (index 1)
        raw_scores = np.array([0.4, 2.8, 0.5, 0.3, 0.2])
        exp_s = np.exp(raw_scores - raw_scores.max())
        attn  = exp_s / exp_s.sum()   # softmax

        # Dashed lines: query → each source token (score lines)
        score_lines = []
        for i in range(N):
            line = DashedLine(
                [0.0, DEC_Y + 0.24, 0],
                [src_xs[i], SRC_Y - 0.22, 0],
                dash_length=0.10,
                stroke_color=C_DIM, stroke_width=0.9,
            )
            score_lines.append(line)

        self.play(
            LaggedStart(*[Create(l) for l in score_lines], lag_ratio=0.10),
            run_time=0.55,
        )

        # ── Attention weight bars ─────────────────────────────────────────────
        BAR_W   = 0.34
        MAX_H   = 1.00
        BAR_Y0  = -0.40   # lowered so tallest bar clears the source token boxes
        bars    = []
        bar_lbls = []
        for i in range(N):
            bh  = max(attn[i] * MAX_H, 0.03)
            col = interpolate_color(ManimColor(C_DEAD), ManimColor(C_TEAL), attn[i] * 2.5)
            bar = Rectangle(
                width=BAR_W, height=bh,
                fill_color=col, fill_opacity=0.80,
                stroke_color=col, stroke_width=0.8,
            ).move_to([src_xs[i], BAR_Y0 + bh / 2, 0])
            wlbl = Text(f"{attn[i]:.2f}", font_size=9, color=col).next_to(bar, UP, buff=0.06)
            bars.append(bar)
            bar_lbls.append(wlbl)

        self.play(
            LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.10),
            run_time=0.65,
        )
        self.play(
            LaggedStart(*[FadeIn(l) for l in bar_lbls], lag_ratio=0.10),
            run_time=0.35,
        )

        # ── Highlight source tokens by attention weight ───────────────────────
        highlights = []
        for i in range(N):
            alpha = float(attn[i])
            col   = interpolate_color(ManimColor(C_DEAD), ManimColor(C_GOLD), min(alpha * 3.5, 1.0))
            hl = src_toks[i][0].copy().set_fill(col, opacity=alpha * 1.2).set_stroke(col)
            highlights.append(hl)

        self.play(
            *[Transform(src_toks[i][0], highlights[i]) for i in range(N)],
            run_time=0.55,
        )

        attn_lbl = Text("attention weights  (softmax of Q·K scores)",
                        font_size=11, color=C_TEAL).move_to([0.0, BAR_Y0 - 0.60, 0])
        self.play(FadeIn(attn_lbl), run_time=0.25)

        caption = (
            Text(
                "output = weighted sum of encoder values  ·  "
                "different decoder steps attend to different source positions",
                font_size=12, color=C_DIM,
            )
            .to_edge(DOWN, buff=0.32)
        )
        self.play(FadeIn(caption), run_time=0.30)
        self.wait(1.8)

        self._p2_all = VGroup(
            sub, *src_toks, query_tok, *score_lines,
            *bars, *bar_lbls, attn_lbl, caption,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 3 — Self-attention score matrix
    # ══════════════════════════════════════════════════════════════════════════

    def _phase3(self):
        self.play(FadeOut(self._p2_all), run_time=0.40)

        sub = (
            Text(
                "Self-attention — every token attends to every other "
                "token in the same sequence",
                font_size=13, color=C_DIM,
            )
            .next_to(self._title, DOWN, buff=0.12)
        )
        self.play(FadeIn(sub), run_time=0.28)

        WORDS = ["The", "cat", "sat", "on", "mat"]
        N     = len(WORDS)

        # Simulated self-attention scores (row = query, col = key)
        np.random.seed(7)
        raw = np.array([
            [2.0, 0.3, 0.2, 0.1, 0.2],
            [0.4, 2.5, 0.8, 0.3, 0.5],
            [0.2, 1.2, 2.1, 0.5, 0.4],
            [0.1, 0.4, 0.6, 2.3, 0.3],
            [0.3, 0.6, 0.4, 0.2, 2.2],
        ])
        # softmax per row
        def softmax_row(r):
            e = np.exp(r - r.max())
            return e / e.sum()
        attn_mat = np.array([softmax_row(raw[i]) for i in range(N)])

        CELL  = 0.72
        OX    = -2.10   # matrix origin x (left edge of col labels)
        OY    =  0.95   # matrix origin y (top edge of row labels)

        # Column labels (key tokens) — top
        col_lbls = []
        for j, w in enumerate(WORDS):
            lbl = Text(w, font_size=10, color=C_BLUE).move_to(
                [OX + (j + 0.5) * CELL, OY + 0.30, 0]
            )
            col_lbls.append(lbl)

        # Row labels (query tokens) — left
        row_lbls = []
        for i, w in enumerate(WORDS):
            lbl = Text(w, font_size=10, color=C_PURP).move_to(
                [OX - 0.38, OY - (i + 0.5) * CELL, 0]
            )
            row_lbls.append(lbl)

        self.play(
            LaggedStart(*[FadeIn(l) for l in col_lbls + row_lbls], lag_ratio=0.06),
            run_time=0.50,
        )

        # Grid cells — fill by row with LaggedStart
        cells = []
        for i in range(N):
            row_cells = []
            for j in range(N):
                v   = attn_mat[i, j]
                col = interpolate_color(ManimColor(C_DEAD), ManimColor(C_TEAL), v * 3.5)
                cx  = OX + (j + 0.5) * CELL
                cy  = OY - (i + 0.5) * CELL
                sq  = Square(
                    side_length=CELL - 0.04,
                    fill_color=col, fill_opacity=min(v * 3.5, 0.85),
                    stroke_color=C_DIM, stroke_width=0.5,
                ).move_to([cx, cy, 0])
                val_lbl = Text(f"{v:.2f}", font_size=7, color=C_WHITE).move_to([cx, cy, 0])
                row_cells.append(VGroup(sq, val_lbl))
            cells.append(row_cells)

        for i in range(N):
            self.play(
                LaggedStart(*[FadeIn(cells[i][j]) for j in range(N)], lag_ratio=0.08),
                run_time=0.35,
            )

        # ── Output row highlight (row for "cat") ──────────────────────────────
        highlight_row = 1  # "cat"
        row_hl = SurroundingRectangle(
            VGroup(*[cells[highlight_row][j] for j in range(N)]),
            color=C_GOLD, stroke_width=1.8, buff=0.04,
        )
        hl_note = Text(
            "\"cat\" attends most to itself, then \"sat\"",
            font_size=11, color=C_GOLD,
        ).next_to(row_hl, RIGHT, buff=0.22)

        self.play(Create(row_hl), FadeIn(hl_note), run_time=0.40)

        caption = (
            Text(
                "each output token = weighted blend of all input tokens  ·  "
                "enables capturing long-range dependencies in one step",
                font_size=12, color=C_DIM,
            )
            .to_edge(DOWN, buff=0.32)
        )
        self.play(FadeIn(caption), run_time=0.30)
        self.wait(2.2)
