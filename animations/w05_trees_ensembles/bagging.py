"""
W05 — Bagging (Bootstrap Aggregating)

Three-phase animation for Lecture 2 (Random Forests).

  Phase 1: Bootstrap sampling — 10-sample dataset; B=3 bootstrap samples
           drawn with replacement; duplicate-index cells highlighted in gold.
  Phase 2: One schematic decision tree per bootstrap sample; different root
           splits illustrate tree diversity.
  Phase 3: Majority vote — each tree predicts a class; tally → ensemble
           result; variance-reduction caption.

Render:
  ../../env/bin/manim -pql bagging.py Bagging
  ../../env/bin/manim -pqh bagging.py Bagging
"""

from manim import *
import numpy as np
from collections import Counter

# ── Palette ───────────────────────────────────────────────────────────────────
C_BG    = "#0d1117"
C_C0    = "#ffa657"   # amber  – class 0
C_C1    = "#2dd4bf"   # teal   – class 1
C_WHITE = "#ffffff"
C_DIM   = "#8b949e"
C_GOLD  = "#e3b341"
C_BLUE  = "#58a6ff"
TREE_COLS = ["#ffa657", "#bc8cff", "#3fb950"]  # accent per tree

# ── Constants ─────────────────────────────────────────────────────────────────
N = 10
BLOCK_W, BLOCK_H, GAP = 0.38, 0.38, 0.055
SAMPLE_COLORS = [C_C0] * 5 + [C_C1] * 5   # 1-5 amber, 6-10 teal

# Pre-defined bootstrap draws (0-indexed) — chosen for visual clarity
BOOTS = [
    [0, 2, 2, 3, 5, 6, 6, 8, 9, 9],
    [0, 1, 3, 4, 4, 5, 7, 8, 9, 9],
    [1, 1, 2, 3, 5, 6, 7, 7, 8, 9],
]
SPLIT_LABELS = ["x1 < 0.42", "x1 < 0.55", "x1 < 0.38"]
VOTES        = [1, 0, 1]    # tree predictions: C1, C0, C1 → majority C1


# ── Helpers ───────────────────────────────────────────────────────────────────

def _block_row(indices, stroke_col, y_center):
    """Return a VGroup of (rect, num_text) cells arranged in a row."""
    cells = VGroup()
    for idx in indices:
        rect = Rectangle(
            width=BLOCK_W, height=BLOCK_H,
            fill_color=SAMPLE_COLORS[idx], fill_opacity=0.80,
            stroke_color=stroke_col, stroke_width=1.4,
        )
        num = Text(str(idx + 1), font_size=10, color=C_BG)
        cell = VGroup(rect, num)
        cells.add(cell)
    cells.arrange(RIGHT, buff=GAP)
    cells.move_to([0.8, y_center, 0])
    for cell in cells:
        cell[1].move_to(cell[0].get_center())
    return cells


def _node_box(label, col, w=1.25, h=0.42):
    box = RoundedRectangle(
        corner_radius=0.07, width=w, height=h,
        fill_color=col, fill_opacity=0.22,
        stroke_color=col, stroke_width=1.6,
    )
    txt = Text(label, font_size=11, color=C_WHITE)
    return VGroup(box, txt)


# ── Scene ─────────────────────────────────────────────────────────────────────

class Bagging(Scene):
    def construct(self):
        self.camera.background_color = C_BG

        # ════════════════════════════════════════════════════════════════════
        #  Phase 1 — Bootstrap sampling
        # ════════════════════════════════════════════════════════════════════
        title = (
            Text("Bagging — Bootstrap Aggregating",
                 font_size=18, color=C_WHITE, weight=BOLD)
            .to_edge(UP, buff=0.35)
        )
        self.play(FadeIn(title), run_time=0.4)

        # Original dataset row
        orig_cells = _block_row(list(range(N)), C_DIM, y_center=2.15)
        orig_hdr = (
            Text("Original  (N=10)", font_size=12, color=C_DIM)
            .next_to(orig_cells, LEFT, buff=0.30)
        )
        self.play(FadeIn(orig_hdr), run_time=0.3)
        self.play(
            LaggedStart(*[GrowFromCenter(c[0]) for c in orig_cells], lag_ratio=0.05),
            run_time=0.8,
        )
        self.play(FadeIn(VGroup(*[c[1] for c in orig_cells])), run_time=0.2)
        self.wait(0.2)

        # Three bootstrap rows
        ROW_Y = [0.95, 0.0, -0.95]
        boot_hdrs, boot_rows = [], []
        for b, (indices, ry) in enumerate(zip(BOOTS, ROW_Y)):
            hdr = (
                Text(f"Bootstrap {b+1}", font_size=12, color=TREE_COLS[b])
                .move_to([-5.2, ry, 0])
            )
            row = _block_row(indices, TREE_COLS[b], y_center=ry)
            boot_hdrs.append(hdr)
            boot_rows.append(row)
            self.play(
                FadeIn(hdr),
                LaggedStart(*[GrowFromCenter(c[0]) for c in row], lag_ratio=0.04),
                run_time=0.6,
            )
            self.play(FadeIn(VGroup(*[c[1] for c in row])), run_time=0.15)
            self.wait(0.15)

        # Highlight duplicate indices in gold
        dup_anims = []
        for b, indices in enumerate(BOOTS):
            cnt = Counter(indices)
            for pos, idx in enumerate(indices):
                if cnt[idx] > 1:
                    dup_anims.append(
                        boot_rows[b][pos][0].animate.set_stroke(C_GOLD, width=2.4)
                    )
        self.play(*dup_anims, run_time=0.5)

        cap1 = (
            Text("sampling with replacement — repeated indices highlighted in gold",
                 font_size=11, color=C_DIM)
            .to_edge(DOWN, buff=0.40)
        )
        self.play(FadeIn(cap1), run_time=0.4)
        self.wait(1.2)

        # ════════════════════════════════════════════════════════════════════
        #  Phase 2 — One tree per bootstrap
        # ════════════════════════════════════════════════════════════════════
        title2 = (
            Text("Each bootstrap sample trains an independent tree",
                 font_size=18, color=C_WHITE, weight=BOLD)
            .to_edge(UP, buff=0.35)
        )
        all_p1 = VGroup(orig_hdr, orig_cells, *boot_hdrs, *boot_rows, cap1)
        self.play(
            FadeOut(all_p1),
            ReplacementTransform(title, title2),
            run_time=0.6,
        )

        TREE_X      = [-3.8, 0.0, 3.8]
        ROOT_Y      =  0.6
        LEAF_Y      = -0.8
        LEAF_OFFSET =  0.85

        tree_groups = []
        for t in range(3):
            col = TREE_COLS[t]
            tx  = TREE_X[t]

            t_lbl = (
                Text(f"Tree {t+1}", font_size=13, color=col, weight=BOLD)
                .move_to([tx, 1.55, 0])
            )
            root  = _node_box(SPLIT_LABELS[t], col)
            root.move_to([tx, ROOT_Y, 0])

            leaf0 = _node_box("Class 0", C_C0, w=0.95, h=0.38)
            leaf0.move_to([tx - LEAF_OFFSET, LEAF_Y, 0])

            leaf1 = _node_box("Class 1", C_C1, w=0.95, h=0.38)
            leaf1.move_to([tx + LEAF_OFFSET, LEAF_Y, 0])

            e0 = Line(root[0].get_bottom(), leaf0[0].get_top(),
                      color=C_DIM, stroke_width=1.4)
            e1 = Line(root[0].get_bottom(), leaf1[0].get_top(),
                      color=C_DIM, stroke_width=1.4)
            e0l = Text("yes", font_size=9, color=C_DIM).next_to(e0.get_center(), LEFT, buff=0.08)
            e1l = Text("no",  font_size=9, color=C_DIM).next_to(e1.get_center(), RIGHT, buff=0.08)

            # Small "bootstrap" badge below tree label
            badge = (
                Text(f"(Bootstrap {t+1})", font_size=10, color=col)
                .move_to([tx, 1.10, 0])
            )

            tg = VGroup(t_lbl, badge, root, leaf0, leaf1, e0, e1, e0l, e1l)
            tree_groups.append(tg)

            self.play(
                FadeIn(t_lbl), FadeIn(badge),
                Create(e0), Create(e1),
                FadeIn(root), FadeIn(leaf0), FadeIn(leaf1),
                FadeIn(e0l), FadeIn(e1l),
                run_time=0.65,
            )
            self.wait(0.25)

        div_cap = (
            Text(
                "different training sets  →  different splits  →  diverse trees",
                font_size=12, color=C_DIM,
            )
            .to_edge(DOWN, buff=0.45)
        )
        self.play(FadeIn(div_cap), run_time=0.4)
        self.wait(1.3)

        # ════════════════════════════════════════════════════════════════════
        #  Phase 3 — Majority vote
        # ════════════════════════════════════════════════════════════════════
        title3 = (
            Text("Majority vote — ensemble prediction",
                 font_size=18, color=C_WHITE, weight=BOLD)
            .to_edge(UP, buff=0.35)
        )
        self.play(
            FadeOut(VGroup(*tree_groups, div_cap)),
            ReplacementTransform(title2, title3),
            run_time=0.6,
        )

        # Vote rows
        VOTE_X   = 0.0
        VOTE_Y0  = 1.5
        VOTE_DY  = 0.85
        VOTE_LBL = ["Class 1", "Class 0", "Class 1"]
        VOTE_COL = [C_C1,      C_C0,      C_C1     ]

        vote_rows = []
        for t in range(3):
            tree_txt = Text(f"Tree {t+1}:", font_size=15, color=TREE_COLS[t], weight=BOLD)
            arrow    = Text("→", font_size=15, color=C_DIM)
            pred_txt = Text(VOTE_LBL[t], font_size=15, color=VOTE_COL[t], weight=BOLD)
            row = VGroup(tree_txt, arrow, pred_txt).arrange(RIGHT, buff=0.40)
            row.move_to([VOTE_X, VOTE_Y0 - t * VOTE_DY, 0])
            vote_rows.append(row)
            self.play(FadeIn(row, shift=RIGHT * 0.12), run_time=0.40)
            self.wait(0.25)

        # Divider + tally
        tally_y = VOTE_Y0 - 3 * VOTE_DY + 0.12
        divider = Line(
            [VOTE_X - 2.4, tally_y, 0],
            [VOTE_X + 2.4, tally_y, 0],
            color=C_DIM, stroke_width=0.9,
        )
        tally = (
            Text("Tally:  Class 0  \u00d7 1     Class 1  \u00d7 2",
                 font_size=13, color=C_DIM)
            .move_to([VOTE_X, tally_y - 0.38, 0])
        )
        self.play(Create(divider), FadeIn(tally), run_time=0.5)
        self.wait(0.4)

        # Ensemble result
        result = (
            Text("Ensemble  \u2192  Class 1", font_size=22, color=C_C1, weight=BOLD)
            .move_to([VOTE_X, tally_y - 1.15, 0])
        )
        self.play(FadeIn(result, scale=1.08), run_time=0.5)
        self.wait(0.3)

        # Variance-reduction caption
        var_cap = (
            Text(
                "B independent trees  \u2192  variance \u2248 \u03c3\u00b2/B    |    bias unchanged",
                font_size=12, color=C_DIM,
            )
            .to_edge(DOWN, buff=0.40)
        )
        self.play(FadeIn(var_cap), run_time=0.5)
        self.wait(2.5)
