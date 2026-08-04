"""
W04 — K-Fold Cross-Validation

Three-phase animation for Lecture 2 (Cross-validation and hyperparameter tuning).

  Phase 1: 20 sample blocks appear, colour-coded by fold; fold labels annotate
           each group.
  Phase 2: Fold rotation (k = 1 → 5).  Each fold's blocks brighten as the
           validation set while the rest dim; a score appears below the blocks
           and a running score list accumulates on the right.
  Phase 3: A bar chart of the five fold scores appears with a dashed mean line,
           a ±std shaded band, and the final "CV Score" in large type.

Render:
  ../../env/bin/manim -pql cross_validation.py CrossValidation
  ../../env/bin/manim -pqh cross_validation.py CrossValidation
"""

from manim import *
import numpy as np

# ── Palette ───────────────────────────────────────────────────────────────────
C_BG   = "#0d1117"
C_DIM  = "#8b949e"
C_WHITE = "#ffffff"
FOLD_COLORS = ["#ffa657", "#2dd4bf", "#bc8cff", "#e3b341", "#3fb950"]

_AX = dict(
    axis_config={"color": "#21262d", "stroke_width": 1.2, "include_ticks": False},
    tips=False,
)

# ── Constants ─────────────────────────────────────────────────────────────────
K       = 5
N_FOLD  = 4
N       = K * N_FOLD     # 20 samples total
BLOCK_W = 0.32
BLOCK_H = 0.32
GAP     = 0.055
SCORES  = [0.84, 0.81, 0.87, 0.79, 0.83]
MEAN    = float(np.mean(SCORES))
STD     = float(np.std(SCORES))

# Bar-chart y-range
Y_LO, Y_HI = 0.74, 0.94


# ── Scene ─────────────────────────────────────────────────────────────────────

class CrossValidation(Scene):
    def construct(self):
        self.camera.background_color = C_BG

        # ── Build the 20 sample blocks ────────────────────────────────────
        blocks       = []    # flat list of all Rectangle objects
        fold_groups  = []    # VGroup per fold (4 blocks each)
        for k in range(K):
            fg = VGroup()
            for _ in range(N_FOLD):
                rect = Rectangle(
                    width=BLOCK_W, height=BLOCK_H,
                    fill_color=FOLD_COLORS[k], fill_opacity=0.82,
                    stroke_color=FOLD_COLORS[k], stroke_width=1.2,
                )
                blocks.append(rect)
                fg.add(rect)
            fold_groups.append(fg)

        all_blocks = VGroup(*blocks)
        all_blocks.arrange(RIGHT, buff=GAP)
        all_blocks.move_to([0, 1.8, 0])

        # Fold labels below each group
        fold_lbls = VGroup(*[
            Text(f"Fold {k+1}", font_size=11, color=FOLD_COLORS[k])
            .next_to(fold_groups[k], DOWN, buff=0.14)
            for k in range(K)
        ])

        # ════════════════════════════════════════════════════════════════════
        #  Phase 1 — Dataset split into K folds
        # ════════════════════════════════════════════════════════════════════
        title = (
            Text("5-Fold Cross-Validation",
                 font_size=18, color=C_WHITE, weight=BOLD)
            .to_edge(UP, buff=0.35)
        )
        n_lbl = (
            Text("20 samples  \u2192  4 per fold",
                 font_size=13, color=C_DIM)
            .next_to(title, DOWN, buff=0.12)
        )
        self.play(FadeIn(title), FadeIn(n_lbl), run_time=0.5)

        # Blocks appear fold by fold
        self.play(
            LaggedStart(
                *[GrowFromCenter(b) for b in blocks],
                lag_ratio=0.04,
            ),
            run_time=1.2,
        )
        self.play(FadeIn(fold_lbls), run_time=0.5)
        self.wait(0.6)

        # ════════════════════════════════════════════════════════════════════
        #  Phase 2 — Fold rotation
        # ════════════════════════════════════════════════════════════════════
        # Right-side score list — fills in as folds complete
        score_col_x  = 5.35
        score_col_y0 = 1.15
        score_row_dy = 0.48
        score_entries = []   # accumulated Text objects

        for k in range(K):
            # ── Dim/brighten blocks ──────────────────────────────────────
            block_anims = []
            for i, block in enumerate(blocks):
                fi = i // N_FOLD
                if fi == k:
                    block_anims.append(
                        block.animate
                        .set_fill(FOLD_COLORS[k], opacity=1.0)
                        .set_stroke(FOLD_COLORS[k], opacity=1.0)
                    )
                else:
                    block_anims.append(
                        block.animate
                        .set_fill(FOLD_COLORS[fi], opacity=0.15)
                        .set_stroke(FOLD_COLORS[fi], opacity=0.15)
                    )

            # Highlight validation fold label, dim others
            lbl_anims = []
            for j, lbl in enumerate(fold_lbls):
                if j == k:
                    lbl_anims.append(lbl.animate.set_opacity(1.0))
                else:
                    lbl_anims.append(lbl.animate.set_opacity(0.25))

            # "VALIDATION" tag below current fold
            val_tag = (
                Text("VAL", font_size=10, color=FOLD_COLORS[k], weight=BOLD)
                .next_to(fold_groups[k], UP, buff=0.08)
            )
            self.play(
                *block_anims,
                *lbl_anims,
                FadeIn(val_tag),
                run_time=0.45,
            )

            # Score appears centred below the blocks
            score_ctr = (
                Text(f"Score: {SCORES[k]:.2f}",
                     font_size=16, color=FOLD_COLORS[k], weight=BOLD)
                .move_to([0, 0.55, 0])
            )
            self.play(FadeIn(score_ctr, shift=UP * 0.1), run_time=0.35)
            self.wait(0.55)

            # Score migrates to the right-side accumulator list
            target_y = score_col_y0 - k * score_row_dy
            score_entry = (
                Text(f"Fold {k+1}:  {SCORES[k]:.2f}",
                     font_size=13, color=FOLD_COLORS[k])
                .move_to([score_col_x, target_y, 0])
            )
            self.play(
                FadeOut(val_tag),
                ReplacementTransform(score_ctr, score_entry),
                run_time=0.40,
            )
            score_entries.append(score_entry)

        self.wait(0.7)

        # ════════════════════════════════════════════════════════════════════
        #  Phase 3 — Score bar chart + CV summary
        # ════════════════════════════════════════════════════════════════════
        # Fade out fold rotation artefacts; blocks move up and shrink
        all_blocks_shrunk = all_blocks.copy().scale(0.6).to_edge(UP, buff=0.7)
        self.play(
            Transform(all_blocks, all_blocks_shrunk),
            FadeOut(fold_lbls),
            FadeOut(n_lbl),
            run_time=0.6,
        )

        # Title update
        title2 = (
            Text("Cross-Validation Score Summary",
                 font_size=18, color=C_WHITE, weight=BOLD)
            .to_edge(UP, buff=0.35)
        )
        self.play(ReplacementTransform(title, title2), run_time=0.4)

        # Bar chart axes
        ax = Axes(
            x_range=[0, K + 1, 1],
            y_range=[Y_LO, Y_HI, 0.04],
            x_length=6.2, y_length=3.2,
            **_AX,
        ).move_to([-1.2, -0.85, 0])

        self.play(Create(ax), run_time=0.5)

        # Y-axis tick labels
        y_ticks = VGroup(*[
            Text(f"{v:.2f}", font_size=10, color=C_DIM)
            .next_to(ax.c2p(0, v), LEFT, buff=0.12)
            for v in np.arange(Y_LO + 0.02, Y_HI, 0.04)
        ])
        self.play(FadeIn(y_ticks), run_time=0.3)

        # Bars — grow from bottom (GrowFromEdge DOWN)
        bars = VGroup()
        x_tick_lbls = VGroup()
        for k in range(K):
            xp = k + 1
            sc = SCORES[k]
            # Bar geometry in screen coords
            bl = ax.c2p(xp - 0.28, Y_LO)
            tr = ax.c2p(xp + 0.28, sc)
            bw = tr[0] - bl[0]
            bh = tr[1] - bl[1]
            bar = Rectangle(
                width=bw, height=bh,
                fill_color=FOLD_COLORS[k], fill_opacity=0.85,
                stroke_width=0,
            ).move_to([(bl[0] + tr[0]) / 2, (bl[1] + tr[1]) / 2, 0])
            bars.add(bar)

            # X-axis label
            xl = (
                Text(f"F{k+1}", font_size=12, color=FOLD_COLORS[k])
                .next_to(ax.c2p(xp, Y_LO), DOWN, buff=0.14)
            )
            x_tick_lbls.add(xl)

        self.play(
            LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.12),
            FadeIn(x_tick_lbls),
            run_time=1.0,
        )
        self.wait(0.3)

        # ±std shaded band
        band_lo = ax.c2p(0, MEAN - STD)
        band_hi = ax.c2p(K + 1, MEAN + STD)
        std_band = Rectangle(
            width=band_hi[0] - band_lo[0],
            height=band_hi[1] - band_lo[1],
            fill_color="#58a6ff", fill_opacity=0.12, stroke_width=0,
        ).move_to([
            (band_lo[0] + band_hi[0]) / 2,
            (band_lo[1] + band_hi[1]) / 2,
            0,
        ])

        # Mean dashed line
        mean_line = DashedLine(
            ax.c2p(0, MEAN), ax.c2p(K + 1, MEAN),
            color="#58a6ff", stroke_width=2.0, dash_length=0.13,
        )
        mean_lbl = (
            Text(f"mean = {MEAN:.2f}", font_size=11, color="#58a6ff")
            .next_to(ax.c2p(0, MEAN), LEFT, buff=0.12)
        )
        self.play(
            FadeIn(std_band), Create(mean_line), FadeIn(mean_lbl),
            run_time=0.7,
        )
        self.wait(0.3)

        # CV summary (large)
        cv_summary = (
            Text(f"CV Score:  {MEAN:.2f}  \u00b1  {STD:.2f}",
                 font_size=17, color=C_WHITE, weight=BOLD)
            .move_to([3.4, -0.5, 0])
        )
        caption = (
            Text("low-variance estimate of\ngeneralisation error",
                 font_size=11, color=C_DIM)
            .next_to(cv_summary, DOWN, buff=0.18)
        )
        self.play(FadeIn(cv_summary), run_time=0.5)
        self.play(FadeIn(caption), run_time=0.4)
        self.wait(2.5)
