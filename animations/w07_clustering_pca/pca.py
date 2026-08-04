"""
W07 — Principal Component Analysis (PCA)

  Phase 1: 2D data cloud with correlation; PC1 and PC2 arrows drawn as the
           directions of maximum / orthogonal variance.
  Phase 2: Project all points onto PC1 — show compression to 1D.
  Phase 3: Explained variance bar chart (PC1, PC2) + cumulative line.

Render:
  ../../env/bin/manim -pql pca.py PCA
  ../../env/bin/manim -pqh pca.py PCA
"""

from manim import *
import numpy as np
from manim.utils.color import ManimColor

# ── Palette ───────────────────────────────────────────────────────────────────
C_BG    = "#0d1117"
C_WHITE = "#ffffff"
C_DIM   = "#8b949e"
C_GRID  = "#21262d"
C_DATA  = "#58a6ff"   # blue – data points
C_PC1   = "#f78166"   # red  – PC1
C_PC2   = "#3fb950"   # green – PC2
C_PROJ  = "#e3b341"   # gold – projections

# ── Data: correlated 2D cloud ─────────────────────────────────────────────────
np.random.seed(7)
_N   = 30
_COV = [[1.6, 1.2], [1.2, 1.0]]
_RAW = np.random.multivariate_normal([0, 0], _COV, _N)
_RAW -= _RAW.mean(axis=0)   # centre

# PCA via SVD
_U, _S, _Vt = np.linalg.svd(_RAW, full_matrices=False)
_PC1 = _Vt[0]   # direction of most variance
_PC2 = _Vt[1]   # orthogonal

_VAR_TOTAL    = float((_S ** 2).sum())
_EXP_VAR      = (_S ** 2) / _VAR_TOTAL          # [ev1, ev2]
_CUM_VAR      = np.cumsum(_EXP_VAR)

# Scale arrows to look good on screen
_PC1_SCALE = 2.8
_PC2_SCALE = 1.4


class PCA(Scene):

    def construct(self):
        self.camera.background_color = C_BG
        self._phase1()
        self._phase2()
        self._phase3()

    # ── Phase 1: Data cloud + PC arrows ───────────────────────────────────────
    def _phase1(self):
        title = (
            Text("PCA — Principal Component Directions",
                 font_size=18, color=C_WHITE, weight=BOLD)
            .to_edge(UP, buff=0.35)
        )
        sub = (
            Text("PC1 = direction of maximum variance  ·  PC2 = orthogonal",
                 font_size=12, color=C_DIM)
            .next_to(title, DOWN, buff=0.12)
        )
        self.play(FadeIn(title), FadeIn(sub), run_time=0.5)

        ax = Axes(
            x_range=[-4.0, 4.0, 1],
            y_range=[-3.5, 3.5, 1],
            x_length=7.8, y_length=6.0,
            axis_config={"color": C_GRID, "stroke_width": 1.0, "include_ticks": False},
            tips=False,
        ).move_to([0, -0.35, 0])
        self.play(Create(ax), run_time=0.4)

        dot_mobs = VGroup(*[
            Dot(ax.c2p(*pt), radius=0.08, color=C_DATA)
            .set_stroke(color=C_BG, width=0.5)
            for pt in _RAW
        ])
        self.play(
            LaggedStart(*[GrowFromCenter(d) for d in dot_mobs], lag_ratio=0.04),
            run_time=0.9,
        )
        self.wait(0.3)

        # PC1 arrow
        origin = ax.c2p(0, 0)
        pc1_tip = ax.c2p(*(_PC1 * _PC1_SCALE))
        pc1_arr = Arrow(
            origin, pc1_tip,
            color=C_PC1, stroke_width=3.0, buff=0,
            max_tip_length_to_length_ratio=0.12,
        )
        pc1_lbl = (
            Text("PC1", font_size=14, color=C_PC1, weight=BOLD)
            .next_to(pc1_tip, UR, buff=0.12)
        )

        # PC2 arrow (shorter)
        pc2_tip = ax.c2p(*(_PC2 * _PC2_SCALE))
        pc2_arr = Arrow(
            origin, pc2_tip,
            color=C_PC2, stroke_width=2.2, buff=0,
            max_tip_length_to_length_ratio=0.16,
        )
        pc2_lbl = (
            Text("PC2", font_size=14, color=C_PC2, weight=BOLD)
            .next_to(pc2_tip, UL, buff=0.12)
        )

        self.play(Create(pc1_arr), FadeIn(pc1_lbl), run_time=0.6)
        self.play(Create(pc2_arr), FadeIn(pc2_lbl), run_time=0.5)

        pct1 = int(round(_EXP_VAR[0] * 100))
        pct2 = int(round(_EXP_VAR[1] * 100))
        caption = (
            Text(
                f"PC1 captures {pct1}% of variance  ·  PC2 captures {pct2}%",
                font_size=12, color=C_DIM,
            )
            .to_edge(DOWN, buff=0.42)
        )
        self.play(FadeIn(caption), run_time=0.4)
        self.wait(1.5)

        self.play(
            FadeOut(VGroup(title, sub, ax, dot_mobs,
                           pc1_arr, pc1_lbl, pc2_arr, pc2_lbl, caption)),
            run_time=0.5,
        )

    # ── Phase 2: Projection onto PC1 ─────────────────────────────────────────
    def _phase2(self):
        title = (
            Text("Projection onto PC1 — compressing to 1D",
                 font_size=18, color=C_WHITE, weight=BOLD)
            .to_edge(UP, buff=0.35)
        )
        sub = (
            Text("each point drops a perpendicular onto PC1 — its coordinate along PC1 is kept",
                 font_size=12, color=C_DIM)
            .next_to(title, DOWN, buff=0.12)
        )
        self.play(FadeIn(title), FadeIn(sub), run_time=0.5)

        ax = Axes(
            x_range=[-4.0, 4.0, 1],
            y_range=[-3.5, 3.5, 1],
            x_length=7.8, y_length=6.0,
            axis_config={"color": C_GRID, "stroke_width": 1.0, "include_ticks": False},
            tips=False,
        ).move_to([0, -0.35, 0])
        self.play(Create(ax), run_time=0.3)

        # Data dots
        dot_mobs = VGroup(*[
            Dot(ax.c2p(*pt), radius=0.08, color=C_DATA).set_stroke(color=C_BG, width=0.5)
            for pt in _RAW
        ])
        self.play(
            LaggedStart(*[GrowFromCenter(d) for d in dot_mobs], lag_ratio=0.03),
            run_time=0.6,
        )

        # PC1 axis line (extended both ways)
        pc1_line = Line(
            ax.c2p(*(-_PC1 * 3.8)),
            ax.c2p(*( _PC1 * 3.8)),
            color=C_PC1, stroke_width=2.0,
        )
        pc1_lbl = (
            Text("PC1", font_size=13, color=C_PC1, weight=BOLD)
            .next_to(ax.c2p(*(_PC1 * 3.8)), RIGHT, buff=0.12)
        )
        self.play(Create(pc1_line), FadeIn(pc1_lbl), run_time=0.5)
        self.wait(0.3)

        # Projections: drop perpendiculars and move dots onto PC1
        projections = (_RAW @ _PC1)[:, None] * _PC1   # shape (N, 2)

        proj_lines = VGroup(*[
            DashedLine(
                ax.c2p(*_RAW[i]),
                ax.c2p(*projections[i]),
                color=C_PROJ, stroke_width=1.0, dash_length=0.08,
            )
            for i in range(_N)
        ])
        proj_dots = VGroup(*[
            Dot(ax.c2p(*projections[i]), radius=0.08, color=C_PROJ)
            .set_stroke(color=C_BG, width=0.5)
            for i in range(_N)
        ])

        self.play(
            LaggedStart(*[Create(l) for l in proj_lines], lag_ratio=0.04),
            run_time=0.7,
        )
        self.play(
            LaggedStart(*[GrowFromCenter(d) for d in proj_dots], lag_ratio=0.04),
            run_time=0.6,
        )
        self.wait(0.3)

        # Fade original dots, leaving only the 1D projections
        caption = (
            Text("original 2D cloud → projected 1D coordinates along PC1",
                 font_size=12, color=C_DIM)
            .to_edge(DOWN, buff=0.42)
        )
        self.play(
            FadeOut(dot_mobs),
            FadeOut(proj_lines),
            FadeIn(caption),
            run_time=0.6,
        )
        self.wait(1.8)
        self.play(
            FadeOut(VGroup(title, sub, ax, pc1_line, pc1_lbl, proj_dots, caption)),
            run_time=0.5,
        )

    # ── Phase 3: Explained variance bar chart ─────────────────────────────────
    def _phase3(self):
        title = (
            Text("Explained Variance per Component",
                 font_size=18, color=C_WHITE, weight=BOLD)
            .to_edge(UP, buff=0.35)
        )
        sub = (
            Text("how much of the total variance each principal component captures",
                 font_size=12, color=C_DIM)
            .next_to(title, DOWN, buff=0.12)
        )
        self.play(FadeIn(title), FadeIn(sub), run_time=0.5)

        ax = Axes(
            x_range=[0.3, 2.7, 1],
            y_range=[0, 1.15, 0.2],
            x_length=5.5, y_length=5.2,
            axis_config={"color": C_GRID, "stroke_width": 1.2, "include_ticks": False},
            tips=False,
        ).move_to([-0.5, -0.6, 0])

        # Y-axis tick labels (%)
        y_tick_lbls = VGroup(*[
            Text(f"{int(v * 100)}%", font_size=12, color=C_DIM)
            .next_to(ax.c2p(0.3, v), LEFT, buff=0.15)
            for v in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        ])
        x_tick_lbls = VGroup(*[
            Text(lbl, font_size=13, color=C_WHITE).next_to(ax.c2p(x, 0), DOWN, buff=0.2)
            for lbl, x in [("PC1", 1.0), ("PC2", 2.0)]
        ])
        ax_y_lbl = (
            Text("Explained\nvariance", font_size=12, color=C_DIM)
            .rotate(PI / 2)
            .next_to(ax, LEFT, buff=0.55)
        )
        self.play(Create(ax), FadeIn(y_tick_lbls), FadeIn(x_tick_lbls),
                  FadeIn(ax_y_lbl), run_time=0.5)

        # Bars
        bar_cols = [C_PC1, C_PC2]
        bar_mobs = []
        val_lbls = []
        for i, (ev, col) in enumerate(zip(_EXP_VAR, bar_cols)):
            x_center = 1.0 + i
            bar_h_px = ax.c2p(x_center, ev)[1] - ax.c2p(x_center, 0)[1]
            bar = Rectangle(
                width=0.7,
                height=bar_h_px,
                color=col, fill_opacity=0.85, stroke_width=0,
            ).align_to(ax.c2p(x_center, 0), DOWN).shift(LEFT * 0.35)
            bar.move_to([ax.c2p(x_center, 0)[0], ax.c2p(x_center, ev / 2)[1], 0])

            pct_lbl = (
                Text(f"{ev * 100:.1f}%", font_size=13, color=col, weight=BOLD)
                .next_to(ax.c2p(x_center, ev), UP, buff=0.12)
            )
            self.play(GrowFromEdge(bar, DOWN), run_time=0.55)
            self.play(FadeIn(pct_lbl), run_time=0.25)
            bar_mobs.append(bar)
            val_lbls.append(pct_lbl)

        # Cumulative line
        cum_x = [ax.c2p(1.0, 0)[0], ax.c2p(1.0, _CUM_VAR[0])[0], ax.c2p(2.0, _CUM_VAR[1])[0]]
        cum_y = [ax.c2p(1.0, _CUM_VAR[0])[1], ax.c2p(1.0, _CUM_VAR[0])[1], ax.c2p(2.0, _CUM_VAR[1])[1]]

        cum_pts_screen = [
            ax.c2p(1.0, _CUM_VAR[0]),
            ax.c2p(2.0, _CUM_VAR[1]),
        ]
        cum_dots = VGroup(*[
            Dot(p, radius=0.1, color=C_PROJ) for p in cum_pts_screen
        ])
        cum_line = Line(cum_pts_screen[0], cum_pts_screen[1],
                        color=C_PROJ, stroke_width=2.0)
        cum_lbl = (
            Text("cumulative", font_size=11, color=C_PROJ)
            .next_to(cum_pts_screen[1], RIGHT, buff=0.12)
        )
        cum_val = (
            Text(f"{_CUM_VAR[1] * 100:.1f}%", font_size=12, color=C_PROJ)
            .next_to(cum_pts_screen[1], UR, buff=0.1)
        )
        self.play(Create(cum_line), FadeIn(cum_dots), FadeIn(cum_lbl), FadeIn(cum_val), run_time=0.6)

        caption = (
            Text(
                f"PC1 alone explains {_EXP_VAR[0]*100:.0f}% — "
                f"two components together capture {_CUM_VAR[1]*100:.0f}% of variance",
                font_size=12, color=C_DIM,
            )
            .to_edge(DOWN, buff=0.42)
        )
        self.play(FadeIn(caption), run_time=0.4)
        self.wait(2.5)
