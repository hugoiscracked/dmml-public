"""
W07 — K-Means Clustering

  Phase 1: Lloyd's algorithm — random centroids → repeated assign/update until
           convergence (3 iterations shown explicitly).
  Phase 2: Elbow method — inertia vs k curve; elbow annotated.

Render:
  ../../env/bin/manim -pql kmeans.py KMeans
  ../../env/bin/manim -pqh kmeans.py KMeans
"""

from manim import *
import numpy as np
from manim.utils.color import ManimColor

# ── Palette ───────────────────────────────────────────────────────────────────
C_BG    = "#0d1117"
C_WHITE = "#ffffff"
C_DIM   = "#8b949e"
C_GRID  = "#21262d"

CLUSTER_COLS = ["#58a6ff", "#3fb950", "#ffa657"]   # blue, green, amber
C_CENT  = "#f78166"   # red-ish centroid marker

# ── Data: three well-separated blobs ─────────────────────────────────────────
np.random.seed(5)
_MEANS = [[-2.2, 1.2], [1.8, 1.5], [0.0, -1.8]]
_N     = 14   # points per cluster
_PTS   = np.vstack([
    np.random.multivariate_normal(m, [[0.28, 0], [0, 0.28]], _N)
    for m in _MEANS
])
_TRUE_LABELS = np.repeat([0, 1, 2], _N)

# ── K-means iterations (k=3, hand-picked init far from true centres) ──────────
_INIT_C = np.array([[-2.5, -1.5], [0.0, 2.5], [2.5, -1.5]])

def _assign(pts, centroids):
    dists = np.linalg.norm(pts[:, None] - centroids[None], axis=2)
    return np.argmin(dists, axis=1)

def _update(pts, labels, k):
    return np.array([pts[labels == i].mean(axis=0) for i in range(k)])

# Pre-compute 4 iterations
_centroids = [_INIT_C.copy()]
_labels    = []
for _ in range(4):
    lbl = _assign(_PTS, _centroids[-1])
    _labels.append(lbl)
    _centroids.append(_update(_PTS, lbl, 3))

# ── Elbow data ────────────────────────────────────────────────────────────────
# Pre-computed inertias for k=1..7 (representative values)
_KS      = list(range(1, 8))
_INERTIA = [62.0, 28.5, 8.2, 6.8, 6.0, 5.5, 5.1]


def _dot_color(label):
    return CLUSTER_COLS[int(label)]


class KMeans(Scene):

    def construct(self):
        self.camera.background_color = C_BG
        self._phase1()
        self._phase2()

    # ── Phase 1: Lloyd's algorithm ────────────────────────────────────────────
    def _phase1(self):
        title = (
            Text("K-Means — Lloyd's Algorithm  (k = 3)",
                 font_size=18, color=C_WHITE, weight=BOLD)
            .to_edge(UP, buff=0.35)
        )
        self.play(FadeIn(title), run_time=0.5)

        ax = Axes(
            x_range=[-3.8, 3.8, 1],
            y_range=[-3.2, 3.2, 1],
            x_length=7.5, y_length=6.2,
            axis_config={"color": C_GRID, "stroke_width": 1.0, "include_ticks": False},
            tips=False,
        ).move_to([0, -0.3, 0])
        self.play(Create(ax), run_time=0.4)

        # Draw all points grey initially
        dot_mobs = VGroup(*[
            Dot(ax.c2p(*pt), radius=0.09, color=C_DIM).set_stroke(color=C_BG, width=0.6)
            for pt in _PTS
        ])
        self.play(
            LaggedStart(*[GrowFromCenter(d) for d in dot_mobs], lag_ratio=0.04),
            run_time=0.8,
        )

        # Place initial centroids
        cent_mobs = VGroup(*[
            Star(n=4, outer_radius=0.18, inner_radius=0.09,
                 color=C_CENT, fill_opacity=1.0, stroke_width=0)
            .move_to(ax.c2p(*c))
            for c in _centroids[0]
        ])
        init_lbl = (
            Text("random initialisation", font_size=12, color=C_DIM)
            .to_edge(DOWN, buff=0.42)
        )
        self.play(
            LaggedStart(*[GrowFromCenter(s) for s in cent_mobs], lag_ratio=0.15),
            FadeIn(init_lbl),
            run_time=0.7,
        )
        self.wait(0.5)
        self.play(FadeOut(init_lbl), run_time=0.2)

        # Iterate
        for it in range(3):
            labels = _labels[it]

            # E-step: colour each point by nearest centroid
            step_lbl = (
                Text(f"Step {it + 1}a — assign each point to nearest centroid",
                     font_size=12, color=C_DIM)
                .to_edge(DOWN, buff=0.42)
            )
            self.play(FadeIn(step_lbl), run_time=0.3)
            self.play(
                *[dot_mobs[j].animate.set_color(_dot_color(labels[j]))
                  for j in range(len(_PTS))],
                run_time=0.6,
            )
            self.wait(0.3)
            self.play(FadeOut(step_lbl), run_time=0.2)

            # M-step: move centroids to cluster means
            new_cents = _centroids[it + 1]
            step_lbl2 = (
                Text(f"Step {it + 1}b — move centroids to cluster means",
                     font_size=12, color=C_DIM)
                .to_edge(DOWN, buff=0.42)
            )
            self.play(FadeIn(step_lbl2), run_time=0.3)
            self.play(
                *[cent_mobs[c].animate.move_to(ax.c2p(*new_cents[c]))
                  for c in range(3)],
                run_time=0.7,
            )
            self.wait(0.3)
            self.play(FadeOut(step_lbl2), run_time=0.2)

        # Final assignment
        final_labels = _labels[3]
        conv_lbl = Text("converged", font_size=13, color=C_WHITE, weight=BOLD).to_edge(DOWN, buff=0.42)
        self.play(
            *[dot_mobs[j].animate.set_color(_dot_color(final_labels[j]))
              for j in range(len(_PTS))],
            FadeIn(conv_lbl),
            run_time=0.6,
        )

        # Draw cluster boundary circles (approximate)
        for i, col in enumerate(CLUSTER_COLS):
            cluster_pts = _PTS[final_labels == i]
            cx, cy = _centroids[-1][i]
            r = float(np.max(np.linalg.norm(cluster_pts - _centroids[-1][i], axis=1))) + 0.25
            circle = Circle(
                radius=ax.c2p(cx + r, cy)[0] - ax.c2p(cx, cy)[0],
                color=col, stroke_width=1.2, stroke_opacity=0.5, fill_opacity=0,
            ).move_to(ax.c2p(cx, cy))
            self.play(Create(circle), run_time=0.4)

        self.wait(1.5)
        self.play(FadeOut(VGroup(title, ax, dot_mobs, cent_mobs, conv_lbl)), run_time=0.6)

    # ── Phase 2: Elbow method ─────────────────────────────────────────────────
    def _phase2(self):
        title = (
            Text("Elbow Method — choosing k",
                 font_size=18, color=C_WHITE, weight=BOLD)
            .to_edge(UP, buff=0.35)
        )
        sub = (
            Text("run k-means for k = 1 … 7  ·  plot inertia (within-cluster sum of squares)",
                 font_size=12, color=C_DIM)
            .next_to(title, DOWN, buff=0.12)
        )
        self.play(FadeIn(title), FadeIn(sub), run_time=0.5)

        ax = Axes(
            x_range=[0.5, 7.5, 1],
            y_range=[0, 70, 10],
            x_length=8.0, y_length=4.8,
            axis_config={"color": C_GRID, "stroke_width": 1.2, "include_ticks": False},
            tips=False,
        ).move_to([0.3, -0.6, 0])

        # Axis labels (manual ticks)
        x_tick_lbls = VGroup(*[
            Text(str(k), font_size=13, color=C_DIM).next_to(ax.c2p(k, 0), DOWN, buff=0.18)
            for k in _KS
        ])
        y_tick_lbls = VGroup(*[
            Text(str(v), font_size=12, color=C_DIM).next_to(ax.c2p(0.5, v), LEFT, buff=0.15)
            for v in range(0, 71, 10)
        ])
        ax_x_lbl = Text("k  (number of clusters)", font_size=13, color=C_DIM).next_to(ax, DOWN, buff=0.5)
        ax_y_lbl = (
            Text("Inertia", font_size=13, color=C_DIM)
            .rotate(PI / 2)
            .next_to(ax, LEFT, buff=0.5)
        )

        self.play(Create(ax), FadeIn(x_tick_lbls), FadeIn(y_tick_lbls),
                  FadeIn(ax_x_lbl), FadeIn(ax_y_lbl), run_time=0.6)

        # Plot points and line one by one
        dot_mobs = VGroup()
        prev_dot = None
        for k, inertia in zip(_KS, _INERTIA):
            dot = Dot(ax.c2p(k, inertia), radius=0.1, color="#58a6ff")
            dot_mobs.add(dot)
            if prev_dot is None:
                self.play(GrowFromCenter(dot), run_time=0.25)
            else:
                line = Line(prev_dot.get_center(), dot.get_center(),
                            color="#58a6ff", stroke_width=2.0)
                self.play(Create(line), GrowFromCenter(dot), run_time=0.3)
            prev_dot = dot

        # Annotate elbow at k=3
        elbow_pt = ax.c2p(3, _INERTIA[2])
        elbow_circle = Circle(radius=0.28, color=C_CENT, stroke_width=2.0, fill_opacity=0).move_to(elbow_pt)
        elbow_lbl = (
            Text("elbow\n(k = 3)", font_size=13, color=C_CENT)
            .next_to(elbow_pt, UR, buff=0.2)
        )
        elbow_arrow = Arrow(
            elbow_lbl.get_left() + LEFT * 0.1,
            elbow_pt + UR * 0.28,
            color=C_CENT, stroke_width=1.8, buff=0.05,
            max_tip_length_to_length_ratio=0.25,
        )
        self.play(Create(elbow_circle), FadeIn(elbow_lbl), Create(elbow_arrow), run_time=0.6)

        caption = (
            Text("beyond k = 3 the gain in inertia flattens — pick the elbow",
                 font_size=12, color=C_DIM)
            .to_edge(DOWN, buff=0.42)
        )
        self.play(FadeIn(caption), run_time=0.4)
        self.wait(2.5)
