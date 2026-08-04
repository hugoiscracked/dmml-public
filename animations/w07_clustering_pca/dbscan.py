"""
W07 — DBSCAN: Density-Based Clustering

  Phase 1: Show two crescent clusters + noise. Grow an ε-ball from a core
           point; highlight its neighbours (min_samples met). Expand each
           cluster colour by colour until all reachable points are labelled.
  Phase 2: Label each point — core / border / noise — and explain why
           k-means would fail on these shapes.

Render:
  ../../env/bin/manim -pql dbscan.py DBSCAN
  ../../env/bin/manim -pqh dbscan.py DBSCAN
"""

from manim import *
import numpy as np
from manim.utils.color import ManimColor
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN as _DBSCAN_SK

# ── Palette ───────────────────────────────────────────────────────────────────
C_BG     = "#0d1117"
C_WHITE  = "#ffffff"
C_DIM    = "#8b949e"
C_GRID   = "#21262d"

C_UNVIS  = "#8b949e"
C_CLUST0 = "#58a6ff"   # blue  – cluster 0
C_CLUST1 = "#3fb950"   # green – cluster 1
C_NOISE  = "#f78166"   # red   – noise
C_EPS    = "#e3b341"   # gold  – ε-ball

EPS        = 0.38
MIN_SAMP   = 4

# ── Data ──────────────────────────────────────────────────────────────────────
np.random.seed(11)
_BASE, _ = make_moons(n_samples=44, noise=0.08, random_state=11)
# Centre and scale to fill the axes comfortably
_BASE -= _BASE.mean(axis=0)
_BASE *= 2.2

# A few explicit noise points (outside both crescents)
_NOISE_PTS = np.array([
    [-3.4,  1.8], [ 3.5, -2.0], [-3.0, -2.2],
    [ 3.2,  1.9], [ 0.2,  2.8], [-0.4, -2.8],
])
_PTS   = np.vstack([_BASE, _NOISE_PTS])
_N     = len(_PTS)

# DBSCAN labels
_SK_LABELS = _DBSCAN_SK(eps=EPS, min_samples=MIN_SAMP).fit_predict(_PTS)

def _label_color(lbl):
    if lbl == -1: return C_NOISE
    if lbl ==  0: return C_CLUST0
    return C_CLUST1


class DBSCAN(Scene):

    def construct(self):
        self.camera.background_color = C_BG
        self._phase1()
        self._phase2()

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _ax():
        return Axes(
            x_range=[-4.2, 4.2, 1],
            y_range=[-3.5, 3.5, 1],
            x_length=7.6, y_length=6.2,
            axis_config={"color": C_GRID, "stroke_width": 1.0, "include_ticks": False},
            tips=False,
        ).move_to([0, -0.3, 0])

    # ── Phase 1: ε-ball expansion ─────────────────────────────────────────────
    def _phase1(self):
        title = (
            Text("DBSCAN — Density-Based Clustering",
                 font_size=18, color=C_WHITE, weight=BOLD)
            .to_edge(UP, buff=0.35)
        )
        sub = (
            Text("clusters grow from dense core points — no need to specify k",
                 font_size=12, color=C_DIM)
            .next_to(title, DOWN, buff=0.12)
        )
        self.play(FadeIn(title), FadeIn(sub), run_time=0.5)

        ax = self._ax()
        self.play(Create(ax), run_time=0.4)

        dot_mobs = [
            Dot(ax.c2p(*_PTS[i]), radius=0.10, color=C_UNVIS)
            .set_stroke(color=C_BG, width=0.6)
            for i in range(_N)
        ]
        dot_grp = VGroup(*dot_mobs)
        self.play(
            LaggedStart(*[GrowFromCenter(d) for d in dot_mobs], lag_ratio=0.04),
            run_time=1.0,
        )
        self.wait(0.3)

        # Pick a core point from cluster 0
        CORE_IDX = next(
            i for i in range(len(_BASE))
            if _SK_LABELS[i] == 0
            and sum(1 for j in range(_N) if np.linalg.norm(_PTS[i] - _PTS[j]) <= EPS) >= MIN_SAMP
        )
        core_pt     = _PTS[CORE_IDX]
        core_screen = ax.c2p(*core_pt)
        eps_r_px    = ax.c2p(core_pt[0] + EPS, core_pt[1])[0] - core_screen[0]

        # Highlight selected point
        self.play(dot_mobs[CORE_IDX].animate.set_color(C_EPS).scale(1.4), run_time=0.3)

        # Draw ε-ball
        eps_circle = Circle(
            radius=eps_r_px,
            color=C_EPS, stroke_width=1.8,
            fill_color=C_EPS, fill_opacity=0.08,
        ).move_to(core_screen)
        eps_lbl = (
            Text("ε", font_size=16, color=C_EPS)
            .next_to(core_screen + RIGHT * eps_r_px, RIGHT, buff=0.1)
        )
        step1_lbl = (
            Text("draw ε-ball around this point", font_size=12, color=C_DIM)
            .to_edge(DOWN, buff=0.42)
        )
        self.play(Create(eps_circle), FadeIn(eps_lbl), FadeIn(step1_lbl), run_time=0.6)
        self.wait(0.4)

        # Count and highlight neighbours
        neighbours = [
            i for i in range(_N)
            if i != CORE_IDX and np.linalg.norm(_PTS[i] - core_pt) <= EPS
        ]
        n_nbrs = len(neighbours)
        step2_lbl = (
            Text(f"{n_nbrs} neighbours inside ε  ≥  min_samples ({MIN_SAMP})  →  core point",
                 font_size=12, color=C_DIM)
            .to_edge(DOWN, buff=0.42)
        )
        self.play(FadeOut(step1_lbl), run_time=0.2)
        self.play(
            *[dot_mobs[i].animate.set_color(C_CLUST0) for i in neighbours],
            run_time=0.5,
        )
        self.play(FadeIn(step2_lbl), run_time=0.3)
        self.wait(0.4)
        self.play(FadeOut(eps_circle), FadeOut(eps_lbl), FadeOut(step2_lbl), run_time=0.3)

        # Expand cluster 0
        c0_rest = [
            i for i in range(_N)
            if _SK_LABELS[i] == 0 and i not in neighbours and i != CORE_IDX
        ]
        expand_lbl = (
            Text("expand: each neighbour becomes a new seed → cluster grows …",
                 font_size=12, color=C_DIM)
            .to_edge(DOWN, buff=0.42)
        )
        self.play(FadeIn(expand_lbl), run_time=0.3)
        if c0_rest:
            self.play(
                LaggedStart(
                    *[dot_mobs[i].animate.set_color(C_CLUST0) for i in c0_rest],
                    lag_ratio=0.06,
                ),
                run_time=0.8,
            )
        self.wait(0.3)

        # Expand cluster 1
        c1_idxs = [i for i in range(_N) if _SK_LABELS[i] == 1]
        expand_lbl2 = (
            Text("second dense region → separate cluster",
                 font_size=12, color=C_DIM)
            .to_edge(DOWN, buff=0.42)
        )
        self.play(ReplacementTransform(expand_lbl, expand_lbl2), run_time=0.3)
        if c1_idxs:
            self.play(
                LaggedStart(
                    *[dot_mobs[i].animate.set_color(C_CLUST1) for i in c1_idxs],
                    lag_ratio=0.06,
                ),
                run_time=0.8,
            )
        self.wait(0.3)

        # Mark noise
        noise_idxs = [i for i in range(_N) if _SK_LABELS[i] == -1]
        noise_lbl = (
            Text("remaining sparse points → noise  (no cluster assigned)",
                 font_size=12, color=C_DIM)
            .to_edge(DOWN, buff=0.42)
        )
        self.play(ReplacementTransform(expand_lbl2, noise_lbl), run_time=0.3)
        if noise_idxs:
            self.play(
                *[dot_mobs[i].animate.set_color(C_NOISE) for i in noise_idxs],
                run_time=0.5,
            )
        self.wait(1.5)
        self.play(FadeOut(VGroup(title, sub, ax, dot_grp, noise_lbl)), run_time=0.5)

    # ── Phase 2: Core / Border / Noise + k-means contrast ────────────────────
    def _phase2(self):
        title = (
            Text("Core, Border, and Noise — and why k-means fails here",
                 font_size=18, color=C_WHITE, weight=BOLD)
            .to_edge(UP, buff=0.35)
        )
        self.play(FadeIn(title), run_time=0.4)

        ax = self._ax()
        self.play(Create(ax), run_time=0.3)

        # Compute core vs border per point
        def _role(i):
            if _SK_LABELS[i] == -1:
                return "noise"
            n_nbrs = sum(
                1 for j in range(_N)
                if np.linalg.norm(_PTS[i] - _PTS[j]) <= EPS
            )
            return "core" if n_nbrs >= MIN_SAMP else "border"

        role_cols = {
            "core":   C_CLUST0,
            "border": "#2dd4bf",
            "noise":  C_NOISE,
        }
        dot_mobs = [
            Dot(ax.c2p(*_PTS[i]), radius=0.10,
                color=role_cols[_role(i)])
            .set_stroke(color=C_BG, width=0.5)
            for i in range(_N)
        ]
        dot_grp = VGroup(*dot_mobs)
        self.play(
            LaggedStart(*[GrowFromCenter(d) for d in dot_mobs], lag_ratio=0.03),
            run_time=0.8,
        )

        # Legend (right side)
        LEG_X = 3.8
        legend_items = [
            ("Core point",   C_CLUST0,   "reachable + dense enough"),
            ("Border point", "#2dd4bf",  "reachable, not dense"),
            ("Noise",        C_NOISE,    "not reachable from any core"),
        ]
        legend = VGroup()
        for k, (name, col, desc) in enumerate(legend_items):
            dot  = Dot([LEG_X, 1.5 - k * 1.2, 0], radius=0.12, color=col)
            lbl  = Text(name, font_size=13, color=C_WHITE).next_to(dot, RIGHT, buff=0.15)
            desc_t = Text(desc, font_size=10, color=C_DIM).next_to(lbl, DOWN, buff=0.06).align_to(lbl, LEFT)
            legend.add(dot, lbl, desc_t)
        self.play(FadeIn(legend), run_time=0.5)

        caption = (
            Text(
                "k-means assumes round clusters and always assigns every point\n"
                "DBSCAN finds arbitrary shapes and detects outliers automatically",
                font_size=12, color=C_DIM,
            )
            .to_edge(DOWN, buff=0.42)
        )
        self.play(FadeIn(caption), run_time=0.5)
        self.wait(2.5)
