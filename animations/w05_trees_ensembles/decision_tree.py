"""
W05 — Decision Tree: Recursive Partitioning

Three-phase animation for Lecture 1 (Decision Trees).

  Phase 1: A 2D scatter appears; the tree builds its axis-aligned splits one
           by one — first a horizontal cut (x₂ < 0.45), then a vertical cut
           in the lower half (x₁ < 0.5).  Each new region is shaded by
           predicted class.

  Phase 2: The scatter group shrinks to the left half; the corresponding
           tree diagram (nodes + edges + Yes/No labels) grows in on the right.

  Phase 3: Side-by-side comparison — depth=2 (clean, 3 regions) vs an
           unconstrained tree trained on the same data plus four noise points
           (11 tiny jagged regions).  Caption: "deep tree overfits noise."

Render:
  ../../env/bin/manim -pql decision_tree.py DecisionTree
  ../../env/bin/manim -pqh decision_tree.py DecisionTree
"""

from manim import *
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# ── Palette ───────────────────────────────────────────────────────────────────
C_BG   = "#0d1117"
C_C0   = "#ffa657"   # amber  – class 0
C_C1   = "#2dd4bf"   # teal   – class 1
C_SPLT = "#e3b341"   # gold   – split lines
C_NODE = "#161b22"   # near-black – internal node fill
C_DIM  = "#8b949e"

_AX = dict(
    axis_config={"color": "#21262d", "stroke_width": 1.2, "include_ticks": False},
    tips=False,
)

# ── Data ──────────────────────────────────────────────────────────────────────
_X0 = np.array([[0.10,0.10],[0.20,0.30],[0.30,0.15],[0.15,0.40],
                [0.35,0.35],[0.25,0.20],[0.40,0.12],[0.10,0.38]])
_X1 = np.array([[0.60,0.20],[0.70,0.50],[0.80,0.80],[0.90,0.30],
                [0.65,0.70],[0.75,0.15],[0.55,0.60],[0.85,0.50],
                [0.10,0.70],[0.20,0.80],[0.35,0.65],[0.30,0.90],[0.15,0.60]])
_X = np.vstack([_X0, _X1])
_y = np.array([0]*8 + [1]*13)

# Four noise points that force the deep tree to overfit
_X_NOISE = np.vstack([_X, [[0.18,0.25],[0.32,0.18],[0.22,0.72],[0.78,0.68]]])
_Y_NOISE = np.concatenate([_y, [1, 1, 0, 0]])

# Hardcoded splits from depth-2 tree (x2 < 0.450, then x1 < 0.500)
_SPLIT_H = 0.450   # horizontal cut: y = 0.450
_SPLIT_V = 0.500   # vertical cut: x = 0.500 (bottom half only)


# ── Leaf-rectangle extraction ─────────────────────────────────────────────────
def _leaf_rects(clf):
    """Return list of (x0, x1, y0, y1, cls) for every leaf of a fitted tree."""
    t = clf.tree_
    out = []

    def _recurse(node, a, b, c, d):
        if t.children_left[node] == -1:
            out.append((a, b, c, d, int(np.argmax(t.value[node][0]))))
            return
        f, th = t.feature[node], t.threshold[node]
        if f == 0:
            _recurse(t.children_left[node],  a, th, c, d)
            _recurse(t.children_right[node], th, b, c, d)
        else:
            _recurse(t.children_left[node],  a, b, c, th)
            _recurse(t.children_right[node], a, b, th, d)

    _recurse(0, 0.0, 1.0, 0.0, 1.0)
    return out


_clf2    = DecisionTreeClassifier(max_depth=2, random_state=0).fit(_X, _y)
_clf_deep = DecisionTreeClassifier(
    max_depth=None, min_samples_leaf=1, random_state=0
).fit(_X_NOISE, _Y_NOISE)

RECTS_SIMPLE = _leaf_rects(_clf2)
RECTS_DEEP   = _leaf_rects(_clf_deep)


# ── Manim helpers ─────────────────────────────────────────────────────────────

def _region(ax, x0, x1, y0, y1, cls, alpha=0.20):
    color = C_C0 if cls == 0 else C_C1
    return Polygon(
        ax.c2p(x0, y0), ax.c2p(x1, y0),
        ax.c2p(x1, y1), ax.c2p(x0, y1),
        fill_color=color, fill_opacity=alpha, stroke_width=0,
    )


def _node_box(label, pos, fill=C_NODE, text_col=WHITE, w=1.90, h=0.58):
    box = RoundedRectangle(
        width=w, height=h, corner_radius=0.12,
        fill_color=fill, fill_opacity=0.95,
        stroke_color="#30363d", stroke_width=1.5,
    )
    txt = Text(label, font_size=12, color=text_col)
    return VGroup(box, txt).move_to(pos)


def _edge(p_pos, c_pos, label, side, gap=0.32):
    """Arrow-free edge between two node boxes, with a Yes/No side-label."""
    start = np.array(p_pos) + np.array([0, -gap, 0])
    end   = np.array(c_pos) + np.array([0,  gap, 0])
    line  = Line(start, end, color=C_DIM, stroke_width=1.5)
    lbl   = Text(label, font_size=11, color=C_DIM).next_to(
        line.get_center(), side, buff=0.10
    )
    return VGroup(line, lbl)


# ── Scene ─────────────────────────────────────────────────────────────────────

class DecisionTree(Scene):
    def construct(self):
        self.camera.background_color = C_BG

        # ════════════════════════════════════════════════════════════════════
        #  Phase 1 — Scatter + sequential splits
        # ════════════════════════════════════════════════════════════════════
        ax = Axes(
            x_range=[0, 1, 0.5], y_range=[0, 1, 0.5],
            x_length=6.2, y_length=6.2,
            **_AX,
        ).move_to([0.2, -0.3, 0])

        ax_lbl_x = Text("x\u2081", font_size=12, color=C_DIM).next_to(ax.x_axis, RIGHT, buff=0.12)
        ax_lbl_y = Text("x\u2082", font_size=12, color=C_DIM).next_to(ax.y_axis, UP,    buff=0.12)

        self.play(Create(ax), FadeIn(ax_lbl_x), FadeIn(ax_lbl_y), run_time=0.7)

        dots0 = VGroup(*[Dot(ax.c2p(float(xi), float(yi)), radius=0.09, color=C_C0) for xi, yi in _X0])
        dots1 = VGroup(*[Dot(ax.c2p(float(xi), float(yi)), radius=0.09, color=C_C1) for xi, yi in _X1])
        legend = VGroup(
            VGroup(Dot(radius=0.08, color=C_C0), Text("Class 0", font_size=12, color=C_C0)).arrange(RIGHT, buff=0.12),
            VGroup(Dot(radius=0.08, color=C_C1), Text("Class 1", font_size=12, color=C_C1)).arrange(RIGHT, buff=0.12),
        ).arrange(DOWN, buff=0.14, aligned_edge=LEFT).to_corner(UL, buff=0.35)

        title1 = (
            Text("Decision Tree: recursive axis-aligned splits",
                 font_size=16, color=WHITE, weight=BOLD)
            .to_edge(UP, buff=0.35)
        )
        self.play(
            LaggedStart(*[GrowFromCenter(d) for d in [*dots0, *dots1]], lag_ratio=0.04),
            run_time=1.2,
        )
        self.play(FadeIn(legend), FadeIn(title1), run_time=0.4)
        self.wait(0.4)

        # ── Split 1: horizontal line at y = _SPLIT_H ─────────────────────
        split1 = Line(
            ax.c2p(0, _SPLIT_H), ax.c2p(1, _SPLIT_H),
            color=C_SPLT, stroke_width=2.4,
        )
        split1_lbl = (
            Text(f"x\u2082 < {_SPLIT_H}", font_size=13, color=C_SPLT)
            .next_to(ax.c2p(0.5, _SPLIT_H), UP, buff=0.14)
        )
        shade_top = _region(ax, 0, 1, _SPLIT_H, 1.0, cls=1)
        self.play(Create(split1), FadeIn(split1_lbl), run_time=0.7)
        self.play(FadeIn(shade_top), run_time=0.5)
        self.wait(0.3)

        # ── Split 2: vertical line at x = _SPLIT_V (bottom half only) ────
        split2 = Line(
            ax.c2p(_SPLIT_V, 0), ax.c2p(_SPLIT_V, _SPLIT_H),
            color=C_SPLT, stroke_width=2.4,
        )
        split2_lbl = (
            Text(f"x\u2081 < {_SPLIT_V}", font_size=13, color=C_SPLT)
            .next_to(ax.c2p(_SPLIT_V, _SPLIT_H / 2), RIGHT, buff=0.14)
        )
        shade_bl = _region(ax, 0,        _SPLIT_V, 0, _SPLIT_H, cls=0)
        shade_br = _region(ax, _SPLIT_V, 1.0,      0, _SPLIT_H, cls=1)
        self.play(Create(split2), FadeIn(split2_lbl), run_time=0.7)
        self.play(FadeIn(shade_bl), FadeIn(shade_br), run_time=0.5)
        self.wait(0.3)

        # Region class labels
        rlbl_top = Text("Class 1", font_size=12, color=C_C1, weight=BOLD).move_to(ax.c2p(0.5,  0.72))
        rlbl_bl  = Text("Class 0", font_size=12, color=C_C0, weight=BOLD).move_to(ax.c2p(0.25, 0.22))
        rlbl_br  = Text("Class 1", font_size=12, color=C_C1, weight=BOLD).move_to(ax.c2p(0.75, 0.22))
        self.play(FadeIn(rlbl_top), FadeIn(rlbl_bl), FadeIn(rlbl_br), run_time=0.5)
        self.wait(0.8)

        # ════════════════════════════════════════════════════════════════════
        #  Phase 2 — Tree diagram alongside scatter
        # ════════════════════════════════════════════════════════════════════
        title2 = (
            Text("Tree structure mirrors the split sequence",
                 font_size=16, color=WHITE, weight=BOLD)
            .to_edge(UP, buff=0.35)
        )

        # Group everything from Phase 1 and slide left
        p1_group = VGroup(ax, ax_lbl_x, ax_lbl_y,
                          dots0, dots1,
                          split1, split1_lbl, split2, split2_lbl,
                          shade_top, shade_bl, shade_br,
                          rlbl_top, rlbl_bl, rlbl_br)
        self.play(
            p1_group.animate.scale(0.68).move_to([-3.4, -0.3, 0]),
            ReplacementTransform(title1, title2),
            FadeOut(legend),
            run_time=0.8,
        )

        # ── Tree node positions ───────────────────────────────────────────
        P_ROOT  = [3.4,  2.3, 0]
        P_LEFT  = [2.0,  0.6, 0]
        P_RIGHT = [4.8,  0.6, 0]
        P_LL    = [1.0, -1.1, 0]
        P_LR    = [3.0, -1.1, 0]

        node_root  = _node_box("x\u2082 < 0.45 ?", P_ROOT)
        node_left  = _node_box("x\u2081 < 0.50 ?", P_LEFT)
        node_right = _node_box("Class 1",          P_RIGHT, fill=C_C1, text_col=C_BG)
        node_ll    = _node_box("Class 0",          P_LL,    fill=C_C0, text_col=C_BG)
        node_lr    = _node_box("Class 1",          P_LR,    fill=C_C1, text_col=C_BG)

        edge_r_l  = _edge(P_ROOT, P_LEFT,  "Yes", LEFT)
        edge_r_r  = _edge(P_ROOT, P_RIGHT, "No",  RIGHT)
        edge_l_ll = _edge(P_LEFT, P_LL,    "Yes", LEFT)
        edge_l_lr = _edge(P_LEFT, P_LR,    "No",  RIGHT)

        self.play(FadeIn(node_root), run_time=0.4)
        self.play(FadeIn(edge_r_l), FadeIn(node_left),  run_time=0.4)
        self.play(FadeIn(edge_r_r), FadeIn(node_right), run_time=0.4)
        self.play(FadeIn(edge_l_ll), FadeIn(node_ll),   run_time=0.4)
        self.play(FadeIn(edge_l_lr), FadeIn(node_lr),   run_time=0.4)
        self.wait(1.2)

        # ════════════════════════════════════════════════════════════════════
        #  Phase 3 — Depth=2 vs deep tree (overfitting)
        # ════════════════════════════════════════════════════════════════════
        title3 = (
            Text("Deeper tree \u2192 more splits \u2192 overfits noise",
                 font_size=16, color=WHITE, weight=BOLD)
            .to_edge(UP, buff=0.35)
        )
        tree_group = VGroup(
            node_root, node_left, node_right, node_ll, node_lr,
            edge_r_l, edge_r_r, edge_l_ll, edge_l_lr,
        )
        self.play(
            FadeOut(p1_group), FadeOut(tree_group),
            ReplacementTransform(title2, title3),
            run_time=0.6,
        )

        # Two side-by-side axes
        def _side_ax(center):
            return Axes(
                x_range=[0, 1, 0.5], y_range=[0, 1, 0.5],
                x_length=4.5, y_length=4.5,
                **_AX,
            ).move_to(center)

        ax_l = _side_ax([-3.4, -0.5, 0])
        ax_r = _side_ax([ 3.4, -0.5, 0])

        lbl_l = Text("max_depth = 2", font_size=13, color=C_C1).next_to(ax_l, UP, buff=0.12)
        lbl_r = Text("max_depth = \u221e  (overfits)", font_size=13, color=C_C0).next_to(ax_r, UP, buff=0.12)
        self.play(
            Create(ax_l), Create(ax_r),
            FadeIn(lbl_l), FadeIn(lbl_r),
            run_time=0.6,
        )

        # Dots on both panels
        def _dots(ax_m):
            d0 = VGroup(*[Dot(ax_m.c2p(float(x), float(y_)), radius=0.07, color=C_C0) for x, y_ in _X0])
            d1 = VGroup(*[Dot(ax_m.c2p(float(x), float(y_)), radius=0.07, color=C_C1) for x, y_ in _X1])
            # noise points (smaller, with ring)
            dn = VGroup(*[
                Dot(ax_m.c2p(float(x), float(y_)), radius=0.07,
                    color=C_C1 if yn == 1 else C_C0)
                for (x, y_), yn in zip(_X_NOISE[len(_X):], _Y_NOISE[len(_X):])
            ])
            return VGroup(d0, d1, dn)

        dots_l = _dots(ax_l)
        dots_r = _dots(ax_r)
        self.play(
            LaggedStart(*[GrowFromCenter(d) for d in [*dots_l[0], *dots_l[1],
                                                       *dots_r[0], *dots_r[1],
                                                       *dots_l[2], *dots_r[2]]],
                        lag_ratio=0.03),
            run_time=1.0,
        )

        # Region rectangles
        rects_l = VGroup(*[_region(ax_l, *r[:4], r[4]) for r in RECTS_SIMPLE])
        rects_r = VGroup(*[_region(ax_r, *r[:4], r[4]) for r in RECTS_DEEP])
        self.play(FadeIn(rects_l), run_time=0.5)
        self.play(FadeIn(rects_r), run_time=0.5)
        self.wait(0.4)

        # Split lines for the simple tree (visual reference)
        sl1 = Line(ax_l.c2p(0, _SPLIT_H), ax_l.c2p(1, _SPLIT_H),
                   color=C_SPLT, stroke_width=1.8)
        sl2 = Line(ax_l.c2p(_SPLIT_V, 0), ax_l.c2p(_SPLIT_V, _SPLIT_H),
                   color=C_SPLT, stroke_width=1.8)
        self.play(Create(sl1), Create(sl2), run_time=0.5)

        caption = (
            Text("deep tree memorises noise — pruning or max_depth limits variance",
                 font_size=12, color=C_DIM)
            .to_edge(DOWN, buff=0.35)
        )
        self.play(FadeIn(caption, shift=UP * 0.1), run_time=0.5)
        self.wait(2.5)
