"""
W08 — MLP Forward Pass

  Phase 1: Build the network graph layer by layer (3 → 4 → 2).
  Phase 2: Animate a forward pass — input values propagate through the hidden
           layer (ReLU applied; dead neurons shown in dark grey) then to the
           output layer.
  Phase 3: Softmax converts logits to probabilities; winning class highlighted
           → "Predicted: Class 1".

Render:
  ../../env/bin/manim -pql mlp_forward.py MLPForward
  ../../env/bin/manim -pqh mlp_forward.py MLPForward
"""

from manim import *
import numpy as np

# ── Palette ───────────────────────────────────────────────────────────────────
C_BG     = "#0d1117"
C_WHITE  = "#ffffff"
C_DIM    = "#8b949e"

C_IN     = "#58a6ff"   # blue   — input nodes
C_HID    = "#ffa657"   # amber  — active hidden nodes
C_DEAD   = "#3d444d"   # dark grey — ReLU-killed neurons
C_OUT    = "#3fb950"   # green  — output nodes
C_EDGE   = "#30363d"   # dim edge (default)
C_EDGE_H = "#e3b341"   # gold   — highlighted edge during forward pass

# ── Network weights (hand-crafted for visual clarity) ─────────────────────────
#   Input x = [0.8, 0.2, 0.6]
#   Hidden pre-activations z1: [+0.68, -0.28, -0.02, +0.24]
#   After ReLU h:             [ 0.68,  0.00,  0.00,  0.24]  ← h2, h3 dead
#   Output logits z2:         [-0.196, +0.528]
#   Softmax p:                [ 0.327,  0.673]              ← Class 1 wins

_X  = np.array([0.8, 0.2, 0.6])

_W1 = np.array([
    [ 0.5, -0.3,  0.4],
    [ 0.2,  0.8, -0.5],
    [-0.6,  0.4,  0.3],
    [ 0.3, -0.1,  0.7],
])
_b1 = np.array([ 0.1, -0.3,  0.2, -0.4])

_W2 = np.array([
    [-0.5,  0.4, -0.3,  0.6],
    [ 0.6, -0.4,  0.3,  0.5],
])
_b2 = np.array([0.0, 0.0])

# Compute once at import time
_z1  = _W1 @ _X + _b1
_h   = np.maximum(0.0, _z1)
_z2  = _W2 @ _h + _b2
_exp = np.exp(_z2 - _z2.max())
_p   = _exp / _exp.sum()

# ── Layout ────────────────────────────────────────────────────────────────────
LX = {"inp": -4.0, "hid": 0.0, "out": 4.0}
LY = {
    "inp": [ 1.2,  0.0, -1.2],
    "hid": [ 1.8,  0.6, -0.6, -1.8],
    "out": [ 0.6, -0.6],
}
NODE_R = 0.32


def _pos(layer, idx):
    return np.array([LX[layer], LY[layer][idx], 0.0])


class MLPForward(Scene):

    def construct(self):
        self.camera.background_color = C_BG
        self._phase1()
        self._phase2()
        self._phase3()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _make_node(self, layer, idx, color=None):
        col = color or {"inp": C_IN, "hid": C_DIM, "out": C_OUT}[layer]
        return (
            Circle(radius=NODE_R, color=col, fill_opacity=0.20, stroke_width=2.2)
            .move_to(_pos(layer, idx))
        )

    def _make_edge(self, layer_a, i, layer_b, j):
        p1 = _pos(layer_a, i) + np.array([NODE_R, 0, 0])
        p2 = _pos(layer_b, j) - np.array([NODE_R, 0, 0])
        return Line(p1, p2, stroke_color=C_EDGE, stroke_width=0.9)

    # ── Phase 1: build the architecture graph ─────────────────────────────────

    def _phase1(self):
        title = (
            Text("Neural Network — Forward Pass",
                 font_size=20, color=C_WHITE, weight=BOLD)
            .to_edge(UP, buff=0.30)
        )
        sub = (
            Text("architecture  3 → 4 → 2   ·   hidden activation: ReLU",
                 font_size=12, color=C_DIM)
            .next_to(title, DOWN, buff=0.12)
        )
        self.play(FadeIn(title), FadeIn(sub), run_time=0.5)

        # Nodes
        in_nodes  = VGroup(*[self._make_node("inp", i) for i in range(3)])
        hid_nodes = VGroup(*[self._make_node("hid", i) for i in range(4)])
        out_nodes = VGroup(*[self._make_node("out", i) for i in range(2)])

        # Node name labels
        in_lbls = VGroup(*[
            Text(f"x{i+1}", font_size=13, color=C_IN)
            .next_to(in_nodes[i], LEFT, buff=0.15)
            for i in range(3)
        ])
        hid_lbls = VGroup(*[
            Text(f"h{i+1}", font_size=11, color=C_DIM)
            .next_to(hid_nodes[i], UP, buff=0.08)
            for i in range(4)
        ])
        out_lbls = VGroup(*[
            Text(f"y{i+1}", font_size=13, color=C_OUT)
            .next_to(out_nodes[i], RIGHT, buff=0.15)
            for i in range(2)
        ])

        # Layer labels (below bottom node of each layer)
        lbl_inp = (
            Text("Input\n(3)", font_size=11, color=C_IN)
            .next_to(in_nodes[2], DOWN, buff=0.35)
        )
        lbl_hid = (
            Text("Hidden (ReLU)\n(4)", font_size=11, color=C_DIM)
            .next_to(hid_nodes[3], DOWN, buff=0.35)
        )
        lbl_out = (
            Text("Output\n(2)", font_size=11, color=C_OUT)
            .next_to(out_nodes[1], DOWN, buff=0.35)
        )

        # Edges  (order matters for indexing in phase 2)
        # edges_ih[i*4 + j]  =  inp_i → hid_j
        edges_ih = VGroup(*[
            self._make_edge("inp", i, "hid", j)
            for i in range(3) for j in range(4)
        ])
        # edges_ho[j*2 + k]  =  hid_j → out_k
        edges_ho = VGroup(*[
            self._make_edge("hid", j, "out", k)
            for j in range(4) for k in range(2)
        ])

        # ── animate ───────────────────────────────────────────────────────────

        # Input layer
        self.play(
            LaggedStart(*[GrowFromCenter(n) for n in in_nodes], lag_ratio=0.15),
            run_time=0.6,
        )
        self.play(FadeIn(in_lbls), FadeIn(lbl_inp), run_time=0.3)

        # inp → hid edges
        self.play(
            LaggedStart(*[Create(e) for e in edges_ih], lag_ratio=0.04),
            run_time=0.8,
        )

        # Hidden layer
        self.play(
            LaggedStart(*[GrowFromCenter(n) for n in hid_nodes], lag_ratio=0.15),
            run_time=0.6,
        )
        self.play(FadeIn(hid_lbls), FadeIn(lbl_hid), run_time=0.3)

        # hid → out edges
        self.play(
            LaggedStart(*[Create(e) for e in edges_ho], lag_ratio=0.06),
            run_time=0.5,
        )

        # Output layer
        self.play(
            LaggedStart(*[GrowFromCenter(n) for n in out_nodes], lag_ratio=0.2),
            run_time=0.5,
        )
        self.play(FadeIn(out_lbls), FadeIn(lbl_out), run_time=0.3)
        self.wait(0.8)

        # Stash for downstream phases
        self._title     = title
        self._sub       = sub
        self._in_nodes  = in_nodes
        self._hid_nodes = hid_nodes
        self._out_nodes = out_nodes
        self._in_lbls   = in_lbls
        self._hid_lbls  = hid_lbls
        self._out_lbls  = out_lbls
        self._lbl_inp   = lbl_inp
        self._lbl_hid   = lbl_hid
        self._lbl_out   = lbl_out
        self._edges_ih  = edges_ih
        self._edges_ho  = edges_ho

    # ── Phase 2: forward pass ─────────────────────────────────────────────────

    def _phase2(self):
        # Step 1 — load inputs
        sub2 = (
            Text("step 1 — load input values",
                 font_size=12, color=C_DIM)
            .next_to(self._title, DOWN, buff=0.12)
        )
        self.play(ReplacementTransform(self._sub, sub2), run_time=0.3)

        in_vals = VGroup(*[
            Text(f"{v:.1f}", font_size=13, color=C_WHITE, weight=BOLD)
            .move_to(self._in_nodes[i])
            for i, v in enumerate(_X)
        ])
        self.play(
            LaggedStart(*[FadeIn(v) for v in in_vals], lag_ratio=0.2),
            run_time=0.5,
        )
        self.wait(0.3)

        # Step 2 — hidden layer
        sub3 = (
            Text("step 2 — hidden layer:  z = W·x + b,   then h = ReLU(z)",
                 font_size=12, color=C_DIM)
            .next_to(self._title, DOWN, buff=0.12)
        )
        self.play(ReplacementTransform(sub2, sub3), run_time=0.3)
        self.wait(0.2)

        post_val_mobs = []
        for j in range(4):
            # Highlight the three incoming edges for hidden unit j
            edges_to_j = [self._edges_ih[i * 4 + j] for i in range(3)]
            self.play(
                *[e.animate.set_stroke(color=C_EDGE_H, width=2.0)
                  for e in edges_to_j],
                run_time=0.30,
            )

            # Show pre-activation z
            z_val = _z1[j]
            pre_lbl = (
                Text(f"z={z_val:+.2f}", font_size=11, color=C_EDGE_H)
                .next_to(self._hid_nodes[j], RIGHT, buff=0.10)
            )
            self.play(FadeIn(pre_lbl), run_time=0.20)

            # ReLU: active vs dead
            if _h[j] > 0:
                self.play(
                    self._hid_nodes[j].animate
                        .set_color(C_HID)
                        .set_fill(C_HID, opacity=0.35),
                    run_time=0.28,
                )
                post_lbl = (
                    Text(f"{_h[j]:.2f}", font_size=12, color=C_WHITE, weight=BOLD)
                    .move_to(self._hid_nodes[j])
                )
            else:
                self.play(
                    self._hid_nodes[j].animate
                        .set_color(C_DEAD)
                        .set_fill(C_DEAD, opacity=0.60),
                    run_time=0.28,
                )
                post_lbl = (
                    Text("0", font_size=12, color=C_DIM)
                    .move_to(self._hid_nodes[j])
                )

            self.play(FadeOut(pre_lbl), FadeIn(post_lbl), run_time=0.22)
            post_val_mobs.append(post_lbl)

            # Dim edges back
            self.play(
                *[e.animate.set_stroke(color=C_EDGE, width=0.9)
                  for e in edges_to_j],
                run_time=0.18,
            )

        relu_note = (
            Text("h2 and h3 had negative pre-activations — ReLU clamps them to 0  ('dead' neurons)",
                 font_size=11, color=C_DIM)
            .to_edge(DOWN, buff=0.42)
        )
        self.play(FadeIn(relu_note), run_time=0.3)
        self.wait(0.8)
        self.play(FadeOut(relu_note), run_time=0.2)

        # Step 3 — output layer
        sub4 = (
            Text("step 3 — output layer:  z = W·h + b",
                 font_size=12, color=C_DIM)
            .next_to(self._title, DOWN, buff=0.12)
        )
        self.play(ReplacementTransform(sub3, sub4), run_time=0.3)

        out_val_mobs = []
        for k in range(2):
            edges_to_k = [self._edges_ho[j * 2 + k] for j in range(4)]
            self.play(
                *[e.animate.set_stroke(color=C_EDGE_H, width=2.0)
                  for e in edges_to_k],
                run_time=0.30,
            )
            z_val = _z2[k]
            out_lbl = (
                Text(f"{z_val:+.2f}", font_size=12, color=C_WHITE, weight=BOLD)
                .move_to(self._out_nodes[k])
            )
            self.play(FadeIn(out_lbl), run_time=0.28)
            out_val_mobs.append(out_lbl)
            self.play(
                *[e.animate.set_stroke(color=C_EDGE, width=0.9)
                  for e in edges_to_k],
                run_time=0.18,
            )

        self.wait(0.4)
        self._sub4         = sub4
        self._out_val_mobs = out_val_mobs

    # ── Phase 3: softmax + prediction ────────────────────────────────────────

    def _phase3(self):
        sub5 = (
            Text("step 4 — softmax → probabilities  ·  argmax → prediction",
                 font_size=12, color=C_DIM)
            .next_to(self._title, DOWN, buff=0.12)
        )
        self.play(ReplacementTransform(self._sub4, sub5), run_time=0.3)

        # Replace logits with softmax probabilities
        p_mobs = []
        for k in range(2):
            p_lbl = (
                Text(f"{_p[k]:.2f}", font_size=13, color=C_WHITE, weight=BOLD)
                .move_to(self._out_nodes[k])
            )
            self.play(
                ReplacementTransform(self._out_val_mobs[k], p_lbl),
                run_time=0.40,
            )
            p_mobs.append(p_lbl)

        self.wait(0.3)

        # Highlight winning class
        winner = int(np.argmax(_p))
        self.play(
            self._out_nodes[winner].animate
                .set_color(C_OUT)
                .set_fill(C_OUT, opacity=0.45)
                .scale(1.28),
            run_time=0.5,
        )

        caption = (
            Text(
                f"Predicted: Class {winner}   (confidence {_p[winner]:.0%})\n"
                "one forward pass = one prediction  —  learning adjusts W and b to minimise loss",
                font_size=12, color=C_OUT, weight=BOLD,
            )
            .to_edge(DOWN, buff=0.40)
        )
        self.play(FadeIn(caption), run_time=0.5)
        self.wait(2.5)
