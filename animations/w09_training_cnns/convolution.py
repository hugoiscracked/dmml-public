"""
W09 — Convolution Operation

  Phase 1: A 3x3 filter slides over a 6x6 input grid. At each position the
           overlapping patch flashes, the dot-product sum is shown, and the
           result is written into the output feature map.
  Phase 2: Three filters (horizontal edge, vertical edge, identity) each
           produce a different 4x4 feature map — illustrating depth.
  Phase 3: 2x2 max-pooling on the feature map: each window lights up, the
           max value is highlighted, and the downsampled 2x2 result appears.

Render:
  ../../env/bin/manim -pql convolution.py Convolution
  ../../env/bin/manim -pqh convolution.py Convolution
"""

from manim import *
import numpy as np

# ── Palette ───────────────────────────────────────────────────────────────────
C_BG     = "#0d1117"
C_WHITE  = "#ffffff"
C_DIM    = "#8b949e"
C_BLUE   = "#58a6ff"   # input grid
C_AMBER  = "#ffa657"   # filter kernel / highlight
C_GREEN  = "#3fb950"   # output feature map
C_GOLD   = "#e3b341"   # max-pool winner
C_TEAL   = "#2dd4bf"   # pooled output
C_PURPLE = "#bc8cff"   # vertical edge filter

CELL = 0.56            # grid cell side length
HALF = CELL / 2.0

# ── Data ──────────────────────────────────────────────────────────────────────
_INPUT = np.array([
    [1, 2, 0, 1, 3, 1],
    [0, 3, 1, 2, 0, 2],
    [2, 1, 3, 0, 1, 0],
    [1, 0, 2, 1, 2, 3],
    [0, 2, 1, 3, 0, 1],
    [3, 1, 0, 2, 1, 0],
], dtype=float)

_FILTER_H = np.array([[ 1,  1,  1],
                      [ 0,  0,  0],
                      [-1, -1, -1]], dtype=float)   # horizontal edge

_FILTER_V = np.array([[ 1,  0, -1],
                      [ 1,  0, -1],
                      [ 1,  0, -1]], dtype=float)   # vertical edge

_FILTER_I = np.array([[ 0,  0,  0],
                      [ 0,  1,  0],
                      [ 0,  0,  0]], dtype=float)   # identity (pass-through)


def _convolve(inp, filt):
    K = filt.shape[0]
    out = np.zeros((inp.shape[0] - K + 1, inp.shape[1] - K + 1))
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            out[r, c] = (inp[r:r+K, c:c+K] * filt).sum()
    return out


_OUTPUT_H = _convolve(_INPUT, _FILTER_H)   # 4x4
_OUTPUT_V = _convolve(_INPUT, _FILTER_V)
_OUTPUT_I = _convolve(_INPUT, _FILTER_I)


# ── Grid helpers ──────────────────────────────────────────────────────────────

def _make_grid(n_rows, n_cols, origin, color,
               values=None, font_size=12, fill_op=0.10, cell_size=None):
    """
    Build a VGroup of Square cells and a VGroup of Text labels.
    origin = top-left corner (np.array [x, y, 0]).
    """
    cs = cell_size or CELL
    hs = cs / 2.0
    cells, labels = VGroup(), VGroup()
    for r in range(n_rows):
        for c in range(n_cols):
            x = origin[0] + c * cs + hs
            y = origin[1] - r * cs - hs
            sq = (
                Square(side_length=cs, color=color,
                       fill_color=color, fill_opacity=fill_op,
                       stroke_width=0.9)
                .move_to([x, y, 0])
            )
            cells.add(sq)
            if values is not None:
                v = values[r, c]
                txt = Text(
                    f"{int(v)}" if v == int(v) else f"{v:.0f}",
                    font_size=font_size, color=color,
                ).move_to([x, y, 0])
                labels.add(txt)
    return cells, labels


def _flat(r, c, n_cols):
    return r * n_cols + c


class Convolution(Scene):

    def construct(self):
        self.camera.background_color = C_BG
        self._phase1()
        self._phase2()
        self._phase3()

    # ── Phase 1: sliding filter ────────────────────────────────────────────────

    def _phase1(self):
        title = (
            Text("Convolution — Sliding Filter",
                 font_size=22, color=C_WHITE, weight=BOLD)
            .to_edge(UP, buff=0.28)
        )
        sub = (
            Text("3x3 filter  ·  6x6 input  ·  valid convolution  →  4x4 output",
                 font_size=13, color=C_DIM)
            .next_to(title, DOWN, buff=0.12)
        )
        self.play(FadeIn(title), FadeIn(sub), run_time=0.5)

        # Layout origins (top-left corner of each grid)
        in_origin  = np.array([-5.5,  1.68, 0.0])   # 6x6 input  (left)
        flt_origin = np.array([ 2.10,  2.10, 0.0])  # 3x3 filter (upper right)
        out_origin = np.array([ 1.70, -0.20, 0.0])  # 4x4 output (lower right)

        # ── Input grid ────────────────────────────────────────────────────────
        in_cells, in_vals = _make_grid(6, 6, in_origin, C_BLUE, _INPUT,
                                       font_size=11, fill_op=0.10)
        self.play(
            LaggedStart(*[FadeIn(c) for c in in_cells], lag_ratio=0.02),
            run_time=0.7,
        )
        self.play(
            LaggedStart(*[FadeIn(v) for v in in_vals], lag_ratio=0.02),
            run_time=0.4,
        )
        in_lbl = (
            Text("Input  6x6", font_size=12, color=C_BLUE)
            .next_to(in_cells, DOWN, buff=0.15)
        )
        self.play(FadeIn(in_lbl), run_time=0.25)

        # ── Filter display (static) ────────────────────────────────────────────
        flt_cells, flt_vals = _make_grid(3, 3, flt_origin, C_AMBER,
                                         _FILTER_H, font_size=11, fill_op=0.22)
        flt_lbl = (
            Text("Filter  3x3", font_size=12, color=C_AMBER)
            .next_to(flt_cells, UP, buff=0.12)
        )
        self.play(
            LaggedStart(*[GrowFromCenter(c) for c in flt_cells], lag_ratio=0.07),
            run_time=0.5,
        )
        self.play(FadeIn(flt_vals), FadeIn(flt_lbl), run_time=0.25)

        # ── Output grid (empty) ───────────────────────────────────────────────
        out_cells, _ = _make_grid(4, 4, out_origin, C_GREEN, fill_op=0.05)
        out_lbl = (
            Text("Output  4x4", font_size=12, color=C_GREEN)
            .next_to(out_cells, DOWN, buff=0.15)
        )
        self.play(
            LaggedStart(*[FadeIn(c) for c in out_cells], lag_ratio=0.03),
            FadeIn(out_lbl),
            run_time=0.4,
        )

        # Arrow: filter → output
        arr = Arrow(
            flt_cells.get_bottom() + DOWN * 0.06,
            out_cells.get_top()    + UP   * 0.06,
            buff=0.04, color=C_DIM, stroke_width=1.4,
            max_tip_length_to_length_ratio=0.20,
        )
        self.play(GrowArrow(arr), run_time=0.25)
        self.wait(0.35)

        # ── Sliding animation helpers ─────────────────────────────────────────

        def _patch_center(pr, pc):
            """Centre of the 3x3 patch at output position (pr, pc)."""
            x = in_origin[0] + pc * CELL + 1.5 * CELL
            y = in_origin[1] - pr * CELL - 1.5 * CELL
            return np.array([x, y, 0.0])

        def _out_center(pr, pc):
            x = out_origin[0] + pc * CELL + HALF
            y = out_origin[1] - pr * CELL - HALF
            return np.array([x, y, 0.0])

        # Reusable highlight rectangle
        highlight = (
            Square(side_length=CELL * 3, color=C_AMBER,
                   fill_color=C_AMBER, fill_opacity=0.15, stroke_width=2.2)
            .move_to(_patch_center(0, 0))
        )
        self.play(FadeIn(highlight), run_time=0.25)

        out_val_mobs = VGroup()

        def _show_slow(pr, pc):
            """Animate one convolution step in detail."""
            self.play(
                highlight.animate.move_to(_patch_center(pr, pc)),
                run_time=0.30,
            )
            # Flash the patch cells
            idxs = [_flat(pr + kr, pc + kc, 6)
                    for kr in range(3) for kc in range(3)]
            self.play(
                *[in_cells[i].animate.set_fill(C_AMBER, opacity=0.40)
                  for i in idxs],
                run_time=0.20,
            )
            # Compute and display result
            val = int((_INPUT[pr:pr+3, pc:pc+3] * _FILTER_H).sum())
            result_txt = (
                Text(f"patch · filter  =  {val}",
                     font_size=11, color=C_GOLD)
                .to_edge(DOWN, buff=0.50)
            )
            self.play(FadeIn(result_txt), run_time=0.20)

            # Write to output cell
            out_cells[_flat(pr, pc, 4)].set_fill(C_GREEN, opacity=0.22)
            val_mob = (
                Text(f"{val}", font_size=11, color=C_GREEN, weight=BOLD)
                .move_to(_out_center(pr, pc))
            )
            self.play(FadeIn(val_mob), run_time=0.22)
            out_val_mobs.add(val_mob)

            # Reset
            self.play(
                *[in_cells[i].animate.set_fill(C_BLUE, opacity=0.10)
                  for i in idxs],
                FadeOut(result_txt),
                run_time=0.18,
            )

        def _show_fast(pr, pc):
            """Animate one convolution step quickly (no formula)."""
            self.play(
                highlight.animate.move_to(_patch_center(pr, pc)),
                run_time=0.10,
            )
            val = int((_INPUT[pr:pr+3, pc:pc+3] * _FILTER_H).sum())
            out_cells[_flat(pr, pc, 4)].set_fill(C_GREEN, opacity=0.22)
            val_mob = (
                Text(f"{val}", font_size=11, color=C_GREEN, weight=BOLD)
                .move_to(_out_center(pr, pc))
            )
            self.play(FadeIn(val_mob), run_time=0.10)
            out_val_mobs.add(val_mob)

        # First row — slow and detailed
        for pc in range(4):
            _show_slow(0, pc)

        self.wait(0.15)
        sub2 = (
            Text("filling remaining positions…", font_size=13, color=C_DIM)
            .next_to(title, DOWN, buff=0.12)
        )
        self.play(ReplacementTransform(sub, sub2), run_time=0.25)

        # Remaining rows — fast
        for pr in range(1, 4):
            for pc in range(4):
                _show_fast(pr, pc)

        self.play(FadeOut(highlight), run_time=0.25)

        note = (
            Text(
                "output size  =  (6 - 3 + 1) = 4   ·   "
                "each output cell = one dot product of the filter with a local patch",
                font_size=11, color=C_DIM,
            )
            .to_edge(DOWN, buff=0.32)
        )
        self.play(FadeIn(note), run_time=0.35)
        self.wait(1.8)

        self._title  = title
        self._p1_all = VGroup(
            in_cells, in_vals, in_lbl,
            flt_cells, flt_vals, flt_lbl,
            out_cells, out_val_mobs, out_lbl,
            arr, note, sub2,
        )

    # ── Phase 2: multiple filters = depth ─────────────────────────────────────

    def _phase2(self):
        self.play(FadeOut(self._p1_all), run_time=0.45)

        sub = (
            Text("multiple filters  →  multiple feature maps  →  depth dimension",
                 font_size=13, color=C_DIM)
            .next_to(self._title, DOWN, buff=0.12)
        )
        self.play(FadeIn(sub), run_time=0.28)

        specs = [
            ("Horizontal\nEdge",  _FILTER_H, _OUTPUT_H, C_AMBER),
            ("Vertical\nEdge",    _FILTER_V, _OUTPUT_V, C_PURPLE),
            ("Identity\n(copy)",  _FILTER_I, _OUTPUT_I, C_TEAL),
        ]
        col_xs   = [-4.4, 0.0, 4.4]
        flt_top  = 1.90
        out_top  = flt_top - 3 * CELL - 0.65

        all_mobs = VGroup()
        arrows   = VGroup()

        for idx, (name, flt, out_arr, col) in enumerate(specs):
            cx = col_xs[idx]
            # Filter grid — centred on cx
            flt_org = np.array([cx - 1.5 * CELL, flt_top, 0.0])
            fc, fv  = _make_grid(3, 3, flt_org, col, flt,
                                 font_size=10, fill_op=0.22)

            # Output grid — centred on cx
            out_org = np.array([cx - 2.0 * CELL, out_top, 0.0])
            oc, ov  = _make_grid(4, 4, out_org, col, out_arr,
                                 font_size=9, fill_op=0.18)

            name_lbl = (
                Text(name, font_size=11, color=col, weight=BOLD)
                .next_to(fc, UP, buff=0.12)
            )
            map_lbl = (
                Text("4x4 map", font_size=10, color=col)
                .next_to(oc, DOWN, buff=0.10)
            )

            self.play(
                LaggedStart(*[GrowFromCenter(c) for c in fc], lag_ratio=0.06),
                run_time=0.45,
            )
            self.play(FadeIn(fv), FadeIn(name_lbl), run_time=0.22)

            ar = Arrow(
                fc.get_bottom() + DOWN * 0.05,
                oc.get_top()    + UP   * 0.05,
                buff=0.04, color=col, stroke_width=1.3,
                max_tip_length_to_length_ratio=0.22,
            )
            self.play(GrowArrow(ar), run_time=0.22)
            arrows.add(ar)

            self.play(
                LaggedStart(*[FadeIn(c) for c in oc], lag_ratio=0.03),
                run_time=0.35,
            )
            self.play(FadeIn(ov), FadeIn(map_lbl), run_time=0.18)

            all_mobs.add(fc, fv, oc, ov, name_lbl, map_lbl)

        depth_note = (
            Text(
                "N filters  →  output tensor  4 x 4 x N   (N = number of channels / depth)",
                font_size=12, color=C_GOLD,
            )
            .to_edge(DOWN, buff=0.32)
        )
        self.play(FadeIn(depth_note), run_time=0.35)
        self.wait(2.0)

        self._p2_all = VGroup(all_mobs, arrows, sub, depth_note)

    # ── Phase 3: max pooling ───────────────────────────────────────────────────

    def _phase3(self):
        self.play(FadeOut(self._p2_all), run_time=0.45)

        sub = (
            Text(
                "Max Pooling  ·  2x2 window, stride 2  →  spatial compression",
                font_size=13, color=C_DIM,
            )
            .next_to(self._title, DOWN, buff=0.12)
        )
        self.play(FadeIn(sub), run_time=0.28)

        # Use absolute values of _OUTPUT_H as the pooling input (all positive)
        pool_in = np.abs(_OUTPUT_H).astype(int)

        # 2x2 max pool → 2x2 output
        pool_out = np.array([
            [pool_in[0:2, 0:2].max(), pool_in[0:2, 2:4].max()],
            [pool_in[2:4, 0:2].max(), pool_in[2:4, 2:4].max()],
        ])

        # Layout
        in_origin = np.array([-4.2,  1.12, 0.0])  # 4x4 input
        POOL_CS   = 0.95                            # larger cells for output
        out_cx    = 2.8                             # output grid x-centre
        out_origin = np.array([out_cx - POOL_CS, POOL_CS, 0.0])  # 2x2

        # Input grid
        in_cells, in_vals = _make_grid(4, 4, in_origin, C_GREEN, pool_in,
                                       font_size=13, fill_op=0.12)
        in_lbl = (
            Text("Feature map  4x4", font_size=12, color=C_GREEN)
            .next_to(in_cells, DOWN, buff=0.15)
        )
        self.play(
            LaggedStart(*[FadeIn(c) for c in in_cells], lag_ratio=0.04),
            run_time=0.55,
        )
        self.play(FadeIn(in_vals), FadeIn(in_lbl), run_time=0.28)

        # Output pool grid (manual — different cell size)
        out_sq, out_ctr = [], []
        for r in range(2):
            for c in range(2):
                x = out_origin[0] + c * POOL_CS + POOL_CS / 2
                y = out_origin[1] - r * POOL_CS - POOL_CS / 2
                sq = (
                    Square(side_length=POOL_CS, color=C_TEAL,
                           fill_color=C_TEAL, fill_opacity=0.08,
                           stroke_width=1.2)
                    .move_to([x, y, 0])
                )
                out_sq.append(sq)
                out_ctr.append(np.array([x, y, 0.0]))

        out_vg  = VGroup(*out_sq)
        out_lbl = (
            Text("Pooled  2x2", font_size=12, color=C_TEAL)
            .next_to(out_vg, DOWN, buff=0.15)
        )
        self.play(
            LaggedStart(*[FadeIn(s) for s in out_sq], lag_ratio=0.12),
            FadeIn(out_lbl),
            run_time=0.45,
        )

        # Arrow input → output
        pool_arr = Arrow(
            in_cells.get_right() + RIGHT * 0.10,
            out_vg.get_left()    + LEFT  * 0.10,
            buff=0.04, color=C_DIM, stroke_width=1.4,
            max_tip_length_to_length_ratio=0.18,
        )
        self.play(GrowArrow(pool_arr), run_time=0.25)

        # 2x2 window label
        win_lbl = (
            Text("2x2 window", font_size=11, color=C_TEAL)
            .to_edge(UP, buff=3.2)
        )

        # Per-window animation
        windows = [
            (slice(0, 2), slice(0, 2), 0),
            (slice(0, 2), slice(2, 4), 1),
            (slice(2, 4), slice(0, 2), 2),
            (slice(2, 4), slice(2, 4), 3),
        ]

        out_val_mobs = VGroup()
        for rs, cs, wi in windows:
            # Indices in flat in_cells
            idxs = [_flat(r, c, 4)
                    for r in range(rs.start, rs.stop)
                    for c in range(cs.start, cs.stop)]

            # Highlight window
            self.play(
                *[in_cells[i].animate.set_fill(C_TEAL, opacity=0.38)
                  for i in idxs],
                run_time=0.22,
            )

            # Find and gold-highlight the max cell
            patch = pool_in[rs, cs]
            max_val = patch.max()
            flat_local = int(np.argmax(patch.flatten()))
            max_r = rs.start + flat_local // 2
            max_c = cs.start + flat_local % 2
            max_idx = _flat(max_r, max_c, 4)

            self.play(
                in_cells[max_idx].animate.set_fill(C_GOLD, opacity=0.55),
                run_time=0.20,
            )

            # Write max to output cell
            out_sq[wi].set_fill(C_TEAL, opacity=0.28)
            val_mob = (
                Text(f"{int(max_val)}", font_size=18, color=C_TEAL, weight=BOLD)
                .move_to(out_ctr[wi])
            )
            self.play(FadeIn(val_mob), run_time=0.22)
            out_val_mobs.add(val_mob)

            # Reset input colours
            self.play(
                *[in_cells[i].animate.set_fill(C_GREEN, opacity=0.12)
                  for i in idxs],
                run_time=0.16,
            )

        caption = (
            Text(
                "pooling halves spatial size  ·  "
                "keeps strongest activations  ·  "
                "adds translation invariance",
                font_size=12, color=C_DIM,
            )
            .to_edge(DOWN, buff=0.32)
        )
        self.play(FadeIn(caption), run_time=0.35)
        self.wait(2.5)
