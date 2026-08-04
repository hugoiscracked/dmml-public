"""
W01 — Data Cleaning

  Phase 1: Missing values.
           An Age column with NaN cells is shown alongside three
           strategies side by side: drop rows, mean imputation,
           and forward-fill — colour-coded for instant comparison.

  Phase 2: Encoding.
           A categorical Sex column is one-hot encoded into two binary
           dummy columns; the dummy-variable trap is noted.

  Phase 3: Normalisation.
           Original Age values are mapped by min-max scaling (→ [0, 1])
           and standardisation (→ mean=0, σ=1) on three parallel number
           lines, showing that shape is preserved while scale changes.

Text uses LaTeX (Tex/MathTex). Every label is auto-fit to its box, so the
LaTeX metrics cannot overflow cells or collide with neighbours.

Render:
  ../../env/bin/manim -pql data_cleaning.py DataCleaning
  ../../env/bin/manim -qk  data_cleaning.py DataCleaning   # 4K master
"""

from manim import *
import numpy as np

# ── Palette ────────────────────────────────────────────────────────────────────
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

# ── Readability-tuned type sizes (upper bounds; fit-to-box shrinks as needed) ───
FS_TITLE = 52
FS_SUB   = 34
FS_STRAT = 32
FS_HEAD  = 32
FS_CELL  = 34
FS_ANN   = 26
FS_CAP   = 28
FS_TICK  = 24
FS_LBL   = 34
FS_FORM  = 28
FS_NOTE  = 28


# ── LaTeX helpers ──────────────────────────────────────────────────────────────
def _tx(s, fs, color=C_DIM, bold=False, math=False, w=None, h=None):
    """A Tex/MathTex mobject, optionally shrunk to fit a width/height box."""
    if math:
        m = MathTex(s, font_size=fs, color=color)
    elif bold:
        m = Tex(r"\textbf{%s}" % s, font_size=fs, color=color)
    else:
        m = Tex(s, font_size=fs, color=color)
    if w is not None and m.width > w:
        m.scale_to_fit_width(w)
    if h is not None and m.height > h:
        m.scale_to_fit_height(h)
    return m


def _stack(lines, fs, colors, cx, cy, math=False, buff=0.14, w=None):
    """Vertically stacked Tex lines centred at (cx, cy)."""
    grp = VGroup()
    for i, ln in enumerate(lines):
        col = colors[i] if isinstance(colors, list) else colors
        grp.add(_tx(ln, fs, color=col, math=math, w=w))
    grp.arrange(DOWN, buff=buff)
    grp.move_to([cx, cy, 0])
    return grp


def _cell(s, cx, cy, w, h, bg=None, tc=C_DIM, fs=FS_CELL, bold=False, math=False):
    rect = Rectangle(
        width=w, height=h,
        fill_color=bg or C_BG, fill_opacity=0.22 if bg else 0.0,
        stroke_color=C_DEAD, stroke_width=1.2,
    ).move_to([cx, cy, 0])
    txt = _tx(s, fs, color=tc, bold=bold, math=math,
              w=w * 0.84, h=h * 0.62).move_to([cx, cy, 0])
    return VGroup(rect, txt)


class DataCleaning(Scene):

    def construct(self):
        self.camera.background_color = C_BG
        self._phase1()
        self._phase2()
        self._phase3()

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 1 — Missing values
    # ══════════════════════════════════════════════════════════════════════════

    def _phase1(self):
        title = _tx("Data Cleaning", FS_TITLE, color=C_WHITE, bold=True).to_edge(UP, buff=0.28)
        sub = _tx("Missing values --- three strategies compared side by side",
                  FS_SUB, color=C_DIM, w=12.6).next_to(title, DOWN, buff=0.16)
        self.play(FadeIn(title), FadeIn(sub), run_time=0.5)

        # ── Layout ────────────────────────────────────────────────────────────
        COL_XS     = [-4.8, -1.6, 1.6, 4.8]
        COL_NAMES  = ["Original", "Drop rows", "Mean impute", "Forward-fill"]
        COL_COLORS = [C_DIM,      C_RED,       C_AMBER,       C_TEAL      ]
        CW, CH     = 1.72, 0.52
        Y0         = 1.30   # header row centre

        def orig_cells():
            return [("22", C_WHITE, None), ("NaN", C_RED, C_RED), ("35", C_WHITE, None),
                    ("NaN", C_RED, C_RED), ("28", C_WHITE, None)]

        def drop_cells():
            return [("22", C_WHITE, None), ("---", C_DEAD, None), ("35", C_WHITE, None),
                    ("---", C_DEAD, None), ("28", C_WHITE, None)]

        def mean_cells():
            return [("22", C_WHITE, None), ("28.3", C_AMBER, C_AMBER), ("35", C_WHITE, None),
                    ("28.3", C_AMBER, C_AMBER), ("28", C_WHITE, None)]

        def ffill_cells():
            return [("22", C_WHITE, None), ("22", C_TEAL, C_TEAL), ("35", C_WHITE, None),
                    ("35", C_TEAL, C_TEAL), ("28", C_WHITE, None)]

        all_cell_data = [orig_cells(), drop_cells(), mean_cells(), ffill_cells()]

        # Strategy name labels (above header row) — free of cells, fit to column pitch
        strat_lbls = VGroup(*[
            _tx(name, FS_STRAT, color=col, bold=True, w=2.9)
            .move_to([COL_XS[c], Y0 + CH + 0.30, 0])
            for c, (name, col) in enumerate(zip(COL_NAMES, COL_COLORS))
        ])

        # "Age" header cells
        headers = VGroup(*[
            _cell("Age", COL_XS[c], Y0, CW, CH, bg=C_DEAD, tc=col, fs=FS_HEAD, bold=True)
            for c, col in enumerate(COL_COLORS)
        ])

        # Data cell groups (one VGroup per column)
        col_groups = []
        for c, cells in enumerate(all_cell_data):
            grp = VGroup(*[
                _cell(val, COL_XS[c], Y0 - (r + 1) * CH, CW, CH, bg=bg, tc=tc, fs=FS_CELL)
                for r, (val, tc, bg) in enumerate(cells)
            ])
            col_groups.append(grp)

        # Annotations below table (two lines each)
        ANN_Y = Y0 - 6 * CH - 0.62
        annotations = [
            None,
            _stack(["rows removed", r"$\rightarrow$ smaller dataset"], FS_ANN, C_RED,
                   COL_XS[1], ANN_Y, w=2.9),
            _stack([r"NaN $\rightarrow$ column mean", r"$\rightarrow$ preserves size"], FS_ANN, C_AMBER,
                   COL_XS[2], ANN_Y, w=2.9),
            _stack([r"NaN $\rightarrow$ previous value", r"$\rightarrow$ assumes ordering"], FS_ANN, C_TEAL,
                   COL_XS[3], ANN_Y, w=2.9),
        ]

        # ── Animate ───────────────────────────────────────────────────────────
        self.play(FadeIn(strat_lbls), FadeIn(headers), run_time=0.40)
        self.play(FadeIn(col_groups[0]), run_time=0.35)
        self.wait(0.4)

        for c in range(1, 4):
            anims = [FadeIn(col_groups[c])]
            if annotations[c]:
                anims.append(FadeIn(annotations[c]))
            self.play(*anims, run_time=0.40)
            self.wait(0.75)

        all_anns = VGroup(*[a for a in annotations if a is not None])
        self._title  = title
        self._p1_all = VGroup(sub, strat_lbls, headers, *col_groups, all_anns)

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 2 — Encoding
    # ══════════════════════════════════════════════════════════════════════════

    def _phase2(self):
        self.play(FadeOut(self._p1_all), run_time=0.40)

        sub = _tx("Encoding --- convert categories to numbers the model can use",
                  FS_SUB, color=C_DIM, w=12.6).next_to(self._title, DOWN, buff=0.16)
        self.play(FadeIn(sub), run_time=0.28)

        # ── Data ──────────────────────────────────────────────────────────────
        SEX_VALS = ["male", "female", "female", "male", "male"]
        MALE_OHE = [1, 0, 0, 1, 1]
        FEM_OHE  = [0, 1, 1, 0, 0]

        CW, CH = 1.72, 0.50
        Y0     = 1.10
        NR     = len(SEX_VALS)

        X_SEX  = -4.7
        X_MALE = -1.5
        X_FEM  =  0.75

        sex_header = _cell("Sex", X_SEX, Y0, CW, CH, bg=C_DEAD, tc=C_DIM, fs=FS_HEAD, bold=True)
        sex_data = VGroup(*[
            _cell(val, X_SEX, Y0 - (r + 1) * CH, CW, CH,
                  bg=C_TEAL if val == "male" else C_PURP,
                  tc=C_TEAL if val == "male" else C_PURP, fs=FS_CELL)
            for r, val in enumerate(SEX_VALS)
        ])

        male_header = _cell(r"Sex\_male", X_MALE, Y0, CW, CH,
                            bg=C_DEAD, tc=C_TEAL, fs=FS_HEAD, bold=True)
        male_data = VGroup(*[
            _cell(str(v), X_MALE, Y0 - (r + 1) * CH, CW, CH,
                  bg=C_TEAL if v == 1 else None,
                  tc=C_TEAL if v == 1 else C_DEAD, fs=FS_CELL, bold=(v == 1))
            for r, v in enumerate(MALE_OHE)
        ])

        fem_header = _cell(r"Sex\_female", X_FEM, Y0, 1.95, CH,
                           bg=C_DEAD, tc=C_PURP, fs=FS_HEAD, bold=True)
        fem_data = VGroup(*[
            _cell(str(v), X_FEM, Y0 - (r + 1) * CH, 1.95, CH,
                  bg=C_PURP if v == 1 else None,
                  tc=C_PURP if v == 1 else C_DEAD, fs=FS_CELL, bold=(v == 1))
            for r, v in enumerate(FEM_OHE)
        ])

        # Arrow and label
        arr_y = Y0 - NR * CH / 2
        arr = Arrow(
            [X_SEX + CW / 2 + 0.10, arr_y, 0], [X_MALE - CW / 2 - 0.10, arr_y, 0],
            buff=0.0, color=C_DIM, stroke_width=2.4, max_tip_length_to_length_ratio=0.22,
        )
        arr_lbl = _tx("one-hot", FS_ANN, color=C_DIM).next_to(arr, UP, buff=0.08)

        # Note box (right side)
        NOTE_X = 3.55
        note_bg = RoundedRectangle(
            width=3.05, height=1.75, corner_radius=0.12,
            fill_color=C_DEAD, fill_opacity=0.20, stroke_color=C_DEAD, stroke_width=1.4,
        ).move_to([NOTE_X, arr_y, 0])
        note_lines = _stack(
            [r"$k = 2$ categories", r"$\rightarrow$ 2 dummy columns",
             "drop one to avoid", "multicollinearity"],
            FS_NOTE, [C_WHITE, C_DIM, C_AMBER, C_AMBER], NOTE_X, arr_y, buff=0.16, w=2.7,
        )
        note = VGroup(note_bg, note_lines)

        caption = _tx(
            r"\mbox{Sex\_female $= 1 - $ Sex\_male $\;\cdot\;$ one column is redundant "
            r"$\;\cdot\;$ keep $k-1$ dummies}",
            FS_CAP, color=C_DIM, w=13.2,
        ).to_edge(DOWN, buff=0.34)

        # ── Animate ───────────────────────────────────────────────────────────
        self.play(FadeIn(sex_header), FadeIn(sex_data), run_time=0.40)
        self.wait(0.3)
        self.play(GrowArrow(arr), FadeIn(arr_lbl), run_time=0.40)
        self.play(FadeIn(male_header), FadeIn(male_data), run_time=0.40)
        self.play(FadeIn(fem_header), FadeIn(fem_data), run_time=0.40)
        self.play(FadeIn(note), run_time=0.35)
        self.play(FadeIn(caption), run_time=0.30)
        self.wait(1.8)

        self._p2_all = VGroup(
            sub, sex_header, sex_data, arr, arr_lbl,
            male_header, male_data, fem_header, fem_data, note, caption,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 3 — Normalisation
    # ══════════════════════════════════════════════════════════════════════════

    def _phase3(self):
        self.play(FadeOut(self._p2_all), run_time=0.40)

        sub = _tx("Normalisation --- rescale features so no column dominates",
                  FS_SUB, color=C_DIM, w=12.6).next_to(self._title, DOWN, buff=0.16)
        self.play(FadeIn(sub), run_time=0.28)

        # ── Data ──────────────────────────────────────────────────────────────
        ages = np.array([22, 28, 35, 38, 45, 52], dtype=float)
        mm   = (ages - ages.min()) / (ages.max() - ages.min())
        z    = (ages - ages.mean()) / ages.std()

        AX_L, AX_R = -2.90, 2.90
        AX_SPAN    = AX_R - AX_L
        Y_LINES    = [1.35, 0.00, -1.35]
        DOT_R      = 0.10
        DOT_COL    = [C_BLUE, C_GREEN, C_AMBER]

        def to_x(vals, v_lo, v_hi):
            return AX_L + (vals - v_lo) / (v_hi - v_lo) * AX_SPAN

        orig_xs = to_x(ages, 18.0, 56.0)
        mm_xs   = to_x(mm,   0.0,  1.0)
        z_xs    = to_x(z,   -2.3,  2.3)

        configs = [
            (orig_xs, "Original",
             (r"range: 22--52", False), (r"(no rescaling)", False),
             to_x(np.array([20., 30., 40., 50.]), 18., 56.), ["20", "30", "40", "50"]),
            (mm_xs, "Min-Max",
             (r"x' = (x - \min)/(\max - \min)", True), (r"maps to $[0, 1]$", False),
             to_x(np.array([0., 0.5, 1.0]), 0., 1.), ["0", "0.5", "1"]),
            (z_xs, "Standard",
             (r"x' = (x - \mu)/\sigma", True), (r"$\mu = 36.7,\ \sigma = 10.2$", False),
             to_x(np.array([-2., -1., 0., 1., 2.]), -2.3, 2.3), ["-2", "-1", "0", "1", "2"]),
        ]

        all_frames = VGroup()
        all_dot_groups = []

        for (dot_xs, label, (form1, m1), (form2, m2), tick_xs, tick_lbls), ly, dcol in \
                zip(configs, Y_LINES, DOT_COL):

            frame = VGroup()
            frame.add(Line([AX_L, ly, 0], [AX_R, ly, 0], stroke_color=C_DIM, stroke_width=2.0))

            for tx, tlbl in zip(tick_xs, tick_lbls):
                frame.add(Line([tx, ly - 0.10, 0], [tx, ly + 0.10, 0],
                               stroke_color=C_DIM, stroke_width=1.2))
                frame.add(_tx(tlbl, FS_TICK, color=C_DIM, w=0.7).move_to([tx, ly - 0.30, 0]))

            frame.add(_tx(label, FS_LBL, color=C_WHITE, bold=True, w=1.9)
                      .move_to([AX_L - 1.55, ly, 0]))

            frame.add(_tx(form1, FS_FORM, color=dcol, math=m1, w=3.4)
                      .move_to([AX_R + 1.95, ly + 0.22, 0]))
            frame.add(_tx(form2, FS_FORM, color=C_DIM, math=m2, w=3.4)
                      .move_to([AX_R + 1.95, ly - 0.22, 0]))

            all_frames.add(frame)

            dots = VGroup(*[
                Dot([dx, ly, 0], radius=DOT_R, fill_color=dcol, fill_opacity=0.90)
                for dx in dot_xs
            ])
            all_dot_groups.append(dots)

        caption = _tx(
            r"\mbox{shape is preserved $\;\cdot\;$ "
            r"use Standard for distance-based models (kNN, SVM, PCA)}",
            FS_CAP, color=C_DIM, w=13.2,
        ).to_edge(DOWN, buff=0.34)

        # ── Animate ───────────────────────────────────────────────────────────
        for frame, dots in zip(all_frames, all_dot_groups):
            self.play(FadeIn(frame), run_time=0.35)
            self.play(
                LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.12),
                run_time=0.55,
            )
            self.wait(0.4)

        self.play(FadeIn(caption), run_time=0.35)
        self.wait(2.5)
