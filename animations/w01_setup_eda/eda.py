"""
W01 — Exploratory Data Analysis

  Phase 1: Dataset at a glance.
           A Titanic-like table appears row by row; each column is
           highlighted in turn with a profile card showing dtype,
           null count, and descriptive statistics.

  Phase 2: Distributions.
           Histograms for Age (near-normal) and Fare (right-skewed)
           plus a bar chart for Survived reveal shape and class imbalance.

  Phase 3: Correlation heatmap.
           A colour-coded 4×4 matrix shows pairwise linear relationships
           between numeric features.

Render:
  ../../env/bin/manim -pql eda.py EDA
  ../../env/bin/manim -pqh eda.py EDA
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


def _cell(text, cx, cy, w, h, bg=None, tc=C_DIM, font_size=10, bold=False):
    rect = Rectangle(
        width=w, height=h,
        fill_color=bg or C_BG, fill_opacity=1.0 if bg else 0.0,
        stroke_color=C_DEAD, stroke_width=0.8,
    ).move_to([cx, cy, 0])
    kw = {"weight": BOLD} if bold else {}
    txt = Text(text, font_size=font_size, color=tc, **kw).move_to([cx, cy, 0])
    return VGroup(rect, txt)


class EDA(Scene):

    def construct(self):
        self.camera.background_color = C_BG
        self._phase1()
        self._phase2()
        self._phase3()

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 1 — Dataset profiling
    # ══════════════════════════════════════════════════════════════════════════

    def _phase1(self):
        title = (
            Text("Exploratory Data Analysis", font_size=24,
                 color=C_WHITE, weight=BOLD)
            .to_edge(UP, buff=0.25)
        )
        sub = (
            Text("Titanic dataset  ·  inspect every column before modelling",
                 font_size=13, color=C_DIM)
            .next_to(title, DOWN, buff=0.12)
        )
        self.play(FadeIn(title), FadeIn(sub), run_time=0.5)

        # ── Table data ────────────────────────────────────────────────────────
        COLS = ["Age", "Fare", "Survived", "Pclass", "Sex"]
        DATA = [
            ["22",  "7.25",  "0", "3", "male"  ],
            ["38",  "71.28", "1", "1", "female"],
            ["NaN", "7.92",  "1", "3", "female"],
            ["35",  "53.10", "0", "1", "male"  ],
            ["28",  "8.05",  "0", "3", "male"  ],
        ]
        CW, CH = 1.22, 0.40
        NC = len(COLS)
        NR = len(DATA)
        X0 = -(NC * CW) / 2 + CW / 2   # centre of leftmost column
        Y0 = 1.30                        # centre of header row

        def cxy(col, row):
            return X0 + col * CW, Y0 - row * CH

        # Header row
        headers = VGroup(*[
            _cell(name, *cxy(c, 0), CW, CH,
                  bg=C_DEAD, tc=C_WHITE, font_size=11, bold=True)
            for c, name in enumerate(COLS)
        ])

        # Data rows (NaN cells highlighted red)
        data_rows = []
        for r, row in enumerate(DATA):
            grp = VGroup(*[
                _cell(val, *cxy(c, r + 1), CW, CH,
                      tc=C_RED if val == "NaN" else C_DIM, font_size=10)
                for c, val in enumerate(row)
            ])
            data_rows.append(grp)

        self.play(FadeIn(headers), run_time=0.30)
        self.play(
            LaggedStart(*[FadeIn(r) for r in data_rows], lag_ratio=0.18),
            run_time=0.75,
        )
        self.wait(0.3)

        # ── Column profiles ───────────────────────────────────────────────────
        profiles = [
            ("Age",      "numeric",     "1 null",   "range  22 – 38",       C_BLUE ),
            ("Fare",     "numeric",     "no nulls",  "range  7.25 – 71.28", C_AMBER),
            ("Survived", "binary",      "no nulls",  "values  0 / 1",       C_GREEN),
            ("Pclass",   "ordinal",     "no nulls",  "values  1, 2, 3",     C_PURP ),
            ("Sex",      "categorical", "no nulls",  "male / female",        C_TEAL ),
        ]

        CARD_Y = Y0 - (NR + 1) * CH - 0.60   # centre of profile card

        prev_highlight = None
        prev_card      = None

        for c, (col_name, dtype, nulls, vals, col_color) in enumerate(profiles):
            # Highlight overlay covering all rows for this column
            highlight = VGroup(*[
                Rectangle(
                    width=CW - 0.06, height=CH - 0.06,
                    fill_color=col_color, fill_opacity=0.20,
                    stroke_color=col_color, stroke_width=1.2,
                ).move_to([*cxy(c, r), 0])
                for r in range(NR + 1)
            ])

            # Profile card
            card_bg = RoundedRectangle(
                width=6.00, height=0.88, corner_radius=0.10,
                fill_color=col_color, fill_opacity=0.12,
                stroke_color=col_color, stroke_width=1.4,
            ).move_to([0, CARD_Y, 0])
            line1 = (
                Text(f"{col_name}  ·  {dtype}", font_size=13,
                     color=col_color, weight=BOLD)
                .move_to([0, CARD_Y + 0.18, 0])
            )
            line2 = (
                Text(f"{nulls}  ·  {vals}", font_size=11, color=C_DIM)
                .move_to([0, CARD_Y - 0.20, 0])
            )
            card = VGroup(card_bg, line1, line2)

            if prev_card is None:
                self.play(FadeIn(highlight), FadeIn(card), run_time=0.35)
            else:
                self.play(
                    FadeOut(prev_highlight), FadeOut(prev_card),
                    FadeIn(highlight), FadeIn(card),
                    run_time=0.35,
                )
            self.wait(0.85)
            prev_highlight = highlight
            prev_card      = card

        self._title  = title
        self._p1_all = VGroup(
            sub, headers, *data_rows,
            prev_highlight, prev_card,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 2 — Distributions
    # ══════════════════════════════════════════════════════════════════════════

    def _phase2(self):
        self.play(FadeOut(self._p1_all), run_time=0.40)

        sub = (
            Text("Distributions — shape, spread, and skew at a glance",
                 font_size=13, color=C_DIM)
            .next_to(self._title, DOWN, buff=0.12)
        )
        self.play(FadeIn(sub), run_time=0.28)

        BASELINE_Y = -1.55
        BAR_MAX_H  =  2.10
        BAR_W      =  0.48
        STEP       =  BAR_W + 0.10   # bar width + gap

        def _make_plot(cx, title_str, labels, heights, colors, note=None):
            """Returns (frame VGroup, list of bar Rectangles).
            Frame holds the baseline, title, and bin labels.
            Bars are returned separately so they can be animated with GrowFromEdge.
            """
            n  = len(heights)
            x0 = cx - (n - 1) * STEP / 2

            frame = VGroup()
            # baseline
            frame.add(Line(
                [x0 - BAR_W / 2 - 0.08, BASELINE_Y, 0],
                [x0 + (n - 1) * STEP + BAR_W / 2 + 0.08, BASELINE_Y, 0],
                stroke_color=C_DIM, stroke_width=1.0,
            ))
            # title
            frame.add(
                Text(title_str, font_size=13, color=C_WHITE, weight=BOLD)
                .move_to([cx, BASELINE_Y + BAR_MAX_H + 0.38, 0])
            )
            # bin labels
            for i, lbl in enumerate(labels):
                frame.add(
                    Text(lbl, font_size=8, color=C_DIM)
                    .move_to([x0 + i * STEP, BASELINE_Y - 0.22, 0])
                )
            if note:
                frame.add(
                    Text(note, font_size=9, color=C_DIM)
                    .move_to([cx, BASELINE_Y - 0.50, 0])
                )

            # bars (not in frame — animated separately)
            col_list = colors if isinstance(colors, list) else [colors] * n
            bars = []
            for i, (h_rel, col) in enumerate(zip(heights, col_list)):
                h   = max(h_rel * BAR_MAX_H, 0.04)
                bx  = x0 + i * STEP
                bar = Rectangle(
                    width=BAR_W, height=h,
                    fill_color=col, fill_opacity=0.78,
                    stroke_color=col, stroke_width=0.8,
                ).move_to([bx, BASELINE_Y + h / 2, 0])
                bars.append(bar)

            return frame, bars

        # Age — near-normal distribution
        frame_age, bars_age = _make_plot(
            -4.20, "Age",
            ["0-15", "16-25", "26-35", "36-45", "46-55", "56+"],
            [0.22, 0.65, 1.00, 0.75, 0.42, 0.18],
            C_BLUE,
            note="near-normal distribution",
        )

        # Fare — right-skewed
        frame_fare, bars_fare = _make_plot(
            0.00, "Fare",
            ["0-25", "26-50", "51-75", "76-100", "101-150", "150+"],
            [1.00, 0.58, 0.30, 0.16, 0.09, 0.05],
            C_AMBER,
            note="right-skewed  ·  long tail",
        )

        # Survived — binary, two colours
        frame_surv, bars_surv = _make_plot(
            4.20, "Survived",
            ["died", "survived"],
            [1.00, 0.61],
            [C_RED, C_GREEN],
            note="class imbalance  ·  38 % survived",
        )

        # Frames appear first, then bars grow upward
        self.play(
            LaggedStart(
                FadeIn(frame_age), FadeIn(frame_fare), FadeIn(frame_surv),
                lag_ratio=0.25,
            ),
            run_time=0.75,
        )
        all_bars = bars_age + bars_fare + bars_surv
        self.play(
            LaggedStart(*[GrowFromEdge(b, DOWN) for b in all_bars], lag_ratio=0.05),
            run_time=1.20,
        )
        self.wait(1.5)

        self._p2_all = VGroup(sub, frame_age, frame_fare, frame_surv, *all_bars)

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 3 — Correlation heatmap
    # ══════════════════════════════════════════════════════════════════════════

    def _phase3(self):
        self.play(FadeOut(self._p2_all), run_time=0.40)

        sub = (
            Text("Correlation heatmap — linear relationships between numeric features",
                 font_size=13, color=C_DIM)
            .next_to(self._title, DOWN, buff=0.12)
        )
        self.play(FadeIn(sub), run_time=0.28)

        FEAT = ["Age", "Fare", "Pclass", "Survived"]
        # Approximate Titanic Pearson correlations
        CORR = [
            [ 1.00, -0.05, -0.37, -0.08],
            [-0.05,  1.00, -0.55,  0.26],
            [-0.37, -0.55,  1.00, -0.34],
            [-0.08,  0.26, -0.34,  1.00],
        ]
        N    = len(FEAT)
        CELL = 1.18
        OX   = -(N - 1) * CELL / 2   # x-centre of column 0
        OY   =  (N - 1) * CELL / 2   # y-centre of row 0

        cells = VGroup()
        for r in range(N):
            for c in range(N):
                v   = CORR[r][c]
                t   = (v + 1) / 2   # map [-1, 1] → [0, 1]
                col = interpolate_color(ManimColor(C_RED), ManimColor(C_GREEN), t)
                cx, cy = OX + c * CELL, OY - r * CELL
                sq = Square(
                    side_length=CELL - 0.07,
                    fill_color=col, fill_opacity=0.85,
                    stroke_color=C_BG, stroke_width=1.5,
                ).move_to([cx, cy, 0])
                # dark text on bright cells, white on dark ones
                txt = Text(
                    f"{v:+.2f}", font_size=10, weight=BOLD,
                    color=C_BG if abs(v) > 0.45 else C_WHITE,
                ).move_to([cx, cy, 0])
                cells.add(VGroup(sq, txt))

        # Row labels (left of grid)
        row_lbls = VGroup(*[
            Text(name, font_size=12, color=C_DIM)
            .move_to([OX - CELL * 0.80, OY - r * CELL, 0])
            for r, name in enumerate(FEAT)
        ])

        # Column labels (below grid)
        col_lbls = VGroup(*[
            Text(name, font_size=12, color=C_DIM)
            .move_to([OX + c * CELL, OY - (N - 1) * CELL - CELL * 0.60, 0])
            for c, name in enumerate(FEAT)
        ])

        self.play(
            LaggedStart(*[FadeIn(cell) for cell in cells], lag_ratio=0.04),
            run_time=1.20,
        )
        self.play(FadeIn(row_lbls), FadeIn(col_lbls), run_time=0.35)

        caption = (
            Text(
                "red = negative  ·  green = positive  ·  "
                "Pclass (3rd class) correlates with lower Fare and survival",
                font_size=11, color=C_DIM,
            )
            .to_edge(DOWN, buff=0.32)
        )
        self.play(FadeIn(caption), run_time=0.35)
        self.wait(2.5)
