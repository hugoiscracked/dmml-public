"""
W05 — Permutation Feature Importance

Three-phase animation for Lecture 2 (Random Forests).

  Phase 1: Trained Random Forest + baseline accuracy; the key question:
           which features actually matter?
  Phase 2: Shuffle each feature in turn — accuracy drops more for
           important features; importance = baseline − shuffled accuracy.
  Phase 3: Horizontal bar chart sorted high → low; caption on using
           importance scores for feature selection.

Dataset:  Iris (4 features).  Values are representative, not exact.

Render:
  ../../env/bin/manim -pql feature_importance.py FeatureImportance
  ../../env/bin/manim -pqh feature_importance.py FeatureImportance
"""

from manim import *
import numpy as np

# ── Palette ───────────────────────────────────────────────────────────────────
C_BG    = "#0d1117"
C_WHITE = "#ffffff"
C_DIM   = "#8b949e"
C_GREEN = "#3fb950"
C_RED   = "#f78166"
C_BLUE  = "#58a6ff"

FEAT_COLS = ["#ffa657", "#2dd4bf", "#bc8cff", "#e3b341"]

# ── Data ──────────────────────────────────────────────────────────────────────
FEATURES    = ["petal length", "petal width", "sepal length", "sepal width"]
BASELINE    = 0.96
SHUF_ACCS   = [0.51, 0.58, 0.85, 0.90]
IMPORTANCES = [round(BASELINE - s, 2) for s in SHUF_ACCS]
# [0.45, 0.38, 0.11, 0.06]

# Descending sort order for bar chart
ORDER = sorted(range(4), key=lambda i: IMPORTANCES[i], reverse=True)
# [0, 1, 2, 3] — already descending


# ── Scene ─────────────────────────────────────────────────────────────────────

class FeatureImportance(Scene):
    def construct(self):
        self.camera.background_color = C_BG

        # ════════════════════════════════════════════════════════════════════
        #  Phase 1 — Baseline
        # ════════════════════════════════════════════════════════════════════
        title = (
            Text("Permutation Feature Importance",
                 font_size=18, color=C_WHITE, weight=BOLD)
            .to_edge(UP, buff=0.35)
        )
        self.play(FadeIn(title), run_time=0.4)

        rf_box = RoundedRectangle(
            corner_radius=0.10, width=5.4, height=0.88,
            fill_color="#161b22", fill_opacity=1.0,
            stroke_color=C_BLUE, stroke_width=1.6,
        ).move_to([0, 1.1, 0])
        rf_txt = (
            Text("Random Forest   (100 trees,  4 features)",
                 font_size=14, color=C_BLUE)
            .move_to(rf_box)
        )

        base_lbl = Text("Baseline accuracy:", font_size=16, color=C_DIM).move_to([-1.2, -0.1, 0])
        base_val = (
            Text(f"{BASELINE:.2f}", font_size=22, color=C_GREEN, weight=BOLD)
            .next_to(base_lbl, RIGHT, buff=0.30)
        )
        question = (
            Text("Which features matter most?", font_size=14, color=C_DIM)
            .move_to([0, -1.05, 0])
        )
        strategy = (
            Text(
                "Strategy: shuffle one feature at a time \u2014 measure the accuracy drop.",
                font_size=12, color=C_DIM,
            )
            .move_to([0, -1.70, 0])
        )

        self.play(FadeIn(rf_box), FadeIn(rf_txt), run_time=0.5)
        self.play(FadeIn(base_lbl), FadeIn(base_val), run_time=0.4)
        self.play(FadeIn(question), run_time=0.3)
        self.play(FadeIn(strategy), run_time=0.4)
        self.wait(1.2)

        # ════════════════════════════════════════════════════════════════════
        #  Phase 2 — Permutation table
        # ════════════════════════════════════════════════════════════════════
        title2 = (
            Text("Shuffle each feature \u2192 measure accuracy drop",
                 font_size=18, color=C_WHITE, weight=BOLD)
            .to_edge(UP, buff=0.35)
        )
        p1 = VGroup(rf_box, rf_txt, base_lbl, base_val, question, strategy)
        self.play(FadeOut(p1), ReplacementTransform(title, title2), run_time=0.6)

        # Formula subtitle
        formula = (
            Text(
                f"importance  =  baseline ({BASELINE:.2f})  \u2212  accuracy after shuffle",
                font_size=12, color=C_DIM,
            )
            .move_to([0, 2.15, 0])
        )
        self.play(FadeIn(formula), run_time=0.4)

        # Table header
        COL_X = [-3.5, -0.3, 1.65, 3.55]
        hdr_f   = Text("Feature",      font_size=11, color=C_DIM).move_to([COL_X[0], 1.65, 0])
        hdr_a   = Text("Shuffled acc", font_size=11, color=C_DIM).move_to([COL_X[1], 1.65, 0])
        hdr_i   = Text("Importance",   font_size=11, color=C_DIM).move_to([COL_X[3], 1.65, 0])
        hdr_sep = Line([-5.5, 1.40, 0], [5.5, 1.40, 0], color=C_DIM, stroke_width=0.7)
        self.play(FadeIn(VGroup(hdr_f, hdr_a, hdr_i, hdr_sep)), run_time=0.30)

        ROW_Y      = [1.0, 0.15, -0.70, -1.55]
        row_groups = []

        for i in range(4):
            ry  = ROW_Y[i]
            col = FEAT_COLS[i]

            fname = (
                Text(f"\u25cf  {FEATURES[i]}", font_size=13, color=col)
                .move_to([COL_X[0], ry, 0])
            )
            sacc = (
                Text(f"{SHUF_ACCS[i]:.2f}", font_size=13, color=C_RED)
                .move_to([COL_X[1], ry, 0])
            )
            eq = (
                Text(
                    f"= {BASELINE:.2f} \u2212 {SHUF_ACCS[i]:.2f}",
                    font_size=11, color=C_DIM,
                )
                .move_to([COL_X[2], ry, 0])
            )
            ival = (
                Text(f"+{IMPORTANCES[i]:.2f}", font_size=14, color=col, weight=BOLD)
                .move_to([COL_X[3], ry, 0])
            )

            rg = VGroup(fname, sacc, eq, ival)
            row_groups.append(rg)

            self.play(FadeIn(rg, shift=RIGHT * 0.10), run_time=0.45)
            self.wait(0.30)

        table_all = VGroup(formula, hdr_f, hdr_a, hdr_i, hdr_sep, *row_groups)
        self.wait(0.9)

        # ════════════════════════════════════════════════════════════════════
        #  Phase 3 — Horizontal bar chart
        # ════════════════════════════════════════════════════════════════════
        title3 = (
            Text("Feature Importance Ranking",
                 font_size=18, color=C_WHITE, weight=BOLD)
            .to_edge(UP, buff=0.35)
        )
        self.play(
            FadeOut(table_all),
            ReplacementTransform(title2, title3),
            run_time=0.6,
        )

        BAR_X0 = -2.6    # left edge of all bars (zero line)
        SCALE  = 9.0     # data units → screen units
        BAR_H  = 0.50
        BAR_Y  = [1.55, 0.65, -0.25, -1.15]

        # Vertical zero-line
        zero_line = Line(
            [BAR_X0, BAR_Y[-1] - 0.55, 0],
            [BAR_X0, BAR_Y[0]  + 0.45, 0],
            color=C_DIM, stroke_width=1.0,
        )
        zero_lbl = (
            Text("0", font_size=9, color=C_DIM)
            .next_to(np.array([BAR_X0, BAR_Y[-1] - 0.55, 0]), DOWN, buff=0.10)
        )

        # Tick marks on x-axis
        ticks = VGroup()
        for v in [0.1, 0.2, 0.3, 0.4]:
            tx = BAR_X0 + v * SCALE
            ty = BAR_Y[-1] - 0.55
            tick = Line([tx, ty, 0], [tx, ty - 0.14, 0], color=C_DIM, stroke_width=0.8)
            tlbl = (
                Text(f"{v:.1f}", font_size=9, color=C_DIM)
                .next_to(np.array([tx, ty - 0.14, 0]), DOWN, buff=0.08)
            )
            ticks.add(tick, tlbl)

        self.play(Create(zero_line), FadeIn(zero_lbl), FadeIn(ticks), run_time=0.5)

        bars     = []
        flabels  = []
        vlabels  = []

        for rank, i in enumerate(ORDER):
            ry  = BAR_Y[rank]
            col = FEAT_COLS[i]
            imp = IMPORTANCES[i]
            bw  = imp * SCALE

            bar = Rectangle(
                width=bw, height=BAR_H,
                fill_color=col, fill_opacity=0.82,
                stroke_width=0,
            ).move_to([BAR_X0 + bw / 2, ry, 0])
            bars.append(bar)

            flbl = (
                Text(FEATURES[i], font_size=13, color=col)
                .next_to(np.array([BAR_X0, ry, 0]), LEFT, buff=0.25)
            )
            flabels.append(flbl)

            vlbl = (
                Text(f"{imp:.2f}", font_size=12, color=col)
                .next_to(np.array([BAR_X0 + bw, ry, 0]), RIGHT, buff=0.18)
            )
            vlabels.append(vlbl)

        self.play(FadeIn(VGroup(*flabels)), run_time=0.4)
        self.play(
            LaggedStart(*[GrowFromEdge(b, LEFT) for b in bars], lag_ratio=0.20),
            run_time=1.1,
        )
        self.play(FadeIn(VGroup(*vlabels)), run_time=0.30)
        self.wait(0.4)

        # Annotation on the top bar
        top_ann = (
            Text("most important", font_size=11, color=FEAT_COLS[ORDER[0]])
            .next_to(bars[0], UP, buff=0.12)
        )
        self.play(FadeIn(top_ann), run_time=0.35)
        self.wait(0.4)

        # Closing caption
        cap = (
            Text(
                "low-importance features can be dropped \u2014 less noise, faster training",
                font_size=12, color=C_DIM,
            )
            .to_edge(DOWN, buff=0.40)
        )
        self.play(FadeIn(cap), run_time=0.5)
        self.wait(2.5)
