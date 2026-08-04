"""
W02 — AR(1) Intuition

Visualises the "pull" mechanic of an AR(1) process: each new value is the
previous value scaled by φ (memory) plus a random kick ε[t] (noise).
Phase 1 shows the formula, Phase 2 builds the series step by step with
annotated arrows, Phase 3 compares three values of φ.

Render with:
  ../../env/bin/manim -pql ar1.py AR1
  ../../env/bin/manim -pqh ar1.py AR1
"""

from manim import *
import numpy as np

# ── Palette ───────────────────────────────────────────────────────────────────
C_BG   = "#0d1117"
C_SER  = "#ffa657"   # amber  – series dots / line
C_MEM  = "#58a6ff"   # blue   – φ·y[t-1] memory component
C_NOIS = "#8b949e"   # grey   – ε[t] noise component
C_DIM  = "#8b949e"
C_PHI  = "#bc8cff"   # purple – φ value callout

_AX = dict(
    axis_config={"color": "#21262d", "stroke_width": 1.2, "include_ticks": False},
    tips=False,
)

# ── Step-by-step series  (φ = 0.8, starts high to make the pull obvious) ─────
PHI    = 0.8
SIGMA  = 0.5
N_SLOW = 5      # steps shown with annotated arrows
N_TOTAL = 18    # total steps drawn in Phase 2

rng_main = np.random.default_rng(3)
eps_main = rng_main.normal(0, SIGMA, N_TOTAL)
y = [2.0]       # start high so pull-back toward 0 is immediately visible
for e in eps_main:
    y.append(PHI * y[-1] + e)
y = np.array(y)   # shape (N_TOTAL+1,)

# ── Comparison series  (same noise, three φ values) ──────────────────────────
N_COMP = 50
rng_comp = np.random.default_rng(9)
eps_comp = rng_comp.normal(0, SIGMA, N_COMP)

PHI_VALS   = [0.95,         0.3,        -0.7]
PHI_LABELS = ["φ = 0.95",  "φ = 0.3",  "φ = \u22120.7"]
PHI_TAGS   = ["persistent", "moderate", "oscillating"]
PHI_COLORS = ["#58a6ff",   "#ffa657",  "#f78166"]

def gen_ar1(phi, eps):
    s = [0.0]
    for e in eps:
        s.append(phi * s[-1] + e)
    return np.array(s[1:])   # length N_COMP

comp_series = [gen_ar1(p, eps_comp) for p in PHI_VALS]


class AR1(Scene):
    def construct(self):
        self.camera.background_color = C_BG

        # ════════════════════════════════════════════════════════════════════
        #  Phase 1 — Formula
        # ════════════════════════════════════════════════════════════════════
        formula = Text("y[t]  =  \u03c6 \u00b7 y[t\u22121]  +  \u03b5[t]",
                       font_size=30, color=WHITE)
        formula.move_to([0, 1.8, 0])

        part_mem = Text("\u03c6 \u00b7 y[t\u22121]   \u2192   memory component   (here \u03c6 = 0.8)",
                        font_size=15, color=C_MEM)
        part_mem.next_to(formula, DOWN, buff=0.35)
        part_noi = Text("\u03b5[t]   \u2192   random noise",
                        font_size=15, color=C_NOIS)
        part_noi.next_to(part_mem, DOWN, buff=0.18)

        self.play(FadeIn(formula, shift=UP * 0.2), run_time=0.7)
        self.play(FadeIn(part_mem), run_time=0.4)
        self.play(FadeIn(part_noi), run_time=0.4)
        self.wait(1.2)
        self.play(FadeOut(VGroup(formula, part_mem, part_noi)), run_time=0.6)

        # ════════════════════════════════════════════════════════════════════
        #  Phase 2 — Step-by-step pull visualisation
        # ════════════════════════════════════════════════════════════════════
        y_min = float(y.min()) - 0.35
        y_max = float(y.max()) + 0.35

        ax = Axes(
            x_range=[0, N_TOTAL, 5],
            y_range=[y_min, y_max, 1.0],
            x_length=10.5, y_length=3.6,
            **_AX,
        ).move_to([0.3, -0.2, 0])

        ax_title = (
            Text("Building AR(1) step by step   (\u03c6 = 0.8)",
                 font_size=15, color=C_DIM)
            .next_to(ax, UP, buff=0.2)
        )

        zero_line = DashedLine(
            ax.c2p(0, 0), ax.c2p(N_TOTAL, 0),
            color="#2d333b", stroke_width=1.2, dash_length=0.15,
        )

        # Corner legend
        legend = VGroup(
            Text("\u2013\u2013  \u03c6\u00b7y[t\u22121]   memory", font_size=11, color=C_MEM),
            Text("\u2195  \u03b5[t]            noise",  font_size=11, color=C_NOIS),
        ).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
        legend.to_corner(UR, buff=0.3)

        self.play(Create(ax), FadeIn(ax_title), Create(zero_line), run_time=0.8)
        self.play(FadeIn(legend), run_time=0.4)

        # Initial dot at t = 0
        dots = [Dot(ax.c2p(0, float(y[0])), color=C_SER, radius=0.09)]
        self.play(GrowFromCenter(dots[0]), run_time=0.4)
        self.wait(0.3)

        segs = []

        for t in range(1, N_SLOW + 1):
            pred_y = PHI * float(y[t - 1])

            # 1. Highlight the previous dot
            self.play(Indicate(dots[t - 1], color=WHITE, scale_factor=1.45),
                      run_time=0.30)

            # 2. Memory component: dashed line from (t-1, y[t-1]) → (t, φ·y[t-1])
            mem_line = DashedLine(
                ax.c2p(t - 1, float(y[t - 1])), ax.c2p(t, pred_y),
                color=C_MEM, stroke_width=2.2, dash_length=0.08,
            )
            pred_dot = Dot(ax.c2p(t, pred_y), color=C_MEM, radius=0.07)
            self.play(Create(mem_line), GrowFromCenter(pred_dot), run_time=0.40)

            # 3. Noise component: arrow from φ·y[t-1] → y[t]
            noise_delta = float(y[t]) - pred_y
            fade_list = [mem_line, pred_dot]
            if abs(noise_delta) > 0.05:
                noise_arr = Arrow(
                    ax.c2p(t, pred_y), ax.c2p(t, float(y[t])),
                    buff=0, color=C_NOIS, stroke_width=2.2,
                    max_tip_length_to_length_ratio=0.4,
                )
                self.play(GrowArrow(noise_arr), run_time=0.35)
                fade_list.append(noise_arr)

            # 4. Final dot + connecting segment
            new_dot = Dot(ax.c2p(t, float(y[t])), color=C_SER, radius=0.09)
            new_seg = Line(
                ax.c2p(t - 1, float(y[t - 1])), ax.c2p(t, float(y[t])),
                color=C_SER, stroke_width=2.2,
            )
            self.play(GrowFromCenter(new_dot), Create(new_seg), run_time=0.38)

            # 5. Fade annotations, keep dot + segment
            self.play(*[FadeOut(obj) for obj in fade_list], run_time=0.28)

            dots.append(new_dot)
            segs.append(new_seg)

        self.wait(0.4)

        # Fast phase — remaining steps
        fast_dots = [Dot(ax.c2p(t, float(y[t])), color=C_SER, radius=0.07)
                     for t in range(N_SLOW + 1, N_TOTAL + 1)]
        fast_segs = [Line(ax.c2p(t - 1, float(y[t - 1])), ax.c2p(t, float(y[t])),
                          color=C_SER, stroke_width=2.0)
                     for t in range(N_SLOW + 1, N_TOTAL + 1)]
        dots.extend(fast_dots)
        segs.extend(fast_segs)

        self.play(
            LaggedStart(*[GrowFromCenter(d) for d in fast_dots], lag_ratio=0.08),
            LaggedStart(*[Create(s)         for s in fast_segs], lag_ratio=0.08),
            run_time=1.4,
        )
        self.wait(0.8)

        # ════════════════════════════════════════════════════════════════════
        #  Phase 3 — Compare three φ values
        # ════════════════════════════════════════════════════════════════════
        phase2 = VGroup(ax, ax_title, zero_line, legend, *dots, *segs)
        self.play(FadeOut(phase2), run_time=0.7)

        comp_title = (
            Text("Effect of \u03c6 on AR(1) behaviour", font_size=18,
                 color=C_DIM, weight=BOLD)
            .to_edge(UP, buff=0.35)
        )
        self.play(FadeIn(comp_title), run_time=0.4)

        x_pos = [-4.5, 0.0, 4.5]
        comp_axes = []
        for xs, cs in zip(x_pos, comp_series):
            yc_min = float(cs.min()) - 0.2
            yc_max = float(cs.max()) + 0.2
            comp_axes.append(
                Axes(
                    x_range=[0, N_COMP - 1, 10],
                    y_range=[yc_min, yc_max, 1.0],
                    x_length=3.8, y_length=2.4,
                    **_AX,
                ).move_to([xs, -0.6, 0])
            )

        self.play(
            LaggedStart(*[Create(a) for a in comp_axes], lag_ratio=0.15),
            run_time=0.8,
        )

        for ax_c, phi, lbl_str, tag_str, col, cs in zip(
            comp_axes, PHI_VALS, PHI_LABELS, PHI_TAGS, PHI_COLORS, comp_series
        ):
            curve = ax_c.plot(
                lambda t, s=cs: float(np.interp(t, np.arange(N_COMP), s)),
                x_range=[0, N_COMP - 1], color=col, stroke_width=1.9,
            )
            lbl_phi = (
                Text(lbl_str, font_size=14, color=col, weight=BOLD)
                .next_to(ax_c, UP, buff=0.18)
            )
            lbl_tag = (
                Text(tag_str, font_size=12, color=C_DIM)
                .next_to(ax_c, DOWN, buff=0.18)
            )
            self.play(Create(curve), FadeIn(lbl_phi), FadeIn(lbl_tag), run_time=0.9)

        self.wait(2.5)
