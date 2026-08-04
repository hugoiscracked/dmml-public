"""
W09 — CNN Architecture

  Phase 1: LeNet-style pipeline
           Input image → Conv1 → Pool1 → Conv2 → Pool2 → FC → Softmax
           Each stage labelled; feature-map volumes shown as 3-D cuboids.

  Phase 2: Hierarchical features
           Three columns (early / mid / late layer) with example
           feature descriptions.

  Phase 3: Transfer learning — freeze / fine-tune
           Pre-trained backbone with frozen (blue, locked) layers and a
           small new head (amber). Two-step reveal: freeze + train head,
           then unfreeze top layers for fine-tuning.

Text uses LaTeX (Tex/MathTex); the 🔒 lock stays a Text glyph (LaTeX can't
render emoji). Labels are fit-to-box; captions use \\mbox.

Render:
  ../../env/bin/manim -pql cnn_architecture.py CNNArchitecture
  ../../env/bin/manim -qk  cnn_architecture.py CNNArchitecture   # 4K master
"""

from manim import *
import numpy as np

# ── Palette ───────────────────────────────────────────────────────────────────
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

# ── Type sizes ─────────────────────────────────────────────────────────────────
FS_TITLE = 44
FS_SUB   = 28
FS_STAGE = 19
FS_ANN   = 17
FS_BK    = 22
FS_HDR   = 22
FS_FEAT  = 19
FS_COLCAP = 19
FS_DEEPER = 18
FS_CAP   = 26
FS_BLOCK = 18
FS_STEP  = 21
FS_GNOTE = 18
FS_LEG   = 18


def _tx(s, fs, color=C_DIM, bold=False, math=False, w=None, h=None):
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


def _stack(lines, fs, colors, bold=False, buff=0.07, w=None):
    grp = VGroup()
    for i, ln in enumerate(lines):
        col = colors[i] if isinstance(colors, list) else colors
        grp.add(_tx(ln, fs, color=col, bold=bold, w=w))
    grp.arrange(DOWN, buff=buff)
    return grp


# ── Cuboid helper ─────────────────────────────────────────────────────────────
def _cuboid(w, h, d, color, fill_op=0.18, stroke_w=1.8):
    dx = d * 0.40
    dy = d * 0.22
    front = Polygon([0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0],
                    fill_color=color, fill_opacity=fill_op, stroke_color=color, stroke_width=stroke_w)
    top = Polygon([0, h, 0], [w, h, 0], [w + dx, h + dy, 0], [dx, h + dy, 0],
                  fill_color=color, fill_opacity=fill_op * 1.4, stroke_color=color, stroke_width=stroke_w)
    right = Polygon([w, 0, 0], [w + dx, dy, 0], [w + dx, h + dy, 0], [w, h, 0],
                    fill_color=color, fill_opacity=fill_op * 0.7, stroke_color=color, stroke_width=stroke_w)
    return VGroup(front, top, right)


def _cuboid_centred(w, h, d, color, **kwargs):
    cub = _cuboid(w, h, d, color, **kwargs)
    cub.shift([-w / 2, -h / 2, 0])
    return cub


class CNNArchitecture(Scene):

    def construct(self):
        self.camera.background_color = C_BG
        self._phase1()
        self._phase2()
        self._phase3()

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 1 — LeNet-style pipeline
    # ══════════════════════════════════════════════════════════════════════════

    def _phase1(self):
        title = _tx("CNN Architecture", FS_TITLE, color=C_WHITE, bold=True, w=9).to_edge(UP, buff=0.28)
        sub = _tx("Convolutions extract features, pooling shrinks maps, FC layers classify",
                  FS_SUB, color=C_DIM, w=13.2).next_to(title, DOWN, buff=0.14)
        self.play(FadeIn(title), FadeIn(sub), run_time=0.5)

        # (label, w, h, d, color, annotation) — raw strings; "||" separates lines
        stages = [
            (r"Input||$28\times28$",   0.60, 1.60, 0.10, C_WHITE, None),
            (r"Conv1||6 maps",         0.22, 1.40, 0.55, C_BLUE,  r"$3\times3$ filters||stride 1"),
            (r"Pool1||6 maps",         0.22, 0.90, 0.55, C_TEAL,  r"$2\times2$ max||$\div\,2$ size"),
            (r"Conv2||16 maps",        0.22, 0.80, 0.90, C_PURP,  r"$3\times3$ filters||stride 1"),
            (r"Pool2||16 maps",        0.22, 0.50, 0.90, C_TEAL,  r"$2\times2$ max||$\div\,2$ size"),
            (r"FC||120",               0.28, 1.00, 0.14, C_AMBER, r"flatten||$+$ ReLU"),
            (r"FC||84",                0.28, 0.70, 0.14, C_AMBER, None),
            (r"Softmax||10",           0.28, 0.42, 0.14, C_GREEN, r"class||scores"),
        ]

        n = len(stages)
        xs = np.linspace(-5.6, 5.6, n)
        y_centre = -0.30

        cuboids, lbl_mobs, ann_mobs, arrows = [], [], [], []

        for i, (lbl, w, h, d, col, ann) in enumerate(stages):
            cub = _cuboid_centred(w, h, d, col).move_to([xs[i], y_centre, 0])
            cuboids.append(cub)
            lbl_mobs.append(_stack(lbl.split("||"), FS_STAGE, col, buff=0.05, w=1.35).next_to(cub, DOWN, buff=0.20))
            if ann:
                ann_mobs.append(_stack(ann.split("||"), FS_ANN, C_DIM, buff=0.05, w=1.45).next_to(cub, UP, buff=0.24))
            if i > 0:
                x_start = xs[i - 1] + stages[i - 1][1] / 2
                x_end   = xs[i] - w / 2
                arrows.append(Arrow([x_start + 0.04, y_centre, 0], [x_end - 0.04, y_centre, 0],
                                    buff=0.0, color=C_DIM, stroke_width=1.8, max_tip_length_to_length_ratio=0.30))

        self.play(FadeIn(cuboids[0]), FadeIn(lbl_mobs[0]), run_time=0.35)

        ann_idx = 0
        for i in range(1, n):
            anims = [GrowArrow(arrows[i - 1]), FadeIn(cuboids[i]), FadeIn(lbl_mobs[i])]
            if stages[i][5]:
                anims.append(FadeIn(ann_mobs[ann_idx]))
                ann_idx += 1
            self.play(*anims, run_time=0.38)

        self.wait(0.5)

        # Bracket groupings (buff raised so brackets clear the enlarged annotations)
        bk_feat = Brace(VGroup(cuboids[1], cuboids[4]), direction=UP, color=C_BLUE, buff=1.35)
        bk_feat_lbl = _tx("Feature Extraction", FS_BK, color=C_BLUE, bold=True, w=3.4).next_to(bk_feat, UP, buff=0.08)
        bk_cls = Brace(VGroup(cuboids[5], cuboids[7]), direction=UP, color=C_AMBER, buff=1.35)
        bk_cls_lbl = _tx("Classifier", FS_BK, color=C_AMBER, bold=True, w=2.2).next_to(bk_cls, UP, buff=0.08)

        self.play(GrowFromCenter(bk_feat), FadeIn(bk_feat_lbl),
                  GrowFromCenter(bk_cls), FadeIn(bk_cls_lbl), run_time=0.55)
        self.wait(1.5)

        self._title  = title
        self._p1_all = VGroup(
            sub, bk_feat, bk_feat_lbl, bk_cls, bk_cls_lbl,
            *cuboids, *lbl_mobs, *ann_mobs, *arrows,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 2 — Hierarchical features
    # ══════════════════════════════════════════════════════════════════════════

    def _phase2(self):
        self.play(FadeOut(self._p1_all), run_time=0.40)

        sub = _tx("Hierarchical features --- deeper layers detect increasingly complex patterns",
                  FS_SUB, color=C_DIM, w=13.2).next_to(self._title, DOWN, buff=0.14)
        self.play(FadeIn(sub), run_time=0.28)

        cols = [
            {"cx": -4.2, "header": "Early layers", "color": C_BLUE,
             "features": ["Horizontal edges", "Vertical edges", "Colour blobs", "Diagonal gradients"],
             "caption": "local, low-level"},
            {"cx": 0.0, "header": "Middle layers", "color": C_PURP,
             "features": ["Textures \\& grids", "Curves \\& corners", "Repeating patterns", "Simple shapes"],
             "caption": "semi-local, mid-level"},
            {"cx": 4.2, "header": "Late layers", "color": C_AMBER,
             "features": ["Object parts", "Semantic regions", "Viewpoint invariance", "Class-specific cues"],
             "caption": "global, high-level"},
        ]

        y_header   =  1.55
        y_feat_top =  0.90
        feat_step  =  0.54
        y_caption  = -1.55
        col_w      =  2.60

        all_col_mobs = VGroup()

        for col in cols:
            cx, color = col["cx"], col["color"]
            hdr_box = RoundedRectangle(width=col_w, height=0.46, corner_radius=0.08,
                                       fill_color=color, fill_opacity=0.15, stroke_color=color, stroke_width=1.8).move_to([cx, y_header, 0])
            hdr_lbl = _tx(col["header"], FS_HDR, color=color, bold=True, w=col_w * 0.85).move_to(hdr_box)
            self.play(FadeIn(hdr_box), FadeIn(hdr_lbl), run_time=0.30)

            feat_mobs = VGroup()
            for fi, feat in enumerate(col["features"]):
                y = y_feat_top - fi * feat_step
                dot = Dot(radius=0.06, color=color).move_to([cx - 1.05, y, 0])
                txt = _tx(feat, FS_FEAT, color=C_WHITE, w=2.0).next_to(dot, RIGHT, buff=0.16)
                feat_mobs.add(VGroup(dot, txt))
            self.play(LaggedStart(*[FadeIn(f) for f in feat_mobs], lag_ratio=0.20), run_time=0.55)

            cap = _tx(col["caption"], FS_COLCAP, color=C_DIM, w=col_w).move_to([cx, y_caption, 0])
            self.play(FadeIn(cap), run_time=0.22)
            all_col_mobs.add(hdr_box, hdr_lbl, feat_mobs, cap)

        arr_ab = Arrow([-4.2 + col_w / 2 + 0.10, y_header, 0], [0.0 - col_w / 2 - 0.10, y_header, 0],
                       buff=0.0, color=C_DIM, stroke_width=1.8, max_tip_length_to_length_ratio=0.25)
        arr_bc = Arrow([0.0 + col_w / 2 + 0.10, y_header, 0], [4.2 - col_w / 2 - 0.10, y_header, 0],
                       buff=0.0, color=C_DIM, stroke_width=1.8, max_tip_length_to_length_ratio=0.25)
        deeper_a = _tx(r"deeper $\to$", FS_DEEPER, color=C_DIM, w=1.6).next_to(arr_ab, UP, buff=0.08)
        deeper_b = _tx(r"deeper $\to$", FS_DEEPER, color=C_DIM, w=1.6).next_to(arr_bc, UP, buff=0.08)
        self.play(GrowArrow(arr_ab), GrowArrow(arr_bc), FadeIn(deeper_a), FadeIn(deeper_b), run_time=0.40)

        caption = _tx(
            r"\mbox{same filters work for many tasks $\;\cdot\;$ only the final classifier is task-specific}",
            FS_CAP, color=C_DIM, w=13.0,
        ).to_edge(DOWN, buff=0.32)
        self.play(FadeIn(caption), run_time=0.35)
        self.wait(2.0)

        self._p2_all = VGroup(sub, all_col_mobs, arr_ab, arr_bc, deeper_a, deeper_b, caption)

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 3 — Transfer learning: freeze → fine-tune
    # ══════════════════════════════════════════════════════════════════════════

    def _phase3(self):
        self.play(FadeOut(self._p2_all), run_time=0.40)

        sub = _tx("Transfer learning --- reuse pre-trained features, train only the new head",
                  FS_SUB, color=C_DIM, w=13.2).next_to(self._title, DOWN, buff=0.14)
        self.play(FadeIn(sub), run_time=0.28)

        BLOCK_W = 3.00
        BLOCK_H = 0.48
        GAP     = 0.12
        CX      = 0.0
        Y_BOT   = -2.55

        frozen_layers = [
            r"Conv Block 1  (edges)",
            r"Conv Block 2  (textures)",
            r"Conv Block 3  (shapes)",
            r"Conv Block 4  (parts)",
        ]
        head_layers = [r"FC 512  (new)", r"Softmax  (new task)"]
        n_frozen = len(frozen_layers)

        def _layer_block(label, color, locked, idx):
            y = Y_BOT + idx * (BLOCK_H + GAP) + BLOCK_H / 2
            rect = RoundedRectangle(width=BLOCK_W, height=BLOCK_H, corner_radius=0.06,
                                    fill_color=color, fill_opacity=0.15, stroke_color=color, stroke_width=2.0).move_to([CX, y, 0])
            lbl = _tx(label, FS_BLOCK, color=color, w=2.2).move_to([CX - 0.28, y, 0])
            mobs = VGroup(rect, lbl)
            if locked:
                lock = Text("🔒", font_size=20).move_to([CX + BLOCK_W / 2 - 0.32, y, 0])
                mobs.add(lock)
            return mobs, y

        frozen_blocks, head_blocks = [], []

        # Step labels sit above the block stack (top block reaches y≈0.93) to avoid collision.
        step1_lbl = _tx("Step 1 --- freeze backbone, train head only", FS_STEP, color=C_BLUE, bold=True, w=5.4)\
            .to_edge(LEFT, buff=0.45).shift(UP * 2.0)
        self.play(FadeIn(step1_lbl), run_time=0.28)

        for i, lbl in enumerate(frozen_layers):
            blk, y = _layer_block(lbl, C_BLUE, locked=True, idx=i)
            frozen_blocks.append((blk, y))
            self.play(FadeIn(blk), run_time=0.22)
        for j, lbl in enumerate(head_layers):
            blk, y = _layer_block(lbl, C_AMBER, locked=False, idx=n_frozen + j)
            head_blocks.append((blk, y))
            self.play(FadeIn(blk), run_time=0.22)

        grad_note = _tx("gradients flow here only", FS_GNOTE, color=C_AMBER, w=2.9)\
            .next_to(head_blocks[-1][0], RIGHT, buff=0.30)
        no_grad_note = _tx("frozen --- no gradient", FS_GNOTE, color=C_DIM, w=2.7)\
            .next_to(frozen_blocks[0][0], RIGHT, buff=0.30)
        self.play(FadeIn(grad_note), FadeIn(no_grad_note), run_time=0.28)
        self.wait(1.2)

        step2_lbl = _tx("Step 2 --- unfreeze top layers, fine-tune with low LR", FS_STEP, color=C_GREEN, bold=True, w=6.2)\
            .next_to(step1_lbl, DOWN, buff=0.28).align_to(step1_lbl, LEFT)
        self.play(FadeIn(step2_lbl), run_time=0.28)

        unfreeze_idxs = [2, 3]
        for ui in unfreeze_idxs:
            old_blk = frozen_blocks[ui][0]
            y = frozen_blocks[ui][1]
            new_blk, _ = _layer_block(frozen_layers[ui], C_GREEN, locked=False, idx=ui)
            self.play(FadeOut(old_blk), FadeIn(new_blk), run_time=0.35)
            frozen_blocks[ui] = (new_blk, y)

        new_grad_note = _tx(r"fine-tune with lr $= 10^{-5}$", FS_GNOTE, color=C_GREEN, w=3.0)\
            .next_to(frozen_blocks[unfreeze_idxs[-1]][0], RIGHT, buff=0.30)
        self.play(FadeOut(grad_note), FadeIn(new_grad_note), run_time=0.30)
        self.wait(0.6)

        leg_items = [
            (C_BLUE,  "frozen (pre-trained)"),
            (C_GREEN, "unfrozen (fine-tune)"),
            (C_AMBER, "new head (from scratch)"),
        ]
        legend = VGroup()
        for col, txt in leg_items:
            row = VGroup(Dot(radius=0.08, color=col), _tx(txt, FS_LEG, color=col, w=3.0)).arrange(RIGHT, buff=0.12)
            legend.add(row)
        legend.arrange(DOWN, aligned_edge=LEFT, buff=0.16).to_edge(RIGHT, buff=0.45).shift(DOWN * 0.55)
        self.play(FadeIn(legend), run_time=0.30)

        caption = _tx(
            r"\mbox{transfer learning cuts training time and data $\;\cdot\;$ "
            r"fine-tuning adapts high-level features to the new task}",
            FS_CAP, color=C_DIM, w=13.2,
        ).to_edge(DOWN, buff=0.32)
        self.play(FadeIn(caption), run_time=0.35)
        self.wait(2.5)
