"""Fixture specifications for DeckWeaver regression tests.

Each FixtureSpec is the single source of truth for one test slide:
  - drives codex HTML generation (via design_brief)
  - drives expected.json auto-generation (via keywords, counts)

Adding a new fixture: append a new FixtureSpec to FIXTURE_SPECS below, then
run `python tests/tools/generate_fixtures.py --only <name>`.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FixtureSpec:
    name: str
    width: int
    height: int
    background: str           # "light" | "dark" | "gradient"
    description: str          # human-readable summary
    design_brief: str         # passed to codex; describes the slide content & layout
    must_appear_text: list[str]               # keywords required in reconstructed PPTX
    expected_image_objects: tuple[int, int]   # (min, max) inclusive
    expected_textboxes: tuple[int, int]
    tags: list[str] = field(default_factory=list)


FIXTURE_SPECS: list[FixtureSpec] = [
    FixtureSpec(
        name="light_cover_hero",
        width=1920,
        height=1080,
        background="light",
        description="Light-themed product cover with hero title, subtitle, logo and decorative shapes.",
        design_brief="""\
A clean, modern product cover slide for an open-source tool called "DeckWeaver".

CRITICAL: Use ONLY flat solid colors. No gradients anywhere — no linear-gradient,
no radial-gradient, no conic-gradient. Translucent overlays are NOT allowed
either (no opacity, no rgba with alpha < 1.0). Every fill must be a single
solid hex color.

Layout (1920x1080, light background, flat off-white #F7F8FA — no pattern, no gradient):
  - Top-left: a small monochrome logo (a stylized "DW" mark inside a rounded square,
    flat dark navy #1E2A44 background, flat white "DW" text).
  - Centered vertically, left-aligned in the left half of the slide:
      * Large hero title "DeckWeaver" — weight 800, size ~140px, flat dark navy #1E2A44.
      * Subtitle below: "把图片重建为可编辑 PPT" — weight 500, size ~52px, flat medium gray #4A5568.
      * One-line tagline below subtitle: "Local OCR · 零云端 API · 极低 Token 消耗" — size ~28px, flat accent blue #3B82F6.
  - Right half of the slide: a decorative composition of THREE non-overlapping solid-color
    geometric shapes arranged in a balanced layout (a circle, a rounded square, a triangle).
    Each shape filled with a different flat color: #3B82F6, #8B5CF6, #F87171.
    They MUST NOT overlap and MUST NOT be translucent. Place them with clear spacing.
  - Bottom-right corner: small version badge "v1.0 · 2026" — size ~22px, flat gray #94A3B8.
  - Use system fonts: -apple-system, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif.
""",
        must_appear_text=[
            "DeckWeaver",
            "可编辑 PPT",
            "Local OCR",
            "v1.0",
        ],
        expected_image_objects=(1, 6),
        expected_textboxes=(3, 8),
        tags=["light", "cover"],
    ),

    FixtureSpec(
        name="light_text_columns",
        width=1280,
        height=960,
        background="light",
        description="Two-column comparison: traditional approach vs DeckWeaver, with bullet lists.",
        design_brief="""\
A 4:3 (1280x960) comparison slide titled "为什么需要 DeckWeaver".

Layout:
  - Top: page title "为什么需要 DeckWeaver" — size ~56px, weight 700, dark gray #1F2937, left-aligned with 80px padding.
  - Thin accent bar under the title (4px tall, accent blue #3B82F6, 120px wide).
  - Main area split into two equal columns with 40px gutter, each column a soft rounded card:
      * Left card (background #FEF2F2, faint red tint): header "传统方式" with a red ✗ icon (use Unicode ✗ or an SVG cross).
        4 bullet points, each prefixed with a small red dot:
          1. 截图无法编辑，只能重新做
          2. 文字、图标都被压平成像素
          3. 修改一个字都要回到原稿
          4. PPT 协作时无法替换内容
      * Right card (background #ECFDF5, faint green tint): header "DeckWeaver" with a green ✓ icon.
        4 bullet points, each prefixed with a small green dot:
          1. 所有文字还原为可编辑文本框
          2. 图标、Logo 拆成独立图片对象
          3. 本地 OCR，无需上传云端
          4. 支持 PNG / JPG / WebP / TIFF
  - Bottom of slide: small caption "对比示意，实际效果依赖源图质量" — size ~20px, gray #9CA3AF, centered.
  - Use system fonts: -apple-system, "PingFang SC", "Hiragino Sans GB", sans-serif.
""",
        must_appear_text=[
            "为什么需要 DeckWeaver",
            "传统方式",
            "DeckWeaver",
            "截图",
            "可编辑文本框",
            "本地 OCR",
        ],
        expected_image_objects=(0, 6),
        expected_textboxes=(8, 18),
        tags=["light", "text-heavy"],
    ),

    FixtureSpec(
        name="dark_dashboard",
        width=2560,
        height=1440,
        background="dark",
        description="Dark dashboard with five-step pipeline, icons and short captions.",
        design_brief="""\
A dark, high-resolution dashboard-style slide (2560x1440) showing the DeckWeaver processing pipeline.

CRITICAL: Use ONLY flat solid colors. No gradients anywhere — no linear-gradient,
no radial-gradient, no conic-gradient, no vignette. No translucent overlays
(no opacity, no rgba with alpha < 1.0). Every fill must be a single solid hex color.

Layout:
  - Background: flat deep navy #0B1220 — no gradient, no vignette, no pattern.
  - Top: page title "DeckWeaver 处理流程" — size ~72px, weight 700, flat white #FFFFFF, centered.
  - Subtitle directly below: "从一张图片到可编辑 PPTX 的五个步骤" — size ~32px, flat gray #94A3B8, centered.
  - Center: a horizontal pipeline of FIVE step cards equally spaced, connected by right-arrow chevrons.
    Each card is a rounded rectangle, flat background #1E293B with thin solid border #334155, padding 28px.
    Each card contains:
      * A circular icon badge at top (60px diameter, FLAT solid fill #3B82F6 — no gradient),
        with a simple white symbol inside (use Unicode: 🔍, ⊞, ▤, ⎙, ✓ — one per step).
      * Step number in small caps: "STEP 01" through "STEP 05" — size ~22px, flat accent cyan #22D3EE.
      * Step title — size ~36px, weight 700, flat white.
      * One-line description — size ~22px, flat gray #CBD5E1.
    The five steps:
      01 OCR 识别   — "本地多引擎并行识别文字"
      02 元素检测   — "切分图标、Logo 和图片对象"
      03 布局重建   — "生成 layout JSON 描述"
      04 PPTX 生成 — "组装为可编辑 PowerPoint"
      05 视觉校准   — "渲染预览自动校准字号位置"
  - Between each pair of cards, draw a right-pointing chevron "›" or ">" in flat accent cyan #22D3EE, size ~48px.
  - Footer bottom-right: "github.com/GuopengLin/Image2PPT" — size ~22px, flat gray #64748B.
  - Use system fonts: -apple-system, "PingFang SC", "Hiragino Sans GB", sans-serif.
""",
        must_appear_text=[
            "DeckWeaver 处理流程",
            "OCR 识别",
            "元素检测",
            "布局重建",
            "PPTX 生成",
            "视觉校准",
        ],
        expected_image_objects=(0, 12),
        expected_textboxes=(10, 25),
        tags=["dark", "pipeline"],
    ),

    FixtureSpec(
        name="dark_features_grid",
        width=1600,
        height=1200,
        background="dark",
        description="Dark 2x3 grid of feature cards with icons, highlight numbers and short captions.",
        design_brief="""\
A dark feature-showcase slide (1600x1200, 4:3) with a 2-row by 3-column grid of feature cards.

CRITICAL: Use ONLY flat solid colors. No gradients anywhere — no linear-gradient,
no radial-gradient, no conic-gradient. No translucent overlays (no opacity,
no rgba with alpha < 1.0). No patterns. Every fill must be a single solid hex color.

Layout:
  - Background: flat very dark slate #111827 — no pattern, no gradient, no stripes.
  - Top: title "核心特性" — size ~64px, weight 800, flat white, centered.
  - Subtitle: "Six reasons teams pick DeckWeaver" — size ~28px, flat gray #9CA3AF, centered, italic.
  - 2x3 grid of feature cards, each card:
      * Rounded rectangle, flat solid background #1F2937 (NO gradient), solid 1px border #374151.
      * Padding 32px, gap between cards 28px.
      * Top: large emoji or symbol as icon (64px).
      * A bold highlight number in accent color (size ~52px, weight 800).
      * Card title under the number (size ~28px, weight 700, flat white).
      * One-line caption under the title (size ~20px, flat gray #9CA3AF).
    The six cards (use different accent colors per card, flat solid only):
      1. 🔒 "0"        本地运行     "OCR 与渲染均在本地完成"          color: green #10B981
      2. 💰 "≈0"      Token 消耗  "命令行模式无需调用大模型"          color: amber #F59E0B
      3. ⚡ "5×"       速度提升     "热模型批量复用，流水线并行"        color: cyan #06B6D4
      4. 🗂 "4+"      支持格式     "PNG / JPG / WebP / TIFF"          color: violet #8B5CF6
      5. ✏️ "100%"    可编辑      "文字与图标皆为独立对象"            color: rose #F43F5E
      6. 🌐 "MIT"    开源协议     "个人免费使用，商用可授权"          color: blue #3B82F6
  - Bottom: small footer "deckweaver.dev" — size ~22px, flat gray #6B7280, centered.
  - Use system fonts: -apple-system, "PingFang SC", "Hiragino Sans GB", sans-serif.
""",
        must_appear_text=[
            "核心特性",
            "本地运行",
            "Token",
            "速度提升",
            "支持格式",
            "可编辑",
            "开源",
        ],
        expected_image_objects=(0, 12),
        expected_textboxes=(15, 30),
        tags=["dark", "grid"],
    ),

    FixtureSpec(
        name="mixed_bilingual_rich",
        width=1600,
        height=900,
        background="light",
        description="Bilingual slide with quote block, mixed CN/EN bullets, decorative images and footer.",
        design_brief="""\
A 16:9 (1600x900) bilingual slide with a flat light background and mixed Chinese/English content.

CRITICAL: Use ONLY flat solid colors. No gradients anywhere — no linear-gradient,
no radial-gradient, no conic-gradient. No translucent overlays (no opacity,
no rgba with alpha < 1.0). No box-shadow with blur (use sharp 1px borders instead
if you want card delineation). Every fill must be a single solid hex color.

Layout:
  - Background: flat soft lavender #EEF2FF — no gradient.
  - Top-left: page title two lines:
      Line 1 (Chinese): "中英文混排示例" — size ~52px, weight 700, flat dark navy #1E293B.
      Line 2 (English): "Bilingual Layouts" — size ~32px, weight 500, flat accent blue #4F46E5, italic.
  - Top-right corner: a small DeckWeaver wordmark "DW" in a rounded badge (40px),
    flat dark navy #1E2A44 background with flat white "DW" text.
  - Middle-left area: a quote block on a flat white #FFFFFF card with a 1px solid border #E2E8F0:
      Large opening quotation mark "" in flat accent #4F46E5, size ~80px.
      Quote text (two lines):
        "DeckWeaver 让图片重新变得可编辑。"
        "Makes flattened images editable again."
      Attribution line below: "— Project README, 2026" — size ~20px, flat gray #6B7280.
  - Middle-right area: two stacked decorative cards (placeholders for product screenshots):
      Each card 320x180, rounded, 1px solid border #CBD5E1, FLAT solid fill #F8FAFC inside.
      No gradient, no shadow. Each card has a small label.
      Card 1 label: "📸 Source slide"
      Card 2 label: "📑 Reconstructed PPTX"
  - Below the quote, a 2-column bullet list:
      Left column header "适用场景" with three bullets:
        • Agent 工作流：直接调用 skill
        • 大批量识别：命令行无 token 消耗
        • 内部知识库：把图片归档重新可搜
      Right column header "Use Cases" with three bullets:
        • Slide regeneration in AI agents
        • Bulk OCR pipelines (no API cost)
        • Re-editing legacy slide screenshots
      Bullet text size ~22px, headers size ~28px weight 700, all flat colors.
  - Bottom footer: "Contact · 1015277323@qq.com · github.com/GuopengLin/Image2PPT" — size ~20px, flat gray #6B7280, centered.
  - Use system fonts: -apple-system, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif.
""",
        must_appear_text=[
            "中英文混排示例",
            "Bilingual Layouts",
            "DeckWeaver",
            "可编辑",
            "Use Cases",
            "1015277323@qq.com",
        ],
        expected_image_objects=(0, 8),
        expected_textboxes=(10, 25),
        tags=["gradient", "bilingual"],
    ),
]


def by_name(name: str) -> FixtureSpec:
    for spec in FIXTURE_SPECS:
        if spec.name == name:
            return spec
    raise KeyError(f"Unknown fixture: {name}")
