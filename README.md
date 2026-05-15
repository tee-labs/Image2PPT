<p align="center">
  <img src="assets/logo.png" alt="DeckWeaver logo" width="220">
</p>

# DeckWeaver

中文 | [English](README_EN.md)

DeckWeaver 是 Image2PPT 的项目展示名。它可以把 PPT 截图、导出的页面图片或演示稿设计稿重建为可编辑的 PowerPoint 文件：普通文字尽量还原成真正的 PPT 文本框，图标、Logo、图片和装饰元素拆成独立图片对象，简单卡片、线条、圆角框等转成原生形状。

它适合你只有图片、没有原始 `.pptx` 的场景。如果你已经有原始 PPT，请优先直接编辑原文件。

## 优点

- 几乎所有文字和图标都可以在 PowerPoint 里继续编辑、移动或替换。
- Token 消耗很低：主体流程使用本地 OCR 和图像算法，不需要把整页反复交给云端多模态模型。
- 生成速度快：批量 OCR 复用热模型，后续页面流水线自动完成。
- 无需额外云端 API：OCR、图像分割、PPTX 生成和预览检查都在本地运行。
- 支持常见位图输入：PNG、JPG/JPEG、WebP、BMP、TIF/TIFF。

## 快速开始

### 最快方式：交给 Codex / Claude Code

1. 克隆项目：

```bash
git clone https://github.com/GuopengLin/Image2PPT.git
cd Image2PPT
```

2. 用 Codex、Claude Code 或其他本地 agent 打开这个项目目录。

3. 把要转换的图片或图片文件夹告诉 agent，例如：

```text
请使用这个项目里的 skill，把 slides/ 文件夹下的所有图片转换成一个可编辑 PPT。
```

也可以指定单张图片：

```text
请使用这个项目里的 skill，把 /path/to/page_01.png 转换成可编辑 PPT。
```

首次运行时，让 agent 先执行 `bash scripts/bootstrap.sh` 安装依赖。之后它会按流程生成 OCR、布局、PPTX 和预览文件，最终结果在 `output_project/<run>/slides.pptx`。

### 手动命令

```bash
git clone https://github.com/GuopengLin/Image2PPT.git
cd Image2PPT
bash scripts/bootstrap.sh
```

`bootstrap.sh` 会安装 Python 依赖、本地 OCR 依赖、LibreOffice/Poppler 预览工具，并预下载模型缓存。macOS 和常见 Linux 发行版可直接使用；Windows 或受管环境可参考 `requirements.txt` 手动安装依赖。

准备一个源图片目录，文件名按页码命名：

```text
slides/
├── page_01.png
├── page_02.jpg
├── page_03.webp
└── page_04.tiff
```

然后运行三步：

```bash
RUN="output_project/demo_$(date +%Y%m%d_%H%M%S)"
SRC="slides"

python scripts/ocr/prepare_ocr.py \
  --source-dir "$SRC" \
  --work-dir "$RUN"

python scripts/ocr/ocr_review_apply.py --work-dir "$RUN"

python scripts/build_deck.py \
  --source-dir "$SRC" \
  --work-dir "$RUN"
```

生成结果会在：

```text
output_project/<run>/
├── slides.pptx       # 最终可编辑 PPT
├── qa.json           # PPTX 结构检查报告
├── previews/         # 预览图，用于人工比对
├── ocr/              # OCR 与可选人工复核文件
├── layouts/          # 页面布局 JSON
├── assets/           # 提取出的图片对象
└── debug/            # 调试可视化
```

如果 `build_deck.py` 提示有 OCR 不确定项，可以打开 `ocr/page_NN.ocr_review.annotated.png` 检查高亮文字，修改对应 `ocr_review.json` 的 `corrected_text` 后重新运行后两步。

## 常用参数

```bash
python scripts/ocr/prepare_ocr.py --pages 1,3,8 ...
python scripts/build_deck.py --skip-render ...
python scripts/build_deck.py --detect-tables ...
python scripts/build_deck.py --icon-review ...
```

- `--pages`：只处理指定页。
- `--skip-render`：跳过 LibreOffice 预览渲染。
- `--detect-tables`：尝试把规则表格还原为 PPT 原生表格。
- `--icon-review` / `--icon-decisions`：导出图标/文字边界判断包，便于人工复核。

## 项目结构

```text
.
├── scripts/
│   ├── ocr/          # OCR、交叉验证、复核应用
│   ├── page/         # 单页擦除文字、检测元素、生成布局
│   ├── deck/         # 合并布局并生成 PPTX
│   ├── verify/       # PPTX 检查与预览渲染
│   ├── tables/       # 可选表格识别
│   └── optional/     # 可选后处理工具
├── references/       # 布局格式与流程说明
├── agents/codex.yaml # 可选 Codex agent 展示元数据
├── SKILL.md          # 作为 Codex skill 使用时的流程说明
└── requirements.txt
```

## 注意事项

- 输入文件必须使用 `page_NN.<ext>` 命名，每页只能保留一个同页码图片文件。
- 多页 PPT 最好保持一致比例；PowerPoint 一个文件只能有一种页面尺寸。
- 复杂图表目前会优先作为可移动图片对象保留，而不是还原为可编辑数据图表。
- 生成文件默认写入 `output_project/`，该目录不会提交到 Git。

## 联系方式

商业授权、定制开发或问题反馈：1015277323@qq.com

## License

个人免费使用。商业使用、商业分发、SaaS/内部生产系统集成等场景需要先联系作者购买商业授权。详见 [LICENSE](LICENSE)。
