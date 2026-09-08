from __future__ import annotations

import argparse
import json
import textwrap
from collections import defaultdict
from datetime import date
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
QUALITY_DIR = REPORT_DIR / "quality"
OUTPUT_PATH = REPORT_DIR / "pipeline_summary.pdf"

FONT_PATHS = [
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\simsun.ttc"),
]


def main():
    """Generate a Chinese PDF report for the current EEG pipeline."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    output = parser.parse_args().output
    output.parent.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    font = load_font()
    bold_font = load_font(bold=True)
    summary = load_summary()

    with PdfPages(output) as pdf:
        add_cover(pdf, font, bold_font)
        add_current_results(pdf, summary, font, bold_font)
        add_pipeline_protocol(pdf, font, bold_font)
        add_quality_overview(pdf, font, bold_font)
        add_image_page(
            pdf,
            "数据量检查：train/test trial 数",
            QUALITY_DIR / "trial_counts.png",
            [
                "2a 基本为 train=288、test=288；2a subject 03 因 QC 少 1 个 test trial。",
                "2b 各 subject 的 trial 数略有差异，这是官方数据本身差异，不是 pipeline 错误。",
            ],
            font,
            bold_font,
        )
        add_image_page(
            pdf,
            "质量控制：被剔除 trial 数",
            QUALITY_DIR / "rejected_trials.png",
            [
                "全部 18 个 subject 中，仅 2a subject 03 剔除 1 个 trial。",
                "说明当前基础 QC 下没有发现大规模异常 trial。",
            ],
            font,
            bold_font,
        )
        add_two_image_page(
            pdf,
            "类别分布检查",
            QUALITY_DIR / "class_balance_2a.png",
            QUALITY_DIR / "class_balance_2b.png",
            [
                "2a 四分类整体均衡；subject 03 的 tongue 类 test 少 1 个，是 QC 剔除造成的。",
                "2b 左右手类别在每个 subject 内保持平衡。",
            ],
            font,
            bold_font,
        )
        add_two_image_page(
            pdf,
            "通道稳定性检查",
            QUALITY_DIR / "channel_std_2a.png",
            QUALITY_DIR / "channel_std_2b.png",
            [
                "颜色表示 train-only channel std / subject median std。",
                "当前没有 bad channel candidates；2b 只有 C3/Cz/C4，不做自动删通道或插值。",
            ],
            font,
            bold_font,
        )
        add_model_integration(pdf, font, bold_font)
        add_issues_and_limits(pdf, font, bold_font)
        add_future_work(pdf, font, bold_font)
        add_interface_shapes(pdf, summary, font, bold_font)
        add_handoff_page(pdf, font, bold_font)

    print(f"PDF saved: {output}")


def load_font(bold=False):
    """Load a Chinese font available on Windows."""

    for font_path in FONT_PATHS:
        if font_path.exists():
            return FontProperties(fname=str(font_path), weight="bold" if bold else "normal")
    return FontProperties(weight="bold" if bold else "normal")


def load_summary():
    """Load processed summary if available."""

    summary_path = ROOT / "processed" / "summary.json"
    if not summary_path.exists():
        return []
    return json.loads(summary_path.read_text(encoding="utf-8"))


def new_page(title, font, bold_font, subtitle=None):
    """Create a slide-like PDF page."""

    fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
    fig.text(0.055, 0.94, title, fontsize=22, fontproperties=bold_font, color="#111827")
    if subtitle:
        fig.text(0.057, 0.905, subtitle, fontsize=11, fontproperties=font, color="#6b7280")
    fig.lines.append(
        plt.Line2D([0.055, 0.945], [0.885, 0.885], transform=fig.transFigure, color="#e5e7eb")
    )
    return fig


def add_cover(pdf, font, bold_font):
    """Add cover page."""

    fig = plt.figure(figsize=(11.69, 8.27), facecolor="#f8fafc")
    fig.text(
        0.08,
        0.72,
        "BCI Competition IV Dataset 2a / 2b\nEEG 数据 Pipeline 汇报",
        fontsize=30,
        fontproperties=bold_font,
        color="#0f172a",
        linespacing=1.3,
    )
    fig.text(
        0.083,
        0.52,
        "当前成果、数据质量可视化、存在问题与后续展望",
        fontsize=16,
        fontproperties=font,
        color="#334155",
    )
    fig.text(
        0.083,
        0.42,
        "关键词：MOABB、official/session split、train-only 标准化、PyTorch DataLoader、EEGNet / EEGNeX 接口",
        fontsize=12,
        fontproperties=font,
        color="#475569",
    )
    fig.text(
        0.083,
        0.28,
        f"生成日期：{date.today().isoformat()}",
        fontsize=11,
        fontproperties=font,
        color="#64748b",
    )
    add_footer(fig, font)
    pdf.savefig(fig)
    plt.close(fig)


def add_current_results(pdf, summary, font, bold_font):
    """Add current results page."""

    fig = new_page("1. 当前已经完成的成果", font, bold_font)
    bullets = [
        "完成 Dataset 2a / Dataset 2b 的统一处理流程，覆盖全部 9 个 subject。",
        "正式数据源使用 MOABB：2a = BNCI2014_001，2b = BNCI2014_004。",
        "正式评估协议使用 official/session split，不再使用随机 80/20 作为 baseline 结果。",
        "输出统一为 X=[N,C,T]、y=[N]，保存为 train.npz / test.npz / meta.json。",
        "PyTorch DataLoader 支持 native / eegnet / eegnex 三种输入格式。",
        "防泄漏：先 split，再只用 train 计算 mean/std；test 只 transform。",
        "质量检查结果：18 个 subject 全部处理成功，仅剔除 1 个 trial，bad channel candidates = 0。",
    ]
    add_bullets(fig, bullets, x=0.075, y=0.80, font=font, size=11.2, line_gap=0.047, wrap_width=48)

    counts = count_subjects(summary)
    info_lines = [
        f"processed subject 数：2a = {counts.get('2a', 0)}，2b = {counts.get('2b', 0)}",
        "2a 输入：EEGNet/EEGNeX -> [B,1,22,1000]",
        "2b 输入：EEGNet/EEGNeX -> [B,1,3,1000]",
        "自动化测试：24 passed",
    ]
    add_info_box(
        fig,
        "交接时模型组需要知道",
        info_lines,
        font,
        bold_font,
        x=0.60,
        y=0.31,
        width=0.34,
        height=0.35,
        wrap_width=30,
    )
    add_footer(fig, font)
    pdf.savefig(fig)
    plt.close(fig)


def add_pipeline_protocol(pdf, font, bold_font):
    """Add pipeline and evaluation protocol page."""

    fig = new_page("2. Pipeline 流程与评估协议", font, bold_font)
    steps = [
        "MOABB 读取",
        "保留 EEG 通道",
        "4-38 Hz 带通",
        "0-4 s epoch",
        "裁剪到 T=1000",
        "trial QC",
        "official split",
        "train-only 标准化",
        "保存 + DataLoader",
    ]
    draw_flow(fig, steps, font, y=0.72)
    protocol = [
        "Dataset 2a：0train -> train，1test -> test。",
        "Dataset 2b：0train/1train/2train -> train，3test/4test -> test。",
        "正式 processed 数据不再随机切分 train/test。",
        "test.npz 只用于最终评估，不用于调参。",
    ]
    add_bullets(fig, protocol, x=0.075, y=0.42, font=font, size=10.8, line_gap=0.039, wrap_width=90)
    add_warning_box(
        fig,
        "防止数据泄漏的核心",
        [
            "任何需要 fit 或估计统计量的步骤，都只能基于 train。",
            "不能把 train + test 拼起来重新标准化、做 PCA/CSP、通道选择或调参。",
        ],
        font,
        bold_font,
    )
    add_footer(fig, font)
    pdf.savefig(fig)
    plt.close(fig)


def add_quality_overview(pdf, font, bold_font):
    """Add quality report overview page."""

    fig = new_page("3. 数据质量总览", font, bold_font)
    metrics = [
        ("18", "subjects checked"),
        ("1", "total rejected trials"),
        ("0", "bad channel candidates"),
        ("MOABB", "source"),
        ("official", "split"),
        ("train-only", "standardization"),
    ]
    for i, (value, label) in enumerate(metrics):
        x = 0.07 + (i % 3) * 0.30
        y = 0.67 - (i // 3) * 0.22
        add_metric_card(fig, value, label, x, y, font, bold_font)

    bullets = [
        "当前质量图用于检查数据是否明显异常，不代表模型准确率。",
        "trial 数、类别分布、坏通道候选和通道方差比例均未发现明显问题。",
        "2a subject 03 有 1 个异常 trial 被剔除，已记录在 meta.json 和 quality_summary.csv。",
    ]
    add_bullets(fig, bullets, x=0.075, y=0.25, font=font)
    add_footer(fig, font)
    pdf.savefig(fig)
    plt.close(fig)


def add_image_page(pdf, title, image_path, notes, font, bold_font):
    """Add one image and notes."""

    fig = new_page(title, font, bold_font)
    add_image(fig, image_path, [0.08, 0.25, 0.84, 0.58])
    add_bullets(fig, notes, x=0.08, y=0.18, font=font, bullet_color="#0f766e")
    add_footer(fig, font)
    pdf.savefig(fig)
    plt.close(fig)


def add_two_image_page(pdf, title, image_path_a, image_path_b, notes, font, bold_font):
    """Add two stacked images and notes."""

    fig = new_page(title, font, bold_font)
    add_image(fig, image_path_a, [0.08, 0.52, 0.84, 0.31])
    add_image(fig, image_path_b, [0.08, 0.19, 0.84, 0.31])
    add_bullets(fig, notes, x=0.08, y=0.13, font=font, size=10, bullet_color="#0f766e")
    add_footer(fig, font)
    pdf.savefig(fig)
    plt.close(fig)


def add_model_integration(pdf, font, bold_font):
    """Show verified model integration results and their practical limits."""
    report = json.loads((REPORT_DIR / "integration/smoke_test_models.json").read_text(encoding="utf-8"))
    fig = new_page("新增成果：模型接入验证与数据汇总", font, bold_font)
    version = report["versions"]["braindecode"]
    lines = [
        f"实现：Braindecode {version} 的 EEGNet / EEGNeX，固定版本和默认网络参数。",
        f"覆盖：2a、2b 的全部 18 个被试，{report['cases_passed']} 个被试/模型组合通过。",
        "每个组合读取一个真实训练 batch，完成前向、交叉熵损失、反向和 Adam 更新。",
        "检查输出维度、损失和梯度是否有限，并确认模型参数确实发生更新。",
        "训练示例支持被试循环；每次重新创建模型和优化器，不读取 test。",
        "数据汇总表包含 18 行：样本数量、类别分布、shape、划分和预处理参数。",
    ]
    add_bullets(fig, lines, 0.075, 0.81, font, size=11.2, line_gap=0.062, wrap_width=88)
    add_shape_table(
        fig,
        ["数据集", "Pipeline batch", "后端实际输入", "模型输出", "自动化测试"],
        [["2a", "[8,1,22,1000]", "[8,22,1000]", "[8,4]", "24 passed"],
         ["2b", "[8,1,3,1000]", "[8,3,1000]", "[8,2]", "24 passed"]],
        font, bold_font, 0.075, 0.27, 0.85, 0.16,
    )
    fig.text(0.09, 0.19, "适配层只移除单例维度 1，不补零、不改变 EEG 通道或时间点。",
             fontsize=11, fontproperties=font, color="#0f766e")
    fig.text(0.09, 0.125, "结论：指定模型可以接入数据完成训练步骤；尚未复现论文准确率。",
             fontsize=11, fontproperties=font, color="#9f1239")
    add_footer(fig, font)
    pdf.savefig(fig)
    plt.close(fig)


def add_issues_and_limits(pdf, font, bold_font):
    """Add current issues and limitations page."""

    fig = new_page("4. 当前存在的问题与边界", font, bold_font)
    issues = [
        "已有最小训练入口和模型接入测试，但尚未完成正式训练与论文准确率复现。",
        "尚未做 ICA、自动坏通道插值、复杂伪迹去除；这是为了保持 baseline 协议清晰、可复现。",
        "2b 只有 3 个 EEG 通道，不适合做自动删通道或复杂插值。",
        "本地 GDF evaluation 需要另配真实标签；当前 local 读取流程仅用于调试。",
        "当前质量报告是统计检查，不是严格 SNR 定量评估。",
        "后续如果加入 CSP/PCA/通道选择等 fit 型步骤，必须继续遵守 train-only 规则。",
        "严格 validation：先从未标准化的官方 train 划分，再只用 inner_train 拟合统计量。",
    ]
    add_bullets(fig, issues, x=0.075, y=0.80, font=font)
    add_footer(fig, font)
    pdf.savefig(fig)
    plt.close(fig)


def add_future_work(pdf, font, bold_font):
    """Add future work page."""

    fig = new_page("5. 后续展望", font, bold_font)
    groups = {
        "交接后优先": [
            "接收方按 README 安装可选模型依赖，在自己的环境运行 smoke test。",
            "确定复现论文、网络实现和参数，区分接口接通与正式准确率复现。",
            "用 dataset_summary.csv 记录样本数量，并说明 1 个 test trial 的 QC 剔除。",
        ],
        "中期扩展": [
            "增加严格 validation：未标准化 train 先划分，统计量只在 inner_train 拟合。",
            "接入 EEGNet / EEGNeX baseline 训练脚本，输出 subject-level accuracy。",
            "支持跨被试实验协议，但要和 within-subject baseline 分开报告。",
        ],
        "谨慎探索": [
            "2a 可探索 ICA / 坏通道修复；2b 不建议复杂通道处理。",
            "ERP 可作为可视化补充；WTC 更适合 EEG-sEMG 数据，不适合当前 2a/2b。",
            "如要声称提高 SNR，需要先定义可复现的 SNR 指标。",
        ],
    }
    y = 0.80
    for title, items in groups.items():
        fig.text(0.075, y, title, fontsize=15, fontproperties=bold_font, color="#1f2937")
        y -= 0.055
        y = add_bullets(fig, items, x=0.095, y=y, font=font, size=11, line_gap=0.042)
        y -= 0.045
    add_footer(fig, font)
    pdf.savefig(fig)
    plt.close(fig)


def add_interface_shapes(pdf, summary, font, bold_font):
    """Add final processed interface shape page."""

    fig = new_page("6. 最终接口 Shape", font, bold_font)
    bullets = [
        "processed 文件里的数组统一为 X=[N,C,T]、y=[N]；这里的 N 也可以写成 D，表示当前 split 的 trial 数。",
        "DataLoader 会按 batch_size 取出 B 个 trial，所以模型实际看到的是 batch shape。",
        "2a 和 2b 只统一接口，不强行统一通道数；C 分别保留为 22 和 3。",
    ]
    add_bullets(fig, bullets, x=0.075, y=0.80, font=font, size=11.2, line_gap=0.045, wrap_width=88)

    columns = ["数据集", "processed X", "native batch", "eegnet/eegnex batch", "label"]
    rows = [
        ["2a", "[N, 22, 1000]", "[B, 22, 1000]", "[B, 1, 22, 1000]", "y: [N] / [B]"],
        ["2b", "[N, 3, 1000]", "[B, 3, 1000]", "[B, 1, 3, 1000]", "y: [N] / [B]"],
    ]
    add_shape_table(fig, columns, rows, font, bold_font, x=0.075, y=0.46, width=0.85, height=0.20)

    shape_lines = build_processed_shape_lines(summary)
    add_info_box(
        fig,
        "真实 processed 数据",
        shape_lines,
        font,
        bold_font,
        x=0.08,
        y=0.17,
        width=0.39,
        height=0.24,
        wrap_width=36,
    )
    add_info_box(
        fig,
        "符号说明",
        [
            "N/D：当前 npz 文件里的 trial 数，不需要手动设置。",
            "B：batch_size，由训练脚本传入。",
            "1：Conv2d 输入通道维度，用于 EEGNet/EEGNeX。",
        ],
        font,
        bold_font,
        x=0.53,
        y=0.17,
        width=0.39,
        height=0.24,
        color="#f8fafc",
        wrap_width=36,
    )
    add_footer(fig, font)
    pdf.savefig(fig)
    plt.close(fig)


def add_handoff_page(pdf, font, bold_font):
    """Add final handoff page."""

    fig = new_page("7. 交接说明", font, bold_font)
    do_items = [
        "调用 get_dataloader 读取 processed 数据。",
        "EEGNet/EEGNeX 使用 eegnet 或 eegnex 格式。",
        "逐 subject 读取 train/test，按 official split 报告。",
        "严格验证：先划分，再拟合标准化。",
    ]
    dont_items = [
        "不要重新随机划分 processed。",
        "不要用 train+test 重算标准化。",
        "不要强行补齐 2a/2b 通道数。",
        "不要用 local GDF 做正式 baseline。",
    ]
    add_info_box(
        fig,
        "推荐做法",
        do_items,
        font,
        bold_font,
        x=0.08,
        y=0.42,
        width=0.39,
        height=0.37,
        wrap_width=31,
    )
    add_info_box(
        fig,
        "不要这样做",
        dont_items,
        font,
        bold_font,
        x=0.53,
        y=0.42,
        width=0.39,
        height=0.37,
        color="#fff1f2",
        wrap_width=31,
    )
    code = (
        "train_loader = get_dataloader(\n"
        "    dataset='2a', subject=1, split='train',\n"
        "    batch_size=64, model_format='eegnet'\n"
        ")"
    )
    add_code_box(fig, code, x=0.08, y=0.18, width=0.85, height=0.18, font=font)
    add_footer(fig, font)
    pdf.savefig(fig)
    plt.close(fig)


def add_bullets(
    fig,
    bullets,
    x,
    y,
    font,
    size=12,
    line_gap=0.052,
    bullet_color="#2563eb",
    wrap_width=None,
):
    """Draw wrapped bullet list."""

    current_y = y
    for bullet in bullets:
        wrapped = wrap_text(bullet, width=wrap_width or (34 if x > 0.5 else 72))
        fig.text(x, current_y, "•", fontsize=size + 2, fontproperties=font, color=bullet_color)
        fig.text(
            x + 0.018,
            current_y,
            wrapped,
            fontsize=size,
            fontproperties=font,
            color="#111827",
            va="top",
            linespacing=1.35,
        )
        current_y -= line_gap * max(1, wrapped.count("\n") + 1)
    return current_y


def wrap_text(text, width):
    """Wrap Chinese/English mixed text approximately."""

    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def add_shape_table(fig, columns, rows, font, bold_font, x, y, width, height):
    """Draw a compact shape summary table."""

    ax = fig.add_axes([x, y, width, height])
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.12, 0.20, 0.20, 0.28, 0.20],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.0, 1.8)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#bfdbfe")
        cell.set_linewidth(0.8)
        cell.set_facecolor("#eff6ff" if row == 0 else "#ffffff")
        cell.get_text().set_fontproperties(bold_font if row == 0 else font)
        cell.get_text().set_color("#0f172a")


def build_processed_shape_lines(summary):
    """Build compact shape lines from processed summary."""

    lines = []
    for dataset in ["2a", "2b"]:
        items = [
            item
            for item in summary
            if item.get("dataset") == dataset and item.get("status") == "ok"
        ]
        if not items:
            continue
        train_shapes = sorted({tuple(item["X_train_shape"]) for item in items})
        test_shapes = sorted({tuple(item["X_test_shape"]) for item in items})
        lines.append(
            f"{dataset}: train {format_shapes(train_shapes)}; test {format_shapes(test_shapes)}。"
        )
    return lines or ["未找到 processed/summary.json，按接口约定读取 X=[N,C,T]。"]


def format_shapes(shapes):
    """Format shape tuples for report text."""

    return " / ".join("[" + ",".join(str(value) for value in shape) + "]" for shape in shapes)


def add_info_box(
    fig,
    title,
    lines,
    font,
    bold_font,
    x,
    y,
    width=0.34,
    height=0.26,
    color="#eff6ff",
    wrap_width=None,
):
    """Draw rounded information box."""

    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        transform=fig.transFigure,
        linewidth=0.8,
        edgecolor="#bfdbfe",
        facecolor=color,
    )
    fig.patches.append(patch)
    fig.text(x + 0.02, y + height - 0.055, title, fontsize=13, fontproperties=bold_font, color="#1e3a8a")
    add_bullets(
        fig,
        lines,
        x=x + 0.025,
        y=y + height - 0.10,
        font=font,
        size=9.8,
        line_gap=0.036,
        wrap_width=wrap_width or max(24, int(width * 82)),
    )


def add_warning_box(fig, title, lines, font, bold_font):
    """Draw warning box."""

    add_info_box(
        fig,
        title,
        lines,
        font,
        bold_font,
        x=0.075,
        y=0.07,
        width=0.84,
        height=0.18,
        color="#fff7ed",
        wrap_width=92,
    )


def add_metric_card(fig, value, label, x, y, font, bold_font):
    """Draw one metric card."""

    patch = FancyBboxPatch(
        (x, y),
        0.24,
        0.14,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        transform=fig.transFigure,
        linewidth=0.8,
        edgecolor="#cbd5e1",
        facecolor="#f8fafc",
    )
    fig.patches.append(patch)
    fig.text(x + 0.025, y + 0.077, value, fontsize=21, fontproperties=bold_font, color="#0f172a")
    fig.text(x + 0.025, y + 0.035, label, fontsize=10, fontproperties=font, color="#64748b")


def draw_flow(fig, steps, font, y):
    """Draw simple pipeline flow."""

    x0 = 0.075
    box_w = 0.145
    gap = 0.027
    row_gap = 0.15
    max_per_row = 5
    for i, step in enumerate(steps):
        if i < max_per_row:
            row = 0
            col = i
        else:
            row = 1
            col = max_per_row - 1 - (i - max_per_row)
        x = x0 + col * (box_w + gap)
        row_y = y - row * row_gap
        patch = FancyBboxPatch(
            (x, row_y),
            box_w,
            0.10,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            transform=fig.transFigure,
            linewidth=0.8,
            edgecolor="#93c5fd",
            facecolor="#eff6ff",
        )
        fig.patches.append(patch)
        fig.text(
            x + box_w / 2,
            row_y + 0.052,
            wrap_text(step, 8),
            fontsize=9,
            fontproperties=font,
            color="#1e3a8a",
            ha="center",
            va="center",
            linespacing=1.2,
        )
        is_row_end = i == max_per_row - 1
        is_last = i == len(steps) - 1
        if not is_last and row == 0 and not is_row_end:
            fig.text(x + box_w + 0.006, row_y + 0.044, "→", fontsize=12, color="#64748b")
        elif not is_last and row == 0 and is_row_end:
            fig.text(x + box_w / 2, row_y - 0.035, "↓", fontsize=13, color="#64748b", ha="center")
        elif not is_last and row == 1:
            fig.text(x - 0.018, row_y + 0.044, "←", fontsize=12, color="#64748b")


def add_image(fig, image_path, rect):
    """Add an image to a figure."""

    ax = fig.add_axes(rect)
    ax.axis("off")
    image = mpimg.imread(image_path)
    ax.imshow(image)


def add_code_box(fig, code, x, y, width, height, font):
    """Draw simple code snippet box."""

    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        transform=fig.transFigure,
        linewidth=0.8,
        edgecolor="#cbd5e1",
        facecolor="#f8fafc",
    )
    fig.patches.append(patch)
    fig.text(x + 0.025, y + height - 0.045, code, fontsize=11, fontproperties=font, color="#0f172a", va="top")


def count_subjects(summary):
    """Count ok subjects by dataset."""

    counts = defaultdict(int)
    for item in summary:
        if item.get("status") == "ok":
            counts[item.get("dataset")] += 1
    return counts


def add_footer(fig, font):
    """Add a simple footer."""

    fig.text(
        0.055,
        0.035,
        "EEG BCI Pipeline | Dataset 2a / 2b | MOABB + official/session split",
        fontsize=8.5,
        fontproperties=font,
        color="#94a3b8",
    )


if __name__ == "__main__":
    main()
