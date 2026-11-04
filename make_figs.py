# -*- coding: utf-8 -*-
"""单独渲染看板里新增的两张图，用于放 GitHub README。

产出：
  images/china_fig4_血糖分布.png
  images/china_fig5_箱线图.png
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent


def _setup_cn_font():
    candidates = [
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            font_manager.fontManager.addfont(path)
            name = font_manager.FontProperties(fname=path).get_name()
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return name
        except Exception:
            continue
    return None


_setup_cn_font()
COLORS = ["#9FD8C0", "#5DCAA5", "#1D9E75", "#0F6E56"]
GROUP_ORDER = ["偏瘦", "正常", "超重", "肥胖"]

df = pd.read_csv(BASE_DIR / "data" / "china_analysis.csv")

# ---------- 图 4：基线血糖分布（双 Y 轴，解决发病组量级太小看不清） ----------
fig, ax = plt.subplots(figsize=(9, 4.5))
fpv_dm = df.loc[df["dm_censor"] == 1, "fpg"].dropna()
fpv_nd = df.loc[df["dm_censor"] == 0, "fpg"].dropna()
# 未发病组（左轴，量大）
ax.hist(fpv_nd, bins=40, range=(3, 7), color="#9FD8C0",
        label="未发病（左轴）", alpha=0.85, edgecolor="white", linewidth=0.3)
ax.set_xlabel("空腹血糖 FPG（mmol/L）", fontsize=11)
ax.set_ylabel("未发病人数", fontsize=11, color="#2D6A4F")
ax.tick_params(axis="y", labelcolor="#2D6A4F")
# 发病组（右轴，量小但关键）
ax2 = ax.twinx()
ax2.hist(fpv_dm, bins=40, range=(3, 7), color="#0F6E56",
         label="发病（右轴）", alpha=0.85, edgecolor="white", linewidth=0.3)
ax2.set_ylabel("发病人数", fontsize=11, color="#1B4332")
ax2.tick_params(axis="y", labelcolor="#1B4332")
ax.axvline(5.6, color="#C9A227", linestyle="--", linewidth=1.2)
ax.text(5.62, ax.get_ylim()[1] * 0.88, "糖尿病前期 5.6", color="#8a6d1a",
        fontsize=9, va="top")
# 合并图例
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=9, loc="upper left")
ax.set_title("基线血糖在 6.99 处被截断——原研究已剔除基线糖尿病人", fontsize=12)
ax.grid(axis="y", alpha=0.25, linewidth=0.6)
ax.set_axisbelow(True)
for s in ["top"]:
    ax.spines[s].set_visible(False)
for s in ["top"]:
    ax2.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(BASE_DIR / "images" / "china_fig4_血糖分布.png", dpi=200)
plt.close(fig)

# ---------- 图 5：发病 vs 未发病箱线图 ----------
box_feats = [
    ("bmi", "BMI", None),
    ("fpg", "空腹血糖", None),
    ("sbp", "收缩压", None),
    ("dbp", "舒张压", None),
    ("tg", "甘油三酯", 0.0),
    ("tc", "总胆固醇", None),
]
fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
for ax, (col, label, _lo) in zip(axes.flat, box_feats):
    data = [
        df.loc[df["dm_censor"] == 0, col].dropna().values,
        df.loc[df["dm_censor"] == 1, col].dropna().values,
    ]
    bp = ax.boxplot(data, tick_labels=["未发病", "发病"], patch_artist=True,
                    widths=0.5, showfliers=False)
    for patch, color in zip(bp["boxes"], ["#9FD8C0", "#0F6E56"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(label, fontsize=11)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
# 用 constrained_layout 避免 suptitle 被裁切
fig.suptitle("发病组在 BMI、血糖、血压、血脂上普遍更高", fontsize=12)
fig.subplots_adjust(top=0.90)
fig.savefig(BASE_DIR / "images" / "china_fig5_箱线图.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print("已生成：")
print("  images/china_fig4_血糖分布.png")
print("  images/china_fig5_箱线图.png")
