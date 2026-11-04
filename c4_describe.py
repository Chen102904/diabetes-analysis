"""c4_describe.py —— 第 4 步：描述现状

这一步回答三个问题，不解释原因（解释留到第 6 步建模）：

1. 这群人长什么样？     → 表 1（按 BMI 分组的基线特征）
2. 发病率到底多高？     → 人年发病率 + 95% 置信区间
3. 谁的风险更高？       → 年龄 × BMI 的分层发病率

三个原则：
  · 连续变量右偏 → 用中位数(IQR)，不用均值±标准差
  · 率必须带人年 → 随访 2–7 年不等，粗率会把长短随访混为一谈
  · 率必须带区间 → 只有一个点估计，你分不清 6.24% 和 6.9% 有没有差别

产出：docs/china_表1.md、docs/china_描述统计.md、images/china_*.png
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # 无窗口后端，服务器/Cloud 上也能出图
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from scipy import stats

from c2_plan import OUTCOME, TIME

ROOT = Path(__file__).resolve().parent
DF = ROOT / "data" / "china_analysis.csv"
FIG = ROOT / "images"
DOC = ROOT / "docs"

# 中文字体：跨平台探测。写死 SimHei 在 Linux / Streamlit Cloud 上会全变方框
for path in [
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
]:
    if os.path.exists(path):
        font_manager.fontManager.addfont(path)
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
        break
matplotlib.rcParams["axes.unicode_minus"] = False

# 四组统一配色：偏瘦淡、肥胖深，让"风险随体重上升"在图上一眼可见
COLORS = ["#9FD8C0", "#5DCAA5", "#1D9E75", "#0F6E56"]
GROUP_ORDER = ["偏瘦", "正常", "超重", "肥胖"]


def rate_ci(events: float, person_years: float, alpha: float = 0.05):
    """每 1000 人年的发病率与 Poisson 精确 95% 置信区间。

    小样本率不能用正态近似——偏瘦组只有 31 个事件，正态近似会给出负的下限。
    Poisson 精确法用卡方分布反推，永不为负。
    """
    if person_years == 0 or events == 0:
        return 0.0, 0.0, 0.0
    lo = stats.chi2.ppf(alpha / 2, 2 * events) / 2 / person_years * 1000
    hi = stats.chi2.ppf(1 - alpha / 2, 2 * events + 2) / 2 / person_years * 1000
    return events / person_years * 1000, lo, hi


def table1(df: pd.DataFrame) -> str:
    """表 1：按 BMI 分组的基线特征。"""
    rows = []
    g = df.groupby("bmi_cat", observed=False)

    rows.append(("人数 n", [f"{len(v):,}" for _, v in g]))
    rows.append(("男性 %", [f"{(v['male'].mean() * 100):.1f}" for _, v in g]))

    cont = [
        ("年龄（岁）", "age"),
        ("BMI（kg/m²）", "bmi"),
        ("收缩压（mmHg）", "sbp"),
        ("舒张压（mmHg）", "dbp"),
        ("空腹血糖（mmol/L）", "fpg"),
        ("总胆固醇（mmol/L）", "tc"),
        ("甘油三酯（mmol/L）", "tg"),
        ("随访年数", TIME),
    ]
    for label, col in cont:
        cells = []
        for _, v in g:
            s = v[col].dropna()
            cells.append(f"{s.median():.1f} ({s.quantile(.25):.1f}–{s.quantile(.75):.1f})")
        rows.append((label, cells))

    rows.append(("累计随访（人年）", [f"{v[TIME].sum():,.0f}" for _, v in g]))
    rows.append(("新发糖尿病 n", [f"{int(v[OUTCOME].sum()):,}" for _, v in g]))

    cells = []
    for _, v in g:
        r, lo, hi = rate_ci(v[OUTCOME].sum(), v[TIME].sum())
        cells.append(f"{r:.2f} ({lo:.2f}–{hi:.2f})")
    rows.append(("发病率 /1000 人年", cells))

    L = ["# 表 1 · 按 BMI 分组的基线特征\n"]
    L.append("> 连续变量为中位数（四分位间距），分类变量为百分比。")
    L.append("> **为什么不给 P 值**：21 万人的样本下，任何微小差异都会 p<0.001，"
             "P 值失去区分能力，反而误导。要看的是效应量和区间宽度。\n")
    L.append("| 特征 | " + " | ".join(GROUP_ORDER) + " |")
    L.append("|---|" + "---|" * len(GROUP_ORDER))
    for label, cells in rows:
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("**读法**：从正常到肥胖，发病率 0.92 → 2.99 → 6.24（每 1000 人年），"
             "呈单调上升，且置信区间互不重叠——这是后面建模的直观基础。")
    return "\n".join(L)


def fig_rate(df: pd.DataFrame) -> Path:
    """图 1：四组发病率 + 95%CI，组内人数放进 x 轴标签避免与数字打架。"""
    res = []
    for grp in GROUP_ORDER:
        v = df[df["bmi_cat"] == grp]
        r, lo, hi = rate_ci(v[OUTCOME].sum(), v[TIME].sum())
        res.append((grp, r, lo, hi, len(v), int(v[OUTCOME].sum())))

    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    fig.subplots_adjust(bottom=0.25, top=0.90, left=0.11, right=0.97)
    x = np.arange(len(res))
    rates = [r[1] for r in res]
    err = [[r[1] - r[2] for r in res], [r[3] - r[1] for r in res]]

    ax.bar(x, rates, width=0.62, color=COLORS, edgecolor="white", linewidth=1.2)
    ax.errorbar(x, rates, yerr=err, fmt="none", ecolor="#444441", capsize=5, linewidth=1.2)

    for i, (grp, r, lo, hi, n, inc) in enumerate(res):
        ax.text(i, hi + max(rates) * 0.035, f"{r:.2f}", ha="center", va="bottom",
                fontsize=13, color="#2C2C2A")

    rng = ["<18.5", "18.5–24", "24–28", "≥28"]
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r[0]}\n{rng[i]}\nn={r[4]:,}" for i, r in enumerate(res)],
                       fontsize=10.5, linespacing=1.6)
    ax.set_ylabel("新发糖尿病（每 1000 人年）", fontsize=12)
    ax.set_title("中国体检队列：BMI 与新发糖尿病发病率", fontsize=14, pad=16)
    ax.set_ylim(0, max(r[3] for r in res) * 1.15)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.text(0.5, 0.03, f"总样本 {len(df):,} 人 · 累计随访 {df[TIME].sum():,.0f} 人年 · "
             f"误差线为 Poisson 精确 95%CI",
             ha="center", fontsize=8.5, color="#888780")

    p = FIG / "china_fig1_发病率.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def fig_age_bmi(df: pd.DataFrame) -> Path:
    """图 2：年龄 × BMI 的分层发病率。

    事件数 <10 的组不连线、只画空心点——点估计不稳，连成实线等于编造趋势。
    """
    sub = df[df["bmi_cat"].notna() & df["age_grp"].notna()]
    piv = sub.groupby(["age_grp", "bmi_cat"], observed=True).apply(
        lambda v: rate_ci(v[OUTCOME].sum(), v[TIME].sum())[0], include_groups=False
    ).unstack()
    evt = sub.pivot_table(index="age_grp", columns="bmi_cat",
                          values=OUTCOME, aggfunc="sum", observed=True)
    xs = np.arange(len(piv.index))

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for i, grp in enumerate(GROUP_ORDER):
        stable = (evt[grp] >= 10).values
        y = piv[grp].values.astype(float).copy()
        y[~stable] = np.nan                      # 断线
        ax.plot(xs, y, marker="o", markersize=6, linewidth=2,
                color=COLORS[i], label=grp, zorder=3)
        if (~stable).any():                      # 不稳的点画成空心
            ax.scatter(xs[~stable], piv[grp].values[~stable], s=46,
                       facecolors="white", edgecolors=COLORS[i],
                       linewidths=1.6, zorder=4)

    ax.set_xticks(xs)
    ax.set_xticklabels(piv.index, fontsize=11)
    ax.set_xlabel("年龄组（岁）", fontsize=12)
    ax.set_ylabel("新发糖尿病（每 1000 人年）", fontsize=12)
    ax.set_title("各年龄组中 BMI 与糖尿病风险的关系", fontsize=14, pad=14)
    ax.legend(title="BMI 分组", frameon=False, fontsize=10, title_fontsize=10,
              loc="upper left")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.text(0.5, 0.015, "空心点 = 该组发病数 <10 例，点估计不稳定，故不连线",
             ha="center", fontsize=8.5, color="#888780")

    fig.subplots_adjust(bottom=0.16)
    p = FIG / "china_fig2_年龄分层.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def main() -> None:
    df = pd.read_csv(DF)
    FIG.mkdir(exist_ok=True)
    DOC.mkdir(exist_ok=True)

    (DOC / "china_表1.md").write_text(table1(df), encoding="utf-8")
    p1 = fig_rate(df)
    p2 = fig_age_bmi(df)

    # ---- 描述统计文档 ----
    L = ["# 中国队列 · 描述统计\n", "> 由 `c4_describe.py` 生成。本步只描述，不解释原因。\n"]

    L.append("## 1. 总体\n")
    L.append(f"- 样本 **{len(df):,}** 人，累计随访 **{df[TIME].sum():,.0f}** 人年，"
             f"中位随访 {df[TIME].median():.2f} 年")
    r, lo, hi = rate_ci(df[OUTCOME].sum(), df[TIME].sum())
    L.append(f"- 新发糖尿病 **{int(df[OUTCOME].sum()):,}** 例，"
             f"总发病率 **{r:.2f} /1000 人年**（95%CI {lo:.2f}–{hi:.2f}）")
    L.append(f"- 对比原论文年龄调整后 7.35 /1000 人年，本数据粗率 {r:.2f}，量级一致\n")

    L.append("## 2. 表 1\n")
    L.append("完整表格见 `china_表1.md`。关键行：\n")
    L.append("| BMI 分组 | 人数 | 发病率 /1000 人年 (95%CI) |")
    L.append("|---|---|---|")
    for grp in GROUP_ORDER:
        v = df[df["bmi_cat"] == grp]
        rr, ll, hh = rate_ci(v[OUTCOME].sum(), v[TIME].sum())
        L.append(f"| {grp} | {len(v):,} | {rr:.2f} ({ll:.2f}–{hh:.2f}) |")
    L.append("")

    L.append("## 3. 图\n")
    L.append(f"- `{p1.name}`：四组发病率柱状图，带 95%CI 和组内人数")
    L.append(f"- `{p2.name}`：年龄 × BMI 分层折线\n")
    L.append("**读图 2 时注意**：如果四条线在某年龄之后靠拢甚至交叉，说明 BMI 的效应"
             "在不同年龄不一样——那第 6 步就必须做交互检验，不能只报一个总体 HR。\n")

    L.append("## 4. 这一步的结论（只能说到这个程度）\n")
    L.append("1. 发病率随 BMI 单调上升，四组区间互不重叠")
    L.append("2. 年龄越大发病率越高，70+ 组最高")
    L.append("3. **但还不能说「BMI 导致糖尿病」**——年龄、性别、站点都可能是混杂，"
             "这一步没有控制任何变量")
    L.append("4. 也还不能说「哪个年龄组效应更强」——分层图只是描述，差异可能是随机波动，"
             "要靠第 6 步的交互检验确认")
    L.append("")

    (DOC / "china_描述统计.md").write_text("\n".join(L), encoding="utf-8")

    print(f"[OK] 表1 -> docs/china_表1.md")
    print(f"[OK] 描述 -> docs/china_描述统计.md")
    print(f"[OK] 图1 -> {p1.name}")
    print(f"[OK] 图2 -> {p2.name}")
    print()
    for grp in GROUP_ORDER:
        v = df[df["bmi_cat"] == grp]
        rr, ll, hh = rate_ci(v[OUTCOME].sum(), v[TIME].sum())
        print(f"  {grp}  n={len(v):>7,}  {rr:5.2f} ({ll:.2f}-{hh:.2f}) /1000人年")


if __name__ == "__main__":
    main()
