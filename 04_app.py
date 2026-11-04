"""
数据看板：中国体检队列 · BMI 与糖尿病风险（Streamlit）

这是数据分析项目的"门面"，面试时可现场打开演示。
数据：瑞慈体检 2010–2016 中国 11 城体检队列（CC0），211,833 人。

结构遵循"结论先行"：顶部先给核心结论，再往下是证据图，最后才是明细。

运行：streamlit run 04_app.py
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
from matplotlib import font_manager
import matplotlib.pyplot as plt

matplotlib.use("Agg")

BASE_DIR = Path(__file__).resolve().parent


def _setup_cn_font():
    """跨平台中文字体：本地 Windows 用雅黑/黑体，云端 Linux 用 apt 装的开源字体。"""
    candidates = [
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        "/System/Library/Fonts/PingFang.ttc",
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
    plt.rcParams["axes.unicode_minus"] = False
    return None


CN_FONT = _setup_cn_font()

st.set_page_config(page_title="中国队列 · BMI 与糖尿病风险", page_icon="🩺",
                   layout="wide")

# 配色：正常淡、肥胖深，让"风险随体重上升"一眼可见
COLORS = ["#9FD8C0", "#5DCAA5", "#1D9E75", "#0F6E56"]
GROUP_ORDER = ["偏瘦", "正常", "超重", "肥胖"]


@st.cache_data
def load_data():
    # 优先读精简 parquet（~2MB，进 Git 供云端用）；本地若无则回退读完整 CSV
    parquet_path = BASE_DIR / "data" / "china_app.parquet"
    csv_path = BASE_DIR / "data" / "china_analysis.csv"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(
            "找不到数据文件。请先运行 `python prepare_app_data.py` 生成 "
            "data/china_app.parquet。"
        )
    df["male"] = (df["gender"] == 1).astype(int)
    # 保证 bmi_cat 是分类，按中国切点 24/28
    if "bmi_cat" not in df.columns:
        df["bmi_cat"] = pd.cut(df["bmi"], [0, 18.5, 24, 28, 100],
                               labels=GROUP_ORDER, right=False)
    return df


df = load_data()

# ---------- 标题 ----------
st.title("🩺 中国体检队列 · BMI 与糖尿病风险")
st.caption("数据：瑞慈体检 2010–2016 中国 11 城 ｜ 211,833 人 ｜ "
           "新发糖尿病 4,174 例（6.31/1000 人年）｜ 作者：陈嘉岩")

# ================= 核心结论（结论先行） =================
st.markdown("### 核心结论")
c1, c2, c3 = st.columns(3)
c1.metric("BMI 每 +1 的糖尿病风险", "HR 1.21", "95%CI 1.20–1.22")
c2.metric("肥胖 vs 正常（发病率）", "20.2 vs 2.9", "约 7 倍", delta_color="off")
c3.metric("年轻人效应更强", "20-29岁 HR 1.37", "70+ 岁仅 1.10", delta_color="off")

st.info(
    "**一句话**：BMI 越高，新发糖尿病风险越高（每 +1 风险 +21%），"
    "且这一效应随年龄递减——年轻人的 BMI 风险远高于老年人。"
    "结论经 6 条敏感性分析验证稳健。"
)

st.divider()

# ---------- 侧边栏筛选 ----------
st.sidebar.header("🔍 筛选条件")
age_min, age_max = int(df["age"].min()), int(df["age"].max())
age_range = st.sidebar.slider("年龄范围（岁）", age_min, age_max,
                              (age_min, age_max))

bmi_min, bmi_max = float(df["bmi"].min()), float(df["bmi"].max())
bmi_range = st.sidebar.slider("BMI 范围", bmi_min, bmi_max, (bmi_min, bmi_max))

sites = sorted(df["site"].unique())
site_sel = st.sidebar.multiselect("站点（默认全部）", sites, default=sites)

mask = (
    df["age"].between(*age_range)
    & df["bmi"].between(*bmi_range)
    & df["site"].isin(site_sel)
)
fdf = df[mask]

if len(fdf) == 0:
    st.warning("当前筛选条件下没有数据，请放宽条件。")
    st.stop()

# ---------- 指标卡 ----------
st.markdown("### 筛选后概况")
m1, m2, m3, m4 = st.columns(4)
m1.metric("样本数", f"{len(fdf):,} 人")
m2.metric("发病数", f"{int(fdf['dm_censor'].sum()):,} 例")
m3.metric("发病率", f"{fdf['dm_censor'].sum() / fdf['followup_years'].sum() * 1000:.2f} /1000 人年")
m4.metric("平均 BMI", f"{fdf['bmi'].mean():.1f}")

st.divider()

# ---------- 图 1：BMI 分组发病率 ----------
st.markdown("### BMI 分组与新发糖尿病发病率")

# 每组算每 1000 人年发病率
def rate_ci(events, py):
    if py == 0:
        return 0.0, 0.0, 0.0
    r = events / py * 1000
    # Poisson 精确 CI
    from scipy.stats import chi2
    lo = chi2.ppf(0.025, 2 * events) / 2 / py * 1000 if events > 0 else 0
    hi = chi2.ppf(0.975, 2 * events + 2) / 2 / py * 1000
    return r, lo, hi

fig, ax = plt.subplots(figsize=(9, 5))
rates, errs, labels_n = [], [], []
for i, grp in enumerate(GROUP_ORDER):
    g = fdf[fdf["bmi_cat"] == grp]
    r, lo, hi = rate_ci(g["dm_censor"].sum(), g["followup_years"].sum())
    rates.append(r)
    errs.append([r - lo, hi - r])
    labels_n.append(f"{grp}\n(n={len(g):,})")

x = range(len(GROUP_ORDER))
ax.bar(x, rates, yerr=[np.array([e[0] for e in errs]), np.array([e[1] for e in errs])],
       capsize=5, color=COLORS, edgecolor="white", linewidth=1.2)
ax.set_xticks(list(x))
ax.set_xticklabels(labels_n, fontsize=10)
ax.set_ylabel("新发糖尿病（每 1000 人年）", fontsize=11)
ax.set_title("BMI 越高，糖尿病发病率越高（梯度干净，区间不重叠）", fontsize=12)
ax.grid(axis="y", alpha=0.25, linewidth=0.6)
ax.set_axisbelow(True)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
st.pyplot(fig)
plt.close(fig)

st.divider()

# ---------- 图 2：年龄分层 HR（森林图） ----------
st.markdown("### BMI 效应随年龄递减（效应修饰）")

# 用第 6 步算好的年龄分层 HR（写死，避免看板里重跑 Cox 拖慢）
age_hr = [
    ("20-29", 1.37, 1.31, 1.43),
    ("30-39", 1.31, 1.29, 1.34),
    ("40-49", 1.27, 1.25, 1.29),
    ("50-59", 1.19, 1.17, 1.21),
    ("60-69", 1.13, 1.10, 1.15),
    ("70+", 1.10, 1.07, 1.13),
]

fig, ax = plt.subplots(figsize=(9, 5.2))
y = list(range(len(age_hr)))[::-1]
for yi, (lab, hr, lo, hi) in zip(y, age_hr):
    ax.plot([lo, hi], [yi, yi], color="#2D6A4F", linewidth=2, zorder=3)
    ax.plot(hr, yi, "o", color="#1B4332", markersize=9, zorder=4)
    ax.text(hi + 0.03, yi, f"{hr:.2f}", va="center", fontsize=10,
            color="#1B4332", fontweight="medium")
ax.axvline(1.0, color="#C9A227", linestyle="--", linewidth=1.2, zorder=2)
ax.set_yticks(y)
ax.set_yticklabels([a[0] for a in age_hr], fontsize=11)
ax.set_xlabel("BMI 每 +1 kg/m² 的风险比（HR，95%CI）", fontsize=11)
ax.set_title("年轻人 BMI 的糖尿病风险是老年人的 1.24 倍", fontsize=12)
ax.set_xlim(0.9, 1.6)
ax.grid(axis="x", alpha=0.25, linewidth=0.6)
ax.set_axisbelow(True)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
st.pyplot(fig)
plt.close(fig)

st.caption("BMI × 年龄交互检验：LRT chi²=210.8，p=9.35e-48，效应修饰是铁证而非边缘显著。")

st.divider()

# ---------- 图 3：年龄 × BMI 分层发病率（效应修饰的第二个视角） ----------
st.markdown("### 各年龄组中，BMI 的绝对风险差反向走")

# 相对风险（率比）随年龄下降，但绝对风险差（能多防几例）随年龄上升
# 这是「相对 vs 绝对」的经典背离，是比 HR 更贴近决策的东西
AGE_ORDER = ["20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
sub = fdf[fdf["bmi_cat"].notna() & fdf["age_grp"].notna()]
piv = sub.groupby(["age_grp", "bmi_cat"], observed=True).apply(
    lambda v: v["dm_censor"].sum() / v["followup_years"].sum() * 1000,
    include_groups=False,
).unstack()
piv = piv.reindex(index=AGE_ORDER, columns=GROUP_ORDER)

fig, ax = plt.subplots(figsize=(9, 5))
xs = np.arange(len(AGE_ORDER))
for i, grp in enumerate(GROUP_ORDER):
    ax.plot(xs, piv[grp].values, marker="o", markersize=5, linewidth=2,
            color=COLORS[i], label=grp)
ax.set_xticks(xs)
ax.set_xticklabels(AGE_ORDER, fontsize=11)
ax.set_xlabel("年龄组（岁）", fontsize=11)
ax.set_ylabel("新发糖尿病（每 1000 人年）", fontsize=11)
ax.set_title("率比随年龄下降，但绝对风险差随年龄上升", fontsize=12)
ax.legend(title="BMI 分组", frameon=False, fontsize=9, title_fontsize=9)
ax.grid(alpha=0.25, linewidth=0.6)
ax.set_axisbelow(True)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
st.pyplot(fig)
plt.close(fig)

st.caption(
    "同一数据的两副面孔：**相对风险**（肥胖/正常的率比）从 20-29 岁的 18.3 倍"
    "降到 70+ 岁的 2.2 倍；但**绝对风险差**（肥胖多出的病例数）却从 +7.2 升到 +22.8。"
    "想找高危个体→优先筛年轻人；想防住最多病例→优先干预老年人。"
)

st.divider()

# ---------- 图 4：基线空腹血糖分布（体现审计发现：截断在 6.99） ----------
st.markdown("### 基线空腹血糖分布（已剔除基线糖尿病患者）")

fig, ax = plt.subplots(figsize=(9, 4.5))
# 用发病/未发病两组对比，同时暴露"截断在 6.99"；双 Y 轴解决量级差太大
fpv_dm = fdf.loc[fdf["dm_censor"] == 1, "fpg"].dropna()
fpv_nd = fdf.loc[fdf["dm_censor"] == 0, "fpg"].dropna()
ax.hist(fpv_nd, bins=40, range=(3, 7), color="#9FD8C0",
        label="未发病（左轴）", alpha=0.85, edgecolor="white", linewidth=0.3)
ax.set_xlabel("空腹血糖 FPG（mmol/L）", fontsize=11)
ax.set_ylabel("未发病人数", fontsize=11, color="#2D6A4F")
ax.tick_params(axis="y", labelcolor="#2D6A4F")
ax2 = ax.twinx()
ax2.hist(fpv_dm, bins=40, range=(3, 7), color="#0F6E56",
         label="发病（右轴）", alpha=0.85, edgecolor="white", linewidth=0.3)
ax2.set_ylabel("发病人数", fontsize=11, color="#1B4332")
ax2.tick_params(axis="y", labelcolor="#1B4332")
ax.axvline(5.6, color="#C9A227", linestyle="--", linewidth=1.2)
ax.text(5.62, ax.get_ylim()[1] * 0.88, "糖尿病前期 5.6", color="#8a6d1a",
        fontsize=9, va="top")
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=9, loc="upper left")
ax.set_title("血糖在 6.99 处被截断——原研究已剔除基线糖尿病人", fontsize=12)
ax.grid(axis="y", alpha=0.25, linewidth=0.6)
ax.set_axisbelow(True)
for s in ["top"]:
    ax.spines[s].set_visible(False)
    ax2.spines[s].set_visible(False)
st.pyplot(fig)
plt.close(fig)

st.caption(
    "一个审计发现：基线 FPG 最大值恰好是 **6.99**，没有一人 ≥7.0——"
    "说明队列在入组时已剔除糖尿病患者。所以结局不能用「基线血糖」判断，"
    "只能用随访后的 FPG 或自报确诊。"
)

st.divider()

# ---------- 图 5：关键代谢指标对比（发病 vs 未发病） ----------
st.markdown("### 发病组 vs 未发病组的基线差异")

box_feats = [
    ("bmi", "BMI", None),
    ("fpg", "空腹血糖", None),
    ("sbp", "收缩压", None),
    ("dbp", "舒张压", None),
    ("tg", "甘油三酯", 0.0),
    ("tc", "总胆固醇", None),
]
fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
for ax, (col, label, log_lo) in zip(axes.flat, box_feats):
    data = [
        fdf.loc[fdf["dm_censor"] == 0, col].dropna().values,
        fdf.loc[fdf["dm_censor"] == 1, col].dropna().values,
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
fig.suptitle("发病组在 BMI、血糖、血压、血脂上普遍更高", fontsize=12)
fig.subplots_adjust(top=0.90)
st.pyplot(fig)
plt.close(fig)

st.caption("箱线图隐藏了极端值（showfliers=False），聚焦中位数的组间差异，避免被离群点带偏。")

st.divider()

# ---------- 分组统计表 ----------
st.markdown("### 各 BMI 组基线特征")
feats = ["age", "bmi", "sbp", "dbp", "fpg", "tg", "tc", "followup_years"]
grouped = fdf.groupby("bmi_cat", observed=True)[feats].median().T
grouped.columns = GROUP_ORDER
grouped = grouped.round(2)
grouped.index = ["年龄（岁）", "BMI", "收缩压", "舒张压", "空腹血糖",
                 "甘油三酯", "总胆固醇", "随访（年）"]
st.dataframe(grouped.style.format("{:.2f}"), width="stretch")

st.divider()

# ---------- 结论摘要（面试 30 秒读完） ----------
st.markdown("### 📌 结论摘要")
st.markdown(
    """
| 结论 | 数字 |
|---|---|
| BMI 是独立危险因素 | 每 +1，糖尿病风险 **+21%**（HR 1.21，调整代谢指标后仍显著） |
| 效应随年龄递减 | 20-29 岁 HR **1.37** → 70+ 岁 **1.10**（交互 p=9e-48） |
| 相对 vs 绝对的背离 | 率比 18.3→2.2 倍，但绝对风险差 +7.2→+22.8 |
| 结论稳健 | 6 条敏感性分析全部扛住，HR 稳在 1.21±2% |
| 方法 | Cox 回归（有随访时间）+ 分析计划写成代码 + 一键复跑 |

**给决策者的一句话**：筛查时，BMI 对年轻人的预警价值远高于老年人；
干预时，老年人肥胖能防住更多病例。策略要根据目标是「找高危个体」还是「防最多病例」来定。
    """
)

st.divider()

# ---------- 数据明细 ----------
with st.expander("查看原始数据（前 50 行）"):
    st.dataframe(fdf.head(50), width="stretch")

st.caption("⚠️ 说明：观察性队列研究，结论为关联而非因果。"
           "结局定义 = FPG≥7.0 或自报医生诊断。")
