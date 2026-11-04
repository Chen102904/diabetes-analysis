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
    df = pd.read_csv(BASE_DIR / "data" / "china_analysis.csv")
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

# ---------- 数据明细 ----------
with st.expander("查看原始数据（前 50 行）"):
    st.dataframe(fdf.head(50), width="stretch")

st.caption("⚠️ 说明：观察性队列研究，结论为关联而非因果。"
           "结局定义 = FPG≥7.0 或自报医生诊断。")
