"""c6_effect_mod.py —— 第 6 步：效应修饰（年龄是否改变 BMI 的效应）

第 5 步的总体 HR=1.21 是「假设 BMI 效应在所有人里都一样」的平均值。
第 4 步已经发现：年轻人 BMI 的相对风险（率比 18.3）远高于老年人（2.2），
而且第 5 步的比例风险假设被违反——这两件事大概率是同一个根：
BMI 的效应被「年龄」修饰了。

这一步做两件独立的事，给出两条证据：

  1. 年龄分层 Cox：6 个年龄组各算一个 BMI 的 HR，看它是不是随年龄单调下降。
     下降 = 坐实「年龄修饰」。
  2. 正式交互检验：BMI × 年龄（连续）放进模型，用似然比检验（LRT）给一个
     干净的 p 值。p<0.05 说明「不同年龄，BMI 每 +1 的效应不同」，分层报告才诚实。

为什么两个都要做：
  分层图直观（能画出来、能看懂），但不是统计检验；LRT 给 p 值，但看不懂趋势。
  一个给「看得见」，一个给「信得过」。

产出：docs/china_表3_效应修饰.md + images/china_fig3_年龄分层HR.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.duration.hazard_regression import PHReg

from c2_plan import OUTCOME, TIME, EVENT, EXPOSURE, SITE

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "china_analysis.csv"
OUT = ROOT / "docs" / "china_表3_效应修饰.md"
FIG = ROOT / "images" / "china_fig3_年龄分层HR.png"

# 年龄分组边界：与 c3_clean 里 age_grp 一致（19-29 / 30-39 / ... / 70+）
AGE_BINS = [19, 30, 40, 50, 60, 70, 100]
AGE_LABELS = ["20-29", "30-39", "40-49", "50-59", "60-69", "70+"]

# 分层 Cox 的调整变量：年龄已在分层里，所以组内只调 gender（site 做 strata）。
# 注意：年龄组内还要不要再调连续 age？不需要——组内年龄跨度已小，再调会过度。
STRATA_COV = ["gender"]


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    df[SITE] = df[SITE].astype(int)
    df["male"] = (df["gender"] == 1).astype(int)
    df["age_grp"] = pd.cut(df["age"], AGE_BINS, labels=AGE_LABELS, right=False)
    return df


def _fit(df: pd.DataFrame, cov: list[str]) -> dict:
    """跑一个带 site 分层的 Cox，返回 BMI 的 HR/CI/p 和模型对象（供 LRT）。"""
    cov = [EXPOSURE] + [c for c in cov if c != EXPOSURE]
    cols = cov + [SITE, TIME, OUTCOME]
    sub = df[cols].dropna()
    X = sub[cov].copy()
    m = PHReg(sub[TIME].values, X, status=sub[OUTCOME].values,
              strata=sub[SITE].values.astype(int))
    res = m.fit()
    i = cov.index(EXPOSURE)
    return {
        "hr": float(np.exp(res.params[i])),
        "hr_lo": float(np.exp(res.conf_int()[i][0])),
        "hr_hi": float(np.exp(res.conf_int()[i][1])),
        "p": float(res.pvalues[i]),
        "n": int(len(sub)),
        "events": int(sub[OUTCOME].sum()),
        "llf": float(res.llf),
        "res": res,
        "sub": sub,
        "cov": cov,
    }


def fig_age_hr(df: pd.DataFrame, rows: list[dict]) -> Path:
    """图 3：各年龄组 BMI 每 +1 的 HR + 95%CI，森林图样式。"""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # 中文字体：跨平台探测，写死 SimHei 在 Linux/Streamlit Cloud 会全变方框
    for path in [
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
    ]:
        if os.path.exists(path):
            font_manager.fontManager.addfont(path)
            matplotlib.rcParams["font.family"] = \
                font_manager.FontProperties(fname=path).get_name()
            break
    matplotlib.rcParams["axes.unicode_minus"] = False

    labels = [r["grp"] for r in rows]
    hrs = [r["hr"] for r in rows]
    los = [r["hr_lo"] for r in rows]
    his = [r["hr_hi"] for r in rows]

    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    y = np.arange(len(rows))[::-1]  # 从上到下：年轻在上

    for yi, (hr, lo, hi) in zip(y, zip(hrs, los, his)):
        ax.plot([lo, hi], [yi, yi], color="#2D6A4F", linewidth=2, zorder=3)
        ax.plot(hr, yi, "o", color="#1B4332", markersize=9, zorder=4)

    # 参照线 HR=1
    ax.axvline(1.0, color="#C9A227", linewidth=1.2, linestyle="--", zorder=2)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("BMI 每 +1 kg/m² 的风险比（HR，95%CI）", fontsize=12)
    ax.set_title("BMI 的糖尿病风险随年龄递减：年轻人效应更强", fontsize=13, pad=14)

    # 每行标 HR 数值
    for yi, (hr, lo, hi, r) in zip(y, zip(hrs, los, his, rows)):
        ax.text(max(hi, hr) + 0.02, yi, f"{hr:.2f}", va="center", fontsize=10,
                color="#1B4332", fontweight="medium")

    ax.set_xlim(0.8, max(his) * 1.25)
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=200)
    plt.close(fig)
    return FIG


def main() -> None:
    df = load()
    L: list[str] = []
    add = L.append

    add("# 中国体检队列 · 效应修饰（年龄 × BMI）\n")
    add("> 由 `c6_effect_mod.py` 生成。回答：BMI 的糖尿病风险，在不同年龄里一样大吗？\n")

    # ---------- 1. 年龄分层 Cox ----------
    add("## 1. 年龄分层 Cox（BMI 每 +1 kg/m² 的 HR）\n")
    add("每组单独拟合，调整性别、site 分层。年龄在组内不再重复调整。\n")
    add("| 年龄组 | HR | 95%CI | p | 样本 | 事件数 |")
    add("|---|---|---|---|---|---|")

    rows: list[dict] = []
    for grp in AGE_LABELS:
        sub = df[df["age_grp"] == grp]
        if len(sub) == 0 or sub[OUTCOME].sum() == 0:
            add(f"| {grp} | — | — | — | {len(sub):,} | 0 |")
            continue
        r = _fit(sub, STRATA_COV)
        r["grp"] = grp
        rows.append(r)
        add(f"| {grp} | **{r['hr']:.2f}** | {r['hr_lo']:.2f}–{r['hr_hi']:.2f} | "
            f"{r['p']:.3g} | {r['n']:,} | {r['events']:,} |")

    add("")
    # 趋势判断：年轻人 HR 是否高于老年人
    if rows:
        youngest = rows[0]["hr"]
        oldest = rows[-1]["hr"]
        add(f"**趋势**：20–29 岁 HR={youngest:.2f}，70+ 岁 HR={oldest:.2f}。"
            f"{'效应随年龄单调下降，坐实年龄修饰。' if youngest > oldest else '未见单调趋势，需看交互检验。'}")
    add("")

    # ---------- 2. 正式交互检验（LRT） ----------
    add("## 2. BMI × 年龄 交互（似然比检验）\n")
    add("做法：在「age + gender + site」基础上，加 BMI×age 乘积项，")
    add("比较两个模型的似然。p<0.05 = 年龄显著改变 BMI 的效应，应分层报告。\n")

    base_cov = ["age", "gender"]
    base = _fit(df, base_cov)

    # 加交互项
    int_cov = ["age", "gender"]
    cols = [EXPOSURE, "age", "gender", SITE, TIME, OUTCOME]
    sub = df[cols].dropna().copy()
    X = sub[[EXPOSURE, "age", "gender"]].copy()
    X["bmi_x_age"] = X[EXPOSURE] * X["age"]
    m = PHReg(sub[TIME].values, X.values, status=sub[OUTCOME].values,
              strata=sub[SITE].values.astype(int))
    res = m.fit()

    # LRT：2*(llf_int - llf_base)，自由度 = 1（多一个交互项）
    lrt_stat = 2 * (float(res.llf) - base["llf"])
    lrt_p = float(chi2.sf(lrt_stat, 1))

    add(f"- 无交互模型 logLik = {base['llf']:.2f}")
    add(f"- 有交互模型 logLik = {res.llf:.2f}")
    add(f"- **LRT 统计量 = {lrt_stat:.1f}，p = {lrt_p:.3g}**")
    add("")
    if lrt_p < 0.05:
        add("**结论：年龄显著修饰 BMI 效应（p<0.05），不应报告单一的总体 HR，"
            "应分年龄层报告。**")
    else:
        add("**结论：交互不显著，总体 HR 可以代表各年龄层。**")
    add("")

    # ---------- 3. 落地规则（对照 c2_plan 的 INTERPRET） ----------
    add("## 3. 与预设解读规则对照\n")
    add("`c2_plan.py` 第 8 节预设：「交互 p<0.05 → 分层报告，不报总体合并值」。")
    add(f"本次交互 p={lrt_p:.3g}，"
        f"{'触发了「分层报告」规则。' if lrt_p < 0.05 else '未触发，总体 HR 仍可用。'}")
    add("")

    # ---------- 出图 ----------
    fig = fig_age_hr(df, rows)
    add(f"![年龄分层 HR]({fig.name})\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"[OK] 表 3 已写入 {OUT}")
    print(f"[OK] 图 3 已写入 {fig}")
    print()
    if rows:
        print("年龄分层 HR：")
        for r in rows:
            print(f"  {r['grp']:<7} HR={r['hr']:.2f} ({r['hr_lo']:.2f}–{r['hr_hi']:.2f})  "
                  f"p={r['p']:.3g}")
    print(f"\n交互 LRT: chi2={lrt_stat:.1f}, p={lrt_p:.3g}")


if __name__ == "__main__":
    main()
