"""
第 7 步：统计推断

前面几步做的是「描述统计」——只说清楚手上这批数据长什么样。
这一步做的是「统计推断」——判断这些规律是真的，还是碰巧撞上的。

四件事：

  1. 组间差异检验
     患病组和未患病组在各项指标上的差距，是真存在还是抽样误差？
     用 Welch t 检验（不假设两组方差相等，更稳健），给出 p 值和 95% 置信区间。

  2. 相关性检验
     相关系数 0.493 显著不为 0 吗？
     用 Pearson r + p 值 + Fisher z 变换求置信区间。

  3. 风险量化
     指标每升高一档，患病风险翻几倍？
     用逻辑回归算出 OR 值 + 95% 置信区间（医学文献的标准写法）。

  4. 分层分析
     年龄大的人本来就容易得糖尿病，BMI 也会更高。
     那"BMI 有风险"里，有多少其实是年龄带来的？
     按年龄分层后每层单独算，看剔除混杂后效应还剩多少。

⚠️ 本脚本的核心是「结论的可信度」，不是「模型的准确率」。
   医疗场景里，一个说不清把握度的结论等于没有结论。

运行：python 07_inference.py
输出：docs/inference_report.md + 控制台摘要
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime

try:
    from scipy import stats
except ImportError:
    raise SystemExit(
        "缺少 scipy，请先安装：\n"
        "  pip install scipy -i https://mirrors.aliyun.com/pypi/simple/"
    )

try:
    import statsmodels.api as sm
    HAS_SM = True
except ImportError:
    HAS_SM = False

CLEAN = "data/diabetes_clean.csv"
OUT_DIR = "docs"
OUT_MD = os.path.join(OUT_DIR, "inference_report.md")

FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
CN = {
    "Pregnancies": "妊娠次数", "Glucose": "血糖", "BloodPressure": "舒张压",
    "SkinThickness": "皮褶厚度", "Insulin": "血清胰岛素", "BMI": "体质指数",
    "DiabetesPedigreeFunction": "家族史函数", "Age": "年龄",
}
TARGET = "Outcome"
ALPHA = 0.05


def group_diff(df, col):
    """患病组 vs 未患病组的差异检验

    返回 (患病均值, 未患病均值, 差值, 95%CI下限, 95%CI上限, p值, Cohen's d)
    """
    a = df.loc[df[TARGET] == 1, col].dropna()
    b = df.loc[df[TARGET] == 0, col].dropna()
    na, nb = len(a), len(b)
    ma, mb = a.mean(), b.mean()
    diff = ma - mb

    t, p = stats.ttest_ind(a, b, equal_var=False)

    # 均值差的标准误（Welch）
    se = np.sqrt(a.var(ddof=1) / na + b.var(ddof=1) / nb)
    df_w = (a.var(ddof=1) / na + b.var(ddof=1) / nb) ** 2 / (
        (a.var(ddof=1) / na) ** 2 / (na - 1) + (b.var(ddof=1) / nb) ** 2 / (nb - 1)
    )
    tcrit = stats.t.ppf(1 - ALPHA / 2, df_w)
    lo, hi = diff - tcrit * se, diff + tcrit * se

    # Cohen's d（合并标准差）
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    d = diff / sp if sp else 0.0

    return ma, mb, diff, lo, hi, p, d


def corr_test(df, col):
    """相关系数及其显著性

    用 Fisher z 变换求 95% 置信区间。
    返回 (r, p, CI下限, CI上限)
    """
    x = df[col]
    y = df[TARGET]
    m = x.notna() & y.notna()
    r, p = stats.pearsonr(x[m], y[m])
    n = int(m.sum())
    if n > 3:
        z = np.arctanh(r)
        se = 1 / np.sqrt(n - 3)
        lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    else:
        lo = hi = float("nan")
    return r, p, lo, hi


def logit_or(df, feats):
    """多元逻辑回归，返回每个变量的 (OR, CI下限, CI上限, p值)

    变量先做标准化，这样 OR 表示「每升高 1 个标准差，患病几率乘以多少」。
    """
    if not HAS_SM:
        return None
    X = df[feats].astype(float)
    X = (X - X.mean()) / X.std(ddof=1)      # z-score 标准化
    X = sm.add_constant(X)
    try:
        res = sm.Logit(df[TARGET].astype(float), X).fit(disp=0)
    except Exception:
        return None
    out = {}
    for f in feats:
        b = res.params[f]
        p = res.pvalues[f]
        lo, hi = res.conf_int().loc[f]
        out[f] = (float(np.exp(b)), float(np.exp(lo)), float(np.exp(hi)), float(p))
    return out


def main():
    if not os.path.exists(CLEAN):
        print(f"找不到 {CLEAN}，请先运行 01_clean.py")
        return

    df = pd.read_csv(CLEAN)
    n = len(df)
    n_pos = int((df[TARGET] == 1).sum())
    n_neg = n - n_pos
    m = len(FEATURES)
    alpha_adj = ALPHA / m          # Bonferroni 校正

    os.makedirs(OUT_DIR, exist_ok=True)
    L = []
    L.append("# 统计推断报告")
    L.append("")
    L.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}　｜　"
             f"数据源：`{CLEAN}`（{n} 行，患病 {n_pos} 条 / 未患病 {n_neg} 条）")
    L.append("")
    L.append("> 前面的分析只回答了「数据长什么样」；这份报告回答"
             "「这些规律是真的还是碰巧的」。")
    L.append("")

    # ---------- 一、组间差异 ----------
    L.append("## 一、组间差异检验（患病 vs 未患病）")
    L.append("")
    L.append("| 指标 | 患病组均值 | 未患病组均值 | 差值 | 95% 置信区间 | p 值 | 效应量 d | 结论 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for f in FEATURES:
        ma, mb, diff, lo, hi, p, d = group_diff(df, f)
        sig = "显著" if p < alpha_adj else ("名义显著" if p < ALPHA else "不显著")
        L.append(f"| {CN[f]} | {ma:.2f} | {mb:.2f} | {diff:+.2f} | "
                   f"[{lo:+.2f}, {hi:+.2f}] | {p:.2e} | {d:+.2f} | {sig} |")
    L.append("")
    L.append(f"显著性水平：原始 α = {ALPHA}，"
             f"Bonferroni 校正后 α' = {alpha_adj:.4f}（共检验 {m} 个指标）")
    L.append("")
    L.append("> 为什么要校正：检验 8 个指标，如果每个都用 p<0.05 判断，"
             "至少撞出一个假阳性的概率约 **34%**。校正后结论才可信。")
    L.append("")

    # ---------- 二、相关性 ----------
    L.append("## 二、各指标与患病的相关性（含显著性）")
    L.append("")
    L.append("| 指标 | 相关系数 r | 95% 置信区间 | p 值 | 结论 |")
    L.append("|---|---|---|---|---|")
    rows = []
    for f in FEATURES:
        r, p, lo, hi = corr_test(df, f)
        rows.append((CN[f], r, p, lo, hi))
    for name, r, p, lo, hi in sorted(rows, key=lambda x: -abs(x[1])):
        sig = "显著" if p < alpha_adj else ("名义显著" if p < ALPHA else "不显著")
        L.append(f"| {name} | {r:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {p:.2e} | {sig} |")
    L.append("")

    # ---------- 三、风险量化 ----------
    L.append("## 三、风险量化：逻辑回归 OR 值")
    L.append("")
    if HAS_SM:
        ors = logit_or(df, FEATURES)
        if ors:
            L.append("**单因素**（逐个指标单独回归）")
            L.append("")
            L.append("| 指标 | OR | 95% CI | p 值 |")
            L.append("|---|---|---|---|")
            for f in FEATURES:
                single = logit_or(df, [f])
                if not single:
                    continue
                o, lo, hi, p = single[f]
                L.append(f"| {CN[f]} | {o:.3f} | [{lo:.3f}, {hi:.3f}] | {p:.2e} |")
            L.append("")
            L.append("**多因素**（全部指标一起回归，控制互相影响）")
            L.append("")
            L.append("| 指标 | OR | 95% CI | p 值 |")
            L.append("|---|---|---|---|")
            for f in FEATURES:
                o, lo, hi, p = ors[f]
                L.append(f"| {CN[f]} | {o:.3f} | [{lo:.3f}, {hi:.3f}] | {p:.2e} |")
            L.append("")
            L.append("变量已标准化，OR 读作："
                     "**该指标每升高 1 个标准差，患病几率乘以 OR**。")
            L.append("OR > 1 = 风险升高；OR < 1 = 风险降低。")
        else:
            L.append("> statsmodels 未安装，跳过逻辑回归。")
    else:
        L.append("> statsmodels 未安装，跳过逻辑回归。")

    # ---------- 四、分层 ----------
    L.append("")
    L.append("## 四、分层分析：剔除年龄的混杂后，BMI 还剩多少效应")
    L.append("")
    L.append("年龄大的人更容易得糖尿病，同时 BMI 也更高。"
             "所以「BMI 与患病相关」里可能混着年龄的作用。")
    L.append("")
    L.append("按年龄分层，每层内部单独算 BMI 的 OR：")
    L.append("")
    L.append("| 年龄层 | 人数 | BMI 的 OR | 95% CI | p 值 |")
    L.append("|---|---|---|---|---|")
    edges = [21, 30, 40, 50, 100]
    labels = ["21-30 岁", "31-40 岁", "41-50 岁", "51 岁以上"]
    for i in range(len(labels)):
        sub = df[(df["Age"] >= edges[i]) & (df["Age"] <= edges[i + 1])]
        if len(sub) < 20 or sub[TARGET].nunique() < 2:
            L.append(f"| {labels[i]} | {len(sub)} | 样本不足 | — | — |")
            continue
        if HAS_SM:
            r = logit_or(sub, ["BMI"])
            if r:
                o, lo, hi, p = r["BMI"]
                L.append(f"| {labels[i]} | {len(sub)} | {o:.3f} | [{lo:.3f}, {hi:.3f}] | {p:.2e} |")
                continue
        L.append(f"| {labels[i]} | {len(sub)} | 需要 statsmodels | — | — |")
    L.append("")
    L.append("**怎么读：**如果各层的 OR 都和整体的 OR 接近，")
    L.append("说明 BMI 的效应**不是被年龄带出来的**，结论稳健；")
    L.append("如果某层 OR 掉到接近 1，说明那一层的关联主要是年龄造成的。")
    L.append("")
    L.append("---")
    L.append("")
    L.append("### 局限声明")
    L.append("")
    L.append("- **相关性不等于因果性**：本数据集是横断面的，")
    L.append("  只能说「血糖高的患者里患病的更多」，不能说「降血糖就能防糖尿病」。")
    L.append("- **样本不代表全人群**：仅含皮马印第安女性，结论不可直接外推。")
    L.append("- **缺失值填充引入了偏差**：胰岛素列近半数是填充值，")
    L.append("  与之相关的统计量被系统性低估（详见《数据质量报告》第四节）。")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    # ---------- 控制台摘要 ----------
    print("=" * 60)
    print("【统计推断】")
    print("=" * 60)
    print(f"样本 {n} 条（患病 {n_pos} / 未患病 {n_neg}）　"
          f"检验 {m} 个指标　α' = {alpha_adj:.4f}（Bonferroni）")
    print()
    print("组间差异（患病组 − 未患病组）：")
    for f in FEATURES:
        ma, mb, diff, lo, hi, p, d = group_diff(df, f)
        sig = "显著" if p < alpha_adj else ("名义显著" if p < ALPHA else "不显著")
        print(f"  {CN[f]:<12} 差 {diff:+7.2f}  "
              f"CI[{lo:+.2f},{hi:+.2f}]  p={p:.2e}  d={d:+.2f}  [{sig}]")
    print()
    if HAS_SM:
        print("多因素逻辑回归 OR（标准化后）：")
        ors = logit_or(df, FEATURES)
        for f in FEATURES:
            o, lo, hi, p = ors[f]
            print(f"  {CN[f]:<12} OR={o:.3f}  CI[{lo:.3f},{hi:.3f}]  p={p:.2e}")
    print()
    print(f"报告已输出：{OUT_MD}")


if __name__ == "__main__":
    main()
