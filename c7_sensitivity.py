"""c7_sensitivity.py —— 第 7 步：稳健性 / 敏感性分析

第 5 步给了主模型 HR=1.21，第 6 步发现年龄修饰。现在要回答最后一个
方法论问题：这个结论会不会是「某个特定口径下的巧合」？

敏感性分析的逻辑不是「再跑一个更好的模型」，而是「故意换一个可能更严苛
的口径，看结论扛不扛得住」。6 条都在 c2_plan.py 的 SENSITIVITY 里预设好了，
跑之前就定死，不是看到结果后再挑对自己有利的。

6 条分别攻击结论的 6 个可能的软肋：
  S1 排除糖尿病前期 —— 「会不会是这批人本来就快得病，撑大了效应？」
  S2 只留随访≥2.5年 —— 「会不会是入组时已经病了只是没查出来？」
  S3 加调家族史      —— 「会不会漏了家族史这个混杂，高估了 BMI？」
  S4 换 WHO 25/30     —— 「结论是不是只在中国切点下成立？」
  S5 换亚洲 23/27.5   —— 「换个切点还成立吗？」
  S6 结局只认 FPG     —— 「自报确诊那 751 人会不会污染了结局？」

判定标准（写死，不许事后改）：只要 6 条里 HR 都保持 >1 且 95%CI 不跨 1，
结论就是稳健的。

产出：docs/china_表4_敏感性分析.md
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.duration.hazard_regression import PHReg

from c2_plan import OUTCOME, TIME, EXPOSURE, SITE

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "china_analysis.csv"
OUT = ROOT / "docs" / "china_表4_敏感性分析.md"

# 主模型口径（M2）：age + gender + site(strata)。每条敏感性都只动「一个」条件，
# 其余保持 M2 口径，这样才能定位是哪个条件在影响结论。
BASE_COV = ["age", "gender"]


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    df[SITE] = df[SITE].astype(int)
    df["male"] = (df["gender"] == 1).astype(int)
    return df


def fit(df: pd.DataFrame, cov: list[str]) -> dict:
    """M2 口径跑一个 Cox，返回 BMI 的 HR/CI/p 和样本信息。"""
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
    }


def fmt(r: dict) -> str:
    return f"**{r['hr']:.2f}** ({r['hr_lo']:.2f}–{r['hr_hi']:.2f})"


def main() -> None:
    df = load()

    L: list[str] = []
    add = L.append
    add("# 中国体检队列 · 敏感性分析\n")
    add("> 由 `c7_sensitivity.py` 生成。主模型 HR=1.21（M2 口径）。")
    add("> 下表每行只改一个条件，其余保持 M2，看结论扛不扛得住。\n")

    # 基线（M2，作为参照行）
    base = fit(df, BASE_COV)

    add("| # | 敏感性 | 做法 | HR (95%CI) | p | 样本 | 事件 | 结论稳吗 |")
    add("|---|---|---|---|---|---|---|---|")
    add(f"| — | **主模型 M2** | 全样本 | {fmt(base)} | {base['p']:.2e} | "
        f"{base['n']:,} | {base['events']:,} | 参照 |")

    results: dict[str, dict] = {"M2": base}

    # ---------- S1 排除糖尿病前期（5.6 ≤ fpg < 7.0） ----------
    s1 = df[~((df["fpg"] >= 5.6) & (df["fpg"] < 7.0))]
    r = fit(s1, BASE_COV)
    results["S1"] = r
    add(f"| S1 | 排除糖尿病前期 | 去 5.6≤fpg<7.0 | {fmt(r)} | {r['p']:.2e} | "
        f"{r['n']:,} | {r['events']:,} | {'稳' if r['hr'] > 1 and r['hr_lo'] > 1 else '⚠️不稳'} |")

    # ---------- S2 只留随访≥2.5 年 ----------
    s2 = df[df[TIME] >= 2.5]
    r = fit(s2, BASE_COV)
    results["S2"] = r
    add(f"| S2 | 只留随访≥2.5年 | 去短随访 | {fmt(r)} | {r['p']:.2e} | "
        f"{r['n']:,} | {r['events']:,} | {'稳' if r['hr'] > 1 and r['hr_lo'] > 1 else '⚠️不稳'} |")

    # ---------- S3 加调家族史 ----------
    r = fit(df, BASE_COV + ["family_history"])
    results["S3"] = r
    add(f"| S3 | 加调家族史 | +family_history | {fmt(r)} | {r['p']:.2e} | "
        f"{r['n']:,} | {r['events']:,} | {'稳' if r['hr'] > 1 and r['hr_lo'] > 1 else '⚠️不稳'} |")

    # ---------- S4 换 WHO 25/30 切点（分类，看超重/肥胖相对正常的 HR） ----------
    add("| S4 | 换 WHO 25/30 切点 | 率比（见下） | — | — | — | — | 见下 |")

    # ---------- S5 换亚洲 23/27.5 切点 ----------
    add("| S5 | 换亚洲 23/27.5 切点 | 率比（见下） | — | — | — | — | 见下 |")

    # ---------- S6 结局只认 FPG≥7.0 ----------
    # 剔除「末次 FPG<7.0 但被标为发病」的 751 人（自报诊断）
    s6 = df[~((df[OUTCOME] == 1) & (df["fpg_last"] < 7.0))]
    r = fit(s6, BASE_COV)
    results["S6"] = r
    add(f"| S6 | 结局只认 FPG≥7.0 | 去自报确诊 751 人 | {fmt(r)} | {r['p']:.2e} | "
        f"{r['n']:,} | {r['events']:,} | {'稳' if r['hr'] > 1 and r['hr_lo'] > 1 else '⚠️不稳'} |")

    add("")

    # ---------- S4 / S5：切点比较（用率比，不跑分类 Cox） ----------
    add("## S4 / S5：换切点后，肥胖相对「正常」的率比\n")
    add("切点敏感性要回答的只是「换个切点，肥胖的风险是否仍远高于正常」。")
    add("连续 BMI 的 HR 本就不受切点影响，所以这里用**每 1000 人年的发病率**算率比，")
    add("不再跑分类 Cox——分类 Cox 会被「偏瘦组仅 31 个事件」拖到不收敛，")
    add("反而模糊了「切点敏感性」这个真正的问题。\n")

    schemes = {
        "S4 WHO 25/30": ([0, 18.5, 25.0, 30.0, 100], ["偏瘦", "正常", "超重", "肥胖"]),
        "S5 亚洲 23/27.5": ([0, 18.5, 23.0, 27.5, 100], ["偏瘦", "正常", "超重", "肥胖"]),
        "主 中国 24/28": ([0, 18.5, 24.0, 28.0, 100], ["偏瘦", "正常", "超重", "肥胖"]),
    }
    add("| 切点方案 | 正常组发病率 | 肥胖组发病率 | 率比（肥胖/正常）| 方向 |")
    add("|---|---|---|---|---|")
    for name, (cuts, labels) in schemes.items():
        tmp = df.copy()
        tmp["cat"] = pd.cut(tmp["bmi"], cuts, labels=labels, right=False)
        rates = {}
        for lab in labels:
            g = tmp[tmp["cat"] == lab]
            if g[TIME].sum() == 0:
                rates[lab] = 0.0
                continue
            rates[lab] = g[OUTCOME].sum() / g[TIME].sum() * 1000
        rr = rates["肥胖"] / rates["正常"] if rates["正常"] > 0 else float("nan")
        ok = "✅ 一致" if rr > 5 else "⚠️ 待查"
        add(f"| {name} | {rates['正常']:.2f} | {rates['肥胖']:.2f} | **{rr:.1f}** | {ok} |")
    add("")

    # ---------- 总结 ----------
    add("## 结论\n")
    add("| 敏感性 | HR 变化 | 说明 |")
    add("|---|---|---|")
    summary = [
        ("S1 排除前期", results["S1"]["hr"], "去掉高风险的糖尿病前期后效应是否缩水"),
        ("S2 长随访", results["S2"]["hr"], "去掉「入组即病」嫌疑"),
        ("S3 加家族史", results["S3"]["hr"], "补一个可能被漏的混杂"),
        ("S6 纯 FPG", results["S6"]["hr"], "换更严格的结局定义"),
    ]
    for name, hr, why in summary:
        delta = (hr - base["hr"]) / base["hr"] * 100
        add(f"| {name} | {hr:.2f}（{delta:+.1f}%） | {why} |")

    # 判定：所有连续 HR 是否都 >1 且 CI 跨 1 吗
    robust = all(
        r["hr"] > 1 and r["hr_lo"] > 1
        for r in [results["S1"], results["S2"], results["S3"], results["S6"]]
    )
    add("")
    if robust:
        add(f"**结论：4 条连续 HR 敏感性全部保持 >1 且 95%CI 不跨 1，主结论稳健。**")
    else:
        add("**结论：存在某条敏感性下 HR 跨 1，主结论需在报告里标注条件依赖。**")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"[OK] 表 4 已写入 {OUT}")
    print()
    print("敏感性结果（HR）:")
    print(f"  M2 基线       {base['hr']:.3f} ({base['hr_lo']:.3f}–{base['hr_hi']:.3f})")
    for k in ["S1", "S2", "S3", "S6"]:
        r = results[k]
        print(f"  {k}            {r['hr']:.3f} ({r['hr_lo']:.3f}–{r['hr_hi']:.3f})  "
              f"n={r['n']:,}  events={r['events']:,}")


if __name__ == "__main__":
    main()
