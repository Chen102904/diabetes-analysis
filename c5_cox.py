"""c5_cox.py —— 第 5 步：Cox 比例风险回归（M1 → M4）

第 4 步证明了「BMI 越高发病率越高」这个描述性关联。这一步要回答的是：
控制混杂之后，这个关联还成不成立、有多大。

模型序列（在 c2_plan.py 里写死，这里只 import，不许改口径）：
    M1   age + gender              —— 不调整时有没有关联
    M2   age + gender + site(strata) —— 主模型，扣掉站点差异
    M3   + tg/tc/sbp/dbp/alt       —— 扣掉代谢指标衰减多少
    M4   + fpg                     —— 扣掉基线血糖（检验中介，敏感性）

关键实现点（为什么要这么写）：
    1. 暴露用连续 BMI，每 +1 kg/m² 的 HR。连续变量信息最全，分类会丢梯度。
       分类结果只放在描述统计，不进模型。
    2. site 用 strata 而不是 15 个哑变量：分层允许每个站点有各自的基线风险函数，
       不需要「各站点基线相同」这个强假设，还省 14 个自由度。
    3. statsmodels.PHReg 只返回 coef 和 coef 的 CI，不直接给 HR，
       所以要自己 exp() 换算。HR 的 CI 也取 exp(conf_int)。
    4. 比例风险假设用 Schoenfeld 残差检验：看每个协变量与时间是否相关，
       p<0.05 说明该变量「加速/减速」不恒定，需要加时间交互项。

产出：docs/china_表2_Cox回归.md（自动生成）
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.duration.hazard_regression import PHReg

from c2_plan import (
    OUTCOME, TIME, EVENT, EXPOSURE, SITE, MODELS, MODEL_MAIN, SENSITIVITY,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "china_analysis.csv"
OUT = ROOT / "docs" / "china_表2_Cox回归.md"

# site 的分层变量必须按整数编号，statsmodels 内部把 strata 当成整数索引。
# c3 里 site 保持原始值即可，这里确保是 int。
def load() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    df[SITE] = df[SITE].astype(int)
    return df


def run_cox(df: pd.DataFrame, cov: list[str], strata_site: bool) -> dict:
    """跑一个 Cox 模型，返回 BMI 的 HR、CI、p 值，以及各协变量的结果。"""
    # 暴露变量 BMI 必须始终在协变量里（M1 的 cfg 只写了混杂，没写 BMI）
    cov = [EXPOSURE] + [c for c in cov if c != EXPOSURE]
    cols = cov + ([SITE] if strata_site else []) + [TIME, OUTCOME]

    # 显式 dropna：PHReg 不会自动删含 NaN 的行，而是静默算错。
    # 第 3 步已核实：M1/M2 的变量 0% 缺失，M3/M4 删 3.09% 且删的人与留下的人无系统差异。
    sub = df[cols].dropna().copy()
    n_drop = len(df) - len(sub)

    X = sub[cov].copy()
    strata = sub[SITE].values.astype(int) if strata_site else None
    dur = sub[TIME].values
    event = sub[OUTCOME].values

    model = PHReg(dur, X, status=event, strata=strata)
    res = model.fit()

    # BMI 必须是协变量之一
    bmi_idx = list(cov).index(EXPOSURE)
    coef = res.params[bmi_idx]
    ci = res.conf_int()[bmi_idx]          # 这是 beta 的 CI，要 exp
    hr = float(np.exp(coef))
    hr_lo = float(np.exp(ci[0]))
    hr_hi = float(np.exp(ci[1]))
    pval = float(res.pvalues[bmi_idx])

    return {
        "n": int(len(sub)),
        "n_drop": int(n_drop),
        "events": int(event.sum()),
        "hr": hr, "hr_lo": hr_lo, "hr_hi": hr_hi, "p": pval,
        "res": res, "cov": cov, "sub": sub,
    }


def fmt(hr: float, lo: float, hi: float, p: float) -> str:
    return f"{hr:.3f} ({lo:.3f}–{hi:.3f})"


def main() -> None:
    df = load()

    L: list[str] = []
    add = L.append
    add("# 中国体检队列 · Cox 回归结果（表 2）\n")
    add("> 由 `c5_cox.py` 生成。结局 = `dm_censor`，时间 = `followup_years`，"
        "暴露 = BMI 每 +1 kg/m²。site 作为分层变量（strata）。\n")
    add(f"| 模型 | 调整变量 | HR (95%CI) | p | 样本 | 事件数 |")
    add("|---|---|---|---|---|---|")

    results: dict[str, dict] = {}
    for name, cfg in MODELS.items():
        strata_site = SITE in cfg["cov"]
        cov = [c for c in cfg["cov"] if c != SITE]   # site 从协变量里拿出来做 strata
        r = run_cox(df, cov, strata_site)
        results[name] = r
        note = f"（删 {r['n_drop']:,}）" if r["n_drop"] else ""
        add(f"| **{name}** | {', '.join(cfg['cov'])} | "
            f"**{fmt(r['hr'], r['hr_lo'], r['hr_hi'], r['p'])}** | {r['p']:.2e} | "
            f"{r['n']:,}{note} | {r['events']:,} |")
    add("")

    # ---- 解读规则（c2_plan 里写死，这里照做，不许改口）----
    add("## 解读（按 c2_plan.py 预设规则）\n")
    m1, m2 = results["M1"], results["M2"]
    atten_12 = (m1["hr"] - m2["hr"]) / (m1["hr"] - 1) * 100 if m1["hr"] > 1 else 0
    add(f"- M1→M2 效应衰减 **{atten_12:.1f}%** "
        f"{'（>20%，站点构成是重要混杂，必须说明）' if abs(atten_12) > 20 else ''}")
    if "M3" in results:
        atten_23 = (m2["hr"] - results["M3"]["hr"]) / (m2["hr"] - 1) * 100
        add(f"- M2→M3 效应衰减 **{atten_23:.1f}%** "
            f"{'（>30%，代谢指标是主要通路，BMI 直接效应有限）' if abs(atten_23) > 30 else ''}")
    if "M4" in results:
        m4 = results["M4"]
        add(f"- M3→M4 效应 {'接近 0 且 CI 跨 1 —— BMI 作用几乎全经由血糖，是中介而非独立危险因素'
            if m4['hr_lo'] < 1 < m4['hr_hi'] else '仍显著 —— BMI 有独立于血糖的效应'}")
    add("")

    # ---- 比例风险假设检验 ----
    # Schoenfeld 残差 = score residuals。检验每个协变量的残差是否随时间系统变化：
    # 用 Kendall 秩相关（残差 vs 时间），p<0.05 说明该变量「加速/减速」不恒定。
    # 残差里有少量 NaN（删失个体无贡献），检验前剔除，否则 kendalltau 返回 nan。
    add("## 比例风险假设（Schoenfeld 残差）\n")
    add("| 模型 | 变量 | ρ(Kendall) | p | 判定 |")
    add("|---|---|---|---|---|")
    from scipy.stats import kendalltau
    main = results[MODEL_MAIN]
    main_cov = [c for c in main["cov"] if c != SITE]
    sr = main["res"].score_residuals
    t_main = main["sub"][TIME].values
    for j, varname in enumerate(main_cov):
        keep = ~np.isnan(sr[:, j])
        if keep.sum() < 2:
            add(f"| {MODEL_MAIN} | `{varname}` | — | — | 残差不足，跳过 |")
            continue
        rho, p = kendalltau(t_main[keep], sr[keep, j])
        add(f"| {MODEL_MAIN} | `{varname}` | {rho:.3f} | {p:.3f} | "
            f"{'⚠️ 违反，需加时间交互项' if p < 0.05 else '成立'} |")
    add("")

    add("## 结论\n")
    r = results[MODEL_MAIN]
    add(f"主模型 **{MODEL_MAIN}**：BMI 每 +1 kg/m²，新发糖尿病风险 "
        f"**HR={r['hr']:.2f}（95%CI {r['hr_lo']:.2f}–{r['hr_hi']:.2f}）**。")
    add("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"[OK] 表 2 已写入 {OUT}")
    print()
    for name, r in results.items():
        print(f"{name}: HR {fmt(r['hr'], r['hr_lo'], r['hr_hi'], r['p'])}  p={r['p']:.2e}  "
              f"n={r['n']:,}  events={r['events']:,}")


if __name__ == "__main__":
    main()
