"""快速验证：能否用这份数据复现原论文的核心结论。

只做验证，不做正式分析。跑完看三件事：
1. 发病率和人年是否对得上（原论文 7.35/1000 人年）
2. BMI 每 +1 的 OR 是否接近 1.23
3. 年龄 × BMI 交互是否显著（原论文 P<0.0001）
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")

df = pd.read_csv("data/raw_china/rc_china.csv")
n = len(df)
y = df["dm_censor"]

print(f"样本 {n:,} 人   发病 {int(y.sum()):,} 人 ({100 * y.mean():.2f}%)")
print(f"累计随访 {df['followup_years'].sum():,.0f} 人年")
print(f"发病率 {1000 * y.sum() / df['followup_years'].sum():.2f} / 1000 人年"
      f"   （原论文年龄调整后 7.35）")
print()

r = sm.Logit(y, sm.add_constant(df[["bmi"]])).fit(disp=0)
b, se = r.params["bmi"], r.bse["bmi"]
print(f"BMI 每 +1 kg/m²：OR = {np.exp(b):.3f}  95%CI [{np.exp(b - 1.96 * se):.3f}, "
      f"{np.exp(b + 1.96 * se):.3f}]   p = {r.pvalues['bmi']:.2e}")
print("   原论文：1.23（95%CI 1.22–1.24）")
print()

d = df.copy()
d["age_c"] = d.age - d.age.mean()
d["bmi_c"] = d.bmi - d.bmi.mean()
d["age_x_bmi"] = d.age_c * d.bmi_c
d["male"] = (d.gender == 1).astype(int)
r2 = sm.Logit(y, sm.add_constant(d[["bmi_c", "age_c", "age_x_bmi", "male"]])).fit(disp=0)
pv = r2.pvalues["age_x_bmi"]
print(f"年龄×BMI 交互项：系数 {r2.params['age_x_bmi']:.5f}，p = {pv:.3e}"
      f"   {'✅ 显著，与原论文 P<0.0001 一致' if pv < 0.001 else '⚠ 不显著'}")
print()

print("分年龄组的 BMI 效应（每 +1 kg/m²）：")
print(f"  {'年龄段':<10}{'样本':>9}{'发病':>8}{'OR':>7}{'95%CI':>18}")
for lo, hi in zip([20, 30, 40, 50, 60, 70], [30, 40, 50, 60, 70, 100]):
    s = d[(d.age >= lo) & (d.age < hi)]
    if len(s) < 100 or s.dm_censor.sum() < 10:
        continue
    m = sm.Logit(s.dm_censor, sm.add_constant(s[["bmi"]])).fit(disp=0)
    bb, ss = m.params["bmi"], m.bse["bmi"]
    print(f"  {lo:>2}-{hi:<6}{len(s):>9,}{int(s.dm_censor.sum()):>8,}"
          f"{np.exp(bb):>7.2f}   [{np.exp(bb - 1.96 * ss):.2f}, {np.exp(bb + 1.96 * ss):.2f}]")
