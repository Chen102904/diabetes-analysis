# -*- coding: utf-8 -*-
"""
为 Power BI 生成「报表专用」的小型聚合数据。

为什么不用 21 万行明细直接导 Power BI？
- 明细 29MB、21 万行，导入慢、报表拖拽卡
- 报表要的是「发病率」「HR」这些聚合结果，不是每一条记录
所以这里把核心结论聚合成两张小表，Power BI 只吃这几百行。

产出（放在 powerbi/ 目录）：
  1. powerbi/发病率.csv       —— BMI 分组 × 年龄组 的发病率（柱状图/折线图用）
  2. powerbi/年龄HR.csv       —— 年龄分层 Cox HR（森林图用）
"""
import numpy as np
import pandas as pd
from pathlib import Path

SRC = Path("data/china_analysis.csv")
OUT = Path("powerbi")
OUT.mkdir(exist_ok=True)

df = pd.read_csv(SRC)

# 确保分类列是 category，避免 cut 出来的 NaN 干扰
BMI_ORDER = ["偏瘦", "正常", "超重", "肥胖"]
AGE_ORDER = ["20-29", "30-39", "40-49", "50-59", "60-69", "70+"]

# ---------- 表 1：BMI 分组发病率（每 1000 人年）----------
def rate_ci(events, py):
    """发病率 + 95%CI（Poisson 近似）"""
    r = events / py * 1000
    se = np.sqrt(events) / py * 1000
    return r, r - 1.96 * se, r + 1.96 * se

rows = []
for grp in BMI_ORDER:
    v = df[df["bmi_cat"] == grp]
    n = len(v)
    ev = int(v["dm_censor"].sum())
    py = v["followup_years"].sum()
    r, lo, hi = rate_ci(ev, py)
    rows.append({
        "BMI分组": grp,
        "人数": n,
        "发病人数": ev,
        "随访人年": round(py, 0),
        "发病率_每千人年": round(r, 2),
        "下限": round(lo, 2),
        "上限": round(hi, 2),
    })

t1 = pd.DataFrame(rows)
t1["BMI分组"] = pd.Categorical(t1["BMI分组"], categories=BMI_ORDER, ordered=True)
t1.to_csv(OUT / "发病率.csv", index=False, encoding="utf-8-sig")

# ---------- 表 2：年龄 × BMI 分层发病率（折线图用）----------
rows2 = []
for ag in AGE_ORDER:
    for grp in BMI_ORDER:
        v = df[(df["age_grp"] == ag) & (df["bmi_cat"] == grp)]
        if len(v) == 0:
            continue
        ev = int(v["dm_censor"].sum())
        py = v["followup_years"].sum()
        r, lo, hi = rate_ci(ev, py)
        rows2.append({
            "年龄组": ag,
            "BMI分组": grp,
            "人数": len(v),
            "发病人数": ev,
            "发病率_每千人年": round(r, 2),
            "事件数小于10": ev < 10,   # 图上要标空心的不稳组
        })

t2 = pd.DataFrame(rows2)
t2["年龄组"] = pd.Categorical(t2["年龄组"], categories=AGE_ORDER, ordered=True)
t2["BMI分组"] = pd.Categorical(t2["BMI分组"], categories=BMI_ORDER, ordered=True)
t2.to_csv(OUT / "年龄BMI分层.csv", index=False, encoding="utf-8-sig")

# ---------- 表 3：年龄分层 HR（森林图用，数字来自 c6 已算结果）----------
# 这几个 HR 是 c6_effect_mod.py 已经跑出来的定值，直接固化，避免这里重复跑 Cox
hr_rows = [
    ("20-29", 1.37, 1.29, 1.45),
    ("30-39", 1.31, 1.27, 1.35),
    ("40-49", 1.27, 1.24, 1.30),
    ("50-59", 1.19, 1.16, 1.22),
    ("60-69", 1.13, 1.10, 1.16),
    ("70+",   1.10, 1.06, 1.14),
]
t3 = pd.DataFrame(hr_rows, columns=["年龄组", "HR", "下限", "上限"])
t3["年龄组"] = pd.Categorical(t3["年龄组"], categories=AGE_ORDER, ordered=True)
t3.to_csv(OUT / "年龄HR.csv", index=False, encoding="utf-8-sig")

# ---------- 打印结果 ----------
print("已生成 Power BI 数据文件（powerbi/ 目录）：")
for f in ["发病率.csv", "年龄BMI分层.csv", "年龄HR.csv"]:
    p = OUT / f
    print(f"  {f}  ({p.stat().st_size/1024:.1f} KB)")
print()
print("=== 表1 发病率 ===")
print(t1.to_string(index=False))
print()
print("=== 表3 年龄HR ===")
print(t3.to_string(index=False))
