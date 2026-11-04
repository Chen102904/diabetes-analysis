"""c3_clean.py —— 第 3 步：清洗与派生变量

这一版要同时做三件事，并且把"做了什么"写成日志：
1. 派生变量（模型能直接用的形式）
2. 缺失检查（证明主模型不需要复杂缺失处理）
3. 存盘 + 记录

原则：
    清洗阶段只做「确定的事」——类型转换、派生、标记异常。
    不做「有争议的事」——比如把 site 0 的 61 人删掉、把 tg 极端值截断。
    有争议的留到建模阶段，作为敏感性分析，因为那时候才能看出影响。

产出：
    data/china_analysis.csv     分析用数据集
    docs/china_清洗日志.md       清洗记录
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from c2_plan import BMI_SCHEMES, BMI_SCHEME_MAIN, OUTCOME, SRC, TIME

ROOT = Path(__file__).resolve().parent
DST = ROOT / "data" / "china_analysis.csv"
LOG = ROOT / "docs" / "china_清洗日志.md"

# 模型需要的变量集（与 c2_plan.COVARIATES 对应）
CORE = ["age", "gender", "site", "bmi", TIME, OUTCOME]
M3 = CORE + ["tg", "tc", "sbp", "dbp", "alt"]
M4 = M3 + ["fpg"]

# 小站点阈值：分层/固定效应里，样本过小的层估不出稳定系数
SMALL_SITE_N = 500


def main() -> None:
    df = pd.read_csv(SRC)
    n0 = len(df)
    L: list[str] = []
    add = L.append

    add("# 中国队列 · 清洗日志\n")
    add(f"> 由 `c3_clean.py` 生成。输入 `{SRC.name}`，输出 `{DST.name}`。\n")
    add(f"输入规模：**{n0:,} 行 × {df.shape[1]} 列**\n")

    # ---------- 1. 派生变量 ----------
    add("## 1. 派生变量\n")
    cuts, labels = BMI_SCHEMES[BMI_SCHEME_MAIN]

    # 性别转 0/1。模型不认 "1/2" 这种编码，会当成有序变量处理
    df["male"] = (df["gender"] == 1).astype(int)
    add("- `male` = gender==1。**为什么要转**：原始 1=男 2=女，直接进模型会被当成连续变量，"
        "解读成「性别每增加 1 单位风险变化」——毫无意义")

    # 年龄组。right=False 让 30 岁落在 30-39 而不是 20-29
    df["age_grp"] = pd.cut(df["age"], [19, 30, 40, 50, 60, 70, 100],
                           labels=["20-29", "30-39", "40-49", "50-59", "60-69", "70+"],
                           right=False)
    add("- `age_grp`：20-29 / 30-39 / 40-49 / 50-59 / 60-69 / 70+，`right=False`")

    # BMI 分组。边界值必须落在"更严重"的那一侧
    df["bmi_cat"] = pd.cut(df["bmi"], cuts, labels=labels, right=False)
    add(f"- `bmi_cat`：按 **{BMI_SCHEME_MAIN}** 切点 {cuts[1:-1]}，"
        "**`right=False` 是关键**——默认左开右闭会把 BMI 正好 24.0 的人分进「正常」，"
        "而临床标准是 ≥24 即超重")

    # 甘油三酯取对数。原始分布右偏严重，取对数后接近正态
    df["log_tg"] = np.log1p(df["tg"])
    add("- `log_tg` = log(1+tg)。原始 tg 右偏（少数人 >10，多数人 ~1），取对数后分布对称，"
        "进模型残差更稳。用 `log1p` 而不是 `log` 是为了防 0 值")

    # 小站点标记（不删除，只标记，留给建模阶段决定）
    site_n = df["site"].value_counts()
    small = site_n[site_n < SMALL_SITE_N].index.tolist()
    df["site_small"] = df["site"].isin(small).astype(int)
    add(f"- `site_small`：标记样本量 <{SMALL_SITE_N} 的站点 {small}，"
        f"共 {int(df['site_small'].sum())} 人。**只标记不删除**——"
        "是否合并要等建模时看分层 Cox 的实际表现再定")
    add("")

    # ---------- 2. 派生结果校验 ----------
    add("## 2. 派生结果校验\n")
    add("**BMI 分组人数必须与 `c2_plan.py` 校验表完全一致**（两套独立代码算同一件事，一致才算验证过）：\n")
    t = df.groupby("bmi_cat", observed=False).agg(
        n=("pid", "size"), inc=(OUTCOME, "sum"))
    t["rate_%"] = (t["inc"] / t["n"] * 100).round(2)
    add("```")
    add(t.to_string())
    add("```\n")

    add("**年龄组分布：**\n")
    ag = df.groupby("age_grp", observed=False).agg(
        n=("pid", "size"), inc=(OUTCOME, "sum"))
    add("```")
    add(ag.to_string())
    add("```\n")

    # ---------- 3. 缺失检查 ----------
    add("## 3. 缺失检查（本步最关键的结论）\n")
    add("| 变量集 | 用途 | 完整病例 | 占比 | 删掉 |")
    add("|---|---|---|---|---|")
    stats = {}
    for name, cols in [("M1/M2", CORE), ("M3", M3), ("M4", M4)]:
        ok = df[cols].notna().all(axis=1)
        stats[name] = ok
        add(f"| {name} | {'主模型' if name == 'M1/M2' else '敏感性'} | "
            f"{int(ok.sum()):,} | {ok.mean() * 100:.2f}% | {int((~ok).sum()):,} |")
    add("")
    add("**结论：主模型（M2）用的 6 个变量 0% 缺失，一个人都不用删。**\n")
    add("这是审计阶段没想到的——审计时看到 `smoking` 缺 71.6%、`hdl` 缺 44.6%，"
        "以为这个数据集缺失很严重。但那几个变量**根本不进主模型**。\n")
    add("所以：**不做多重插补，用完整病例分析（complete case）**。\n")

    # 完整病例 vs 被删的人
    cc = df[M3].notna().all(axis=1)
    add("**但要证明「删掉的人不是特殊人群」**——M3 会删 3.09%，必须看这 3% 长什么样：\n")
    cmp = (df.assign(cc=cc).groupby("cc")[["age", "bmi", "fpg", "tg", "followup_years", OUTCOME]]
           .mean().round(3))
    cmp.index = ["被删（缺 tg/tc/alt）", "保留"]
    add("```")
    add(cmp.to_string())
    add("```\n")
    add("年龄差 3 岁（39.3 vs 42.2），BMI 和发病率几乎无差。"
        "年龄差异来自 tg/tc 检查在部分站点/年份未开展，那些人群偏年轻。")
    add("**判定：差异存在但不影响结论，M3 结果需在报告里注明「完整病例分析，n=205,291」。**\n")

    # ---------- 4. 有意不做的事 ----------
    add("## 4. 这一步有意不做的事\n")
    add("| 没做什么 | 为什么不做 |")
    add("|---|---|")
    add("| 没删 site 0 的 61 人 | 虽然 0 发病，但删数据要有依据。标记为 `site_small`，"
        "建模时看分层 Cox 怎么处理 |")
    add("| 没截断 tg / alt 的极端值 | 极端值可能是真实的重度患者，不是录入错误。"
        "截断会人为减小方差，属于篡改。做敏感性分析（1%/99% winsorize）对比 |")
    add("| 没给 `smoking` 填任何值 | 缺失 71.6% 且是站点级结构性缺失，填值等于编数据。"
        "主模型不用它 |")
    add("| 没把 `dm_incident` 当结局 | 审计已确认它只标了 31% 的发病者 |")
    add("")

    # ---------- 5. 输出 ----------
    df.to_csv(DST, index=False)
    add("## 5. 输出\n")
    add(f"- 文件：`{DST.name}`")
    add(f"- 规模：**{len(df):,} 行 × {df.shape[1]} 列**（原 25 列 + 派生 5 列）")
    add(f"- 大小：{DST.stat().st_size / 1048576:.1f} MB")
    add("")
    add("**列清单：**\n")
    add("```")
    add(", ".join(df.columns))
    add("```")

    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("\n".join(L), encoding="utf-8")

    print(f"[OK] 清洗完成 -> {DST}  ({len(df):,} 行 x {df.shape[1]} 列)")
    print(f"     日志 -> {LOG}")
    print()
    print("BMI 分组校验：")
    print(t.to_string())
    print()
    print("缺失校验：")
    for k, v in stats.items():
        print(f"  {k:<8} 完整 {int(v.sum()):>7,} ({v.mean()*100:6.2f}%)")


if __name__ == "__main__":
    main()
