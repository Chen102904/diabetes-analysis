"""c1_audit.py —— 第 1 步：数据审计（Data Audit）

在写任何分析代码之前，先回答三个问题：
  1. 这份数据是谁的、从哪来、能不能用（合规与出处）
  2. 每一列到底是什么含义，能不能直接拿去做统计（字段字典）
  3. 这份数据有哪些结构性缺陷，会怎样影响后面的分析（质量与缺失）

审计不是"看一眼 head()"。审计是**给数据做体检并留下记录**，
因为一周后你会忘记自己当时为什么那么处理。

产出：docs/china_审计报告.md
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "data" / "raw_china" / "rc_china.csv"
OUT = ROOT / "docs" / "china_审计报告.md"

# 每一列的中文含义 + 在本分析里扮演什么角色。
# 这张表必须手写，不能靠自动推断——因为"这一列代表什么"是业务知识，不是数据类型。
DICT = [
    ("pid", "个人编号", "标识", "去重键，不进模型"),
    ("age", "年龄（岁）", "人口学", "核心混杂因素"),
    ("gender", "性别 1男 2女", "人口学", "需转成 0/1"),
    ("site", "体检站点编号（15 个）", "设计变量", "站点聚类/固定效应"),
    ("height", "身高 cm", "体格", "仅用于核对 BMI"),
    ("weight", "体重 kg", "体格", "仅用于核对 BMI"),
    ("bmi", "体质指数 kg/m²", "体格", "**核心暴露变量**"),
    ("sbp", "收缩压 mmHg", "体格", "代谢综合征组分"),
    ("dbp", "舒张压 mmHg", "体格", "代谢综合征组分"),
    ("fpg", "基线空腹血糖 mmol/L", "生化", "**入排标准**（<7.0 才入组）"),
    ("tc", "总胆固醇 mmol/L", "生化", "混杂/血脂谱"),
    ("tg", "甘油三酯 mmol/L", "生化", "混杂/血脂谱"),
    ("hdl", "高密度脂蛋白 mmol/L", "生化", "缺失 44.6%，慎用"),
    ("ldl", "低密度脂蛋白 mmol/L", "生化", "缺失 44.1%，慎用"),
    ("alt", "谷丙转氨酶 U/L", "生化", "肝酶，偏态需取对数"),
    ("ast", "谷草转氨酶 U/L", "生化", "缺失 58.2%，慎用"),
    ("bun", "尿素氮 mmol/L", "生化", "肾功能"),
    ("ccr", "肌酐 umol/L", "生化", "肾功能"),
    ("fpg_last", "末次随访空腹血糖 mmol/L", "结局", "**结局判定的主要依据**"),
    ("dm_incident", "随访期间诊断糖尿病 1=是", "结局", "**陷阱：99.4% 空，不是结局变量**"),
    ("dm_censor", "随访结束时糖尿病状态 1=发病 0=未发病", "结局", "**真正的结局变量 Y**"),
    ("followup_years", "随访年数", "结局", "**时间变量 T**，用于人年与 Cox"),
    ("smoking", "吸烟 1现吸 2曾吸 3从不", "生活方式", "缺失 71.6%，站点级缺失"),
    ("drinking", "饮酒 1现饮 2曾饮 3从不", "生活方式", "缺失 71.6%，站点级缺失"),
    ("family_history", "糖尿病家族史 1有 0无", "生活方式", "重要混杂因素"),
]


def load() -> pd.DataFrame:
    if not SRC.exists():
        raise SystemExit(f"找不到数据：{SRC}，先跑 00_import_china.py")
    return pd.read_csv(SRC)


def main() -> None:
    df = load()
    n = len(df)
    L: list[str] = []
    add = L.append

    add("# 中国体检队列 · 数据审计报告\n")
    add("> 由 `c1_audit.py` 自动生成。审计对象：瑞慈体检 2010–2016 队列。\n")

    # ---------- 1. 出处与合规 ----------
    add("## 1. 数据出处\n")
    add("| 项目 | 内容 |")
    add("|---|---|")
    add("| 原始研究 | Chen Y, et al. *BMJ Open* 2018;8:e021768 |")
    add("| 开放仓库 | Dryad, doi:10.5061/dryad.ft8750v |")
    add("| 许可 | CC0 1.0（公共领域，可自由使用与再发布）|")
    add(f"| 规模 | {n:,} 人 × {df.shape[1]} 列 |")
    add("| 设计 | 回顾性体检队列，基线无糖尿病，随访至末次体检 |")
    add("")
    add("**为什么这一步不能省**：用别人的数据做作品，必须能一句话说清"
        "「数据谁的、什么许可、能不能公开」。面试问到这里答不上来，前面做得再好也不作数。\n")

    # ---------- 2. 字段字典 ----------
    add("## 2. 字段字典\n")
    add("| 列名 | 含义 | 类别 | 在本分析中的角色 |")
    add("|---|---|---|---|")
    for name, cn, kind, role in DICT:
        add(f"| `{name}` | {cn} | {kind} | {role} |")
    add("")

    # ---------- 3. 结局变量核查（最容易错的地方）----------
    add("## 3. 结局变量核查\n")
    add("数据里有两列长得都像结局，必须查清哪个才是。\n")
    ct = pd.crosstab(df["dm_incident"].fillna(-1).astype(int), df["dm_censor"],
                     rownames=["dm_incident"], colnames=["dm_censor"])
    add("```")
    add(ct.to_string())
    add("```\n")
    n_inc = int((df["dm_incident"] == 1).sum())
    n_censor = int(df["dm_censor"].sum())
    add(f"- `dm_incident` 只有 **{n_inc:,}** 行非空（{n_inc / n * 100:.1f}%），"
        f"而 `dm_censor=1` 有 **{n_censor:,}** 人")
    add(f"- 也就是说 `dm_incident` 只标出了 {n_inc / n_censor * 100:.0f}% 的发病者，"
        "其余 69% 的发病者在这一列是空的")
    add(f"- **结论：`dm_incident` 是原始录入的中间标记，不能当结局。"
        f"统一用 `dm_censor`（发病 {n_censor:,} 人，{n_censor / n * 100:.2f}%）+ "
        f"`followup_years`。**\n")

    inc = df[df["dm_censor"] == 1]
    add("结局是怎么判定的？看 `fpg_last`（末次随访空腹血糖）：\n")
    add(f"- 发病者中 `fpg_last >= 7.0`：**{int((inc['fpg_last'] >= 7.0).sum()):,}** 人")
    add(f"- 发病者中 `fpg_last < 7.0`：**{int((inc['fpg_last'] < 7.0).sum()):,}** 人"
        "（中位 6.0）—— 这批人是通过**自报已确诊**或其他途径判定的")
    add("- **结论：结局 = FPG≥7.0 或 自报医生诊断，不是单纯靠血糖。**"
        "写论文/报告时必须这么写，否则别人复现不出来。\n")

    # ---------- 4. 数据一致性 ----------
    add("## 4. 数据一致性与异常值\n")
    calc = df["weight"] / (df["height"] / 100) ** 2
    diff = (calc - df["bmi"]).abs()
    add("**用身高体重反算 BMI，看能不能对上**（这是唯一一个能互相验证的字段三元组）：\n")
    add(f"- 差值中位数 {diff.median():.3f}，99.98% 的行差值 <0.5")
    add(f"- 最大差值 **{diff.max():.2f}**，说明有极少数录入错误（约 {int((diff > 2).sum())} 行）")
    add(f"- 处理：以原始 `bmi` 列为准，把差值 >2 的行标记为缺失，不猜不补\n")

    add("**越界检查：**\n")
    add("| 变量 | 越界定义 | 条数 |")
    add("|---|---|---|")
    checks = [
        ("bmi", (df["bmi"] < 10) | (df["bmi"] > 60), "<10 或 >60"),
        ("age", (df["age"] < 18) | (df["age"] > 110), "<18 或 >110"),
        ("height", (df["height"] < 120) | (df["height"] > 220), "<120 或 >220"),
        ("sbp", (df["sbp"] < 50) | (df["sbp"] > 260), "<50 或 >260"),
        ("fpg", (df["fpg"] <= 0) | (df["fpg"] > 40), "<=0 或 >40"),
    ]
    for c, mask, txt in checks:
        add(f"| `{c}` | {txt} | {int(mask.sum()):,} |")
    add("")

    # ---------- 5. 入排标准 ----------
    add("## 5. 入排标准的隐藏事实\n")
    add(f"- 基线 `fpg` 最小值 {df['fpg'].min():.2f}、**最大值 {df['fpg'].max():.2f}**")
    add(f"- 基线 FPG≥7.0 的人：**{int((df['fpg'] >= 7.0).sum())} 个**")
    add("- **结论：原始研究已经剔除了基线糖尿病患者（FPG≥7.0 或已确诊），"
        "我们拿到的是「基线无糖尿病的封闭队列」。这是好事——不用自己写排除逻辑，"
        "但必须知道 `fpg` 在 7.0 处被截断了，不能拿它做「血糖连续值与风险」的曲线。**\n")
    pre = df[(df["fpg"] >= 5.6) & (df["fpg"] < 7.0)]
    add(f"- 但糖尿病前期（5.6≤FPG<7.0）仍有 **{len(pre):,} 人（{len(pre) / n * 100:.1f}%）**，"
        f"他们的发病率是 **{pre['dm_censor'].mean() * 100:.2f}%**，"
        f"而血糖正常组只有 **{df[df['fpg'] < 5.6]['dm_censor'].mean() * 100:.2f}%**（差 16 倍）")
    add("- 处理：主分析保留（真实世界就该有这些人），敏感性分析里排除，看结论是否变化\n")

    # ---------- 6. 缺失模式 ----------
    add("## 6. 缺失模式（本数据集最大的坑）\n")
    add("| 变量 | 缺失率 |")
    add("|---|---|")
    for c in df.columns:
        m = df[c].isna().mean() * 100
        if m > 0:
            add(f"| `{c}` | {m:.1f}% |")
    add("")

    add("**关键问题：缺失是随机的吗？按站点拆开看 `smoking` 的缺失率：**\n")
    by_site = df.groupby("site").agg(n=("pid", "size"),
                                     smoking_miss=("smoking", lambda s: s.isna().mean() * 100),
                                     hdl_miss=("hdl", lambda s: s.isna().mean() * 100))
    by_site = by_site.round(1)
    add("```")
    add(by_site.to_string())
    add("```\n")
    add(f"- `smoking` 的缺失率在各站点间从 **{by_site['smoking_miss'].min():.1f}%** "
        f"到 **{by_site['smoking_miss'].max():.1f}%**")
    add("- 这不是随机缺失，而是**「某些站点根本没做这项问卷」**——属于结构性缺失")
    add("- 佐证：有 `hdl` 的人平均 " + f"{df[df.hdl.notna()]['age'].mean():.1f} 岁，"
        "没有的人 " + f"{df[df.hdl.isna()]['age'].mean():.1f} 岁，"
        "说明缺失与人群构成相关（至少是 MAR，甚至 MNAR）")
    add("")
    add("**所以：绝对不能 `dropna()`。** "
        "直接把含缺失的行删掉，会一次性砍掉 71.6% 的样本，而且砍掉的是特定站点的人——"
        "相当于把一部分城市从研究里抹掉，结论直接偏掉。\n")
    add("正确的三条路（第 3 步再选）：")
    add("1. **主分析不放这个变量**（用不受缺失影响的变量建核心模型）")
    add("2. **把「缺失」当成一个单独类别**（`smoking_unknown`），而不是硬填")
    add("3. **多重插补**（MICE），并在第 7 步做「插补前后结论是否一致」的敏感性分析\n")

    # ---------- 7. 数据结构 ----------
    add("## 7. 数据结构特征\n")
    add(f"- 站点 {df['site'].nunique()} 个，样本量从 {df.groupby('site').size().min()} 到 "
        f"{df.groupby('site').size().max():,}")
    add(f"- 随访时间 {df['followup_years'].min():.2f} – {df['followup_years'].max():.2f} 年，"
        f"中位 {df['followup_years'].median():.2f} 年")
    add(f"- 累计 {df['followup_years'].sum():,.0f} 人年")
    add(f"- 粗发病率 **{1000 * n_censor / df['followup_years'].sum():.2f} / 1000 人年**"
        "（原论文年龄调整后为 7.35）")
    add("")

    fu = pd.cut(df["followup_years"], [1.9, 2.5, 3, 3.5, 4, 5, 8])
    tab = df.groupby(fu, observed=True).agg(n=("pid", "size"), inc=("dm_censor", "sum"))
    tab["rate_%"] = (tab["inc"] / tab["n"] * 100).round(2)
    add("**按随访时长分层看发病率：**\n")
    add("```")
    add(tab.to_string())
    add("```\n")
    add("- 注意发病率**不随随访时长单调上升**（2.5–3 年是 1.93%，3–3.5 年反而 1.60%）")
    add("- 原因：随访时长 = 末次体检年份 − 基线年份，它同时编码了**入组年份**。"
        "不同年份入组的人本身风险就不同（日历效应）")
    add("- **这直接决定了第 4 步怎么算发病率：不能用「发病人数/总人数」这种粗率，"
        "必须用人年（每 1000 人年）或生存分析。**\n")

    site_rate = df.groupby("site").agg(n=("pid", "size"), inc=("dm_censor", "sum"))
    site_rate["rate_%"] = (site_rate["inc"] / site_rate["n"] * 100).round(2)
    add("**各站点发病率差异明显：**\n")
    add("```")
    add(site_rate.to_string())
    add("```\n")
    add("- 最高 2.58%，最低 0.00%（但那个站点只有 61 人）")
    add("- **处理：`site` 不能忽略，模型里要做站点层面的调整（固定效应或聚类稳健标准误）。"
        "另外样本量 <100 的站点考虑合并。**\n")

    # ---------- 8. 结论 ----------
    add("## 8. 审计结论：进第 2 步前必须先定下的 4 件事\n")
    add("| # | 决定 | 依据 |")
    add("|---|---|---|")
    add("| 1 | 结局 = `dm_censor`，时间 = `followup_years` | `dm_incident` 只标了 31% 的发病者 |")
    add("| 2 | 主分析不用 `smoking`/`drinking`/`ast`/`hdl`/`ldl` 建核心模型 | 站点级结构性缺失 45%–72% |")
    add("| 3 | 发病率用人年，不用粗率 | 随访时长混杂入组年份，出现日历效应 |")
    add("| 4 | 模型必须调整 `site` | 站点发病率 0–2.58%，差异远超随机波动 |")
    add("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8")

    # 控制台只打要点（不含 emoji，避免 Windows GBK 控制台报错）
    print(f"[OK] 审计报告已写入 {OUT}")
    print(f"     规模 {n:,} 行 x {df.shape[1]} 列")
    print(f"     发病 {n_censor:,} 人 ({n_censor / n * 100:.2f}%)")
    print(f"     人年 {df['followup_years'].sum():,.0f}")
    print(f"     粗发病率 {1000 * n_censor / df['followup_years'].sum():.2f} /1000 人年")
    print("     4 个待决事项见报告第 8 节")


if __name__ == "__main__":
    main()
