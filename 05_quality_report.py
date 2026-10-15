"""
第 5 步：数据质量报告

对清洗前后的数据做逐列体检，产出一份可以直接贴进 README 的报告。

这个项目要做的关键区分：
数据里看着"脏"的地方，其实是三种完全不同的问题，混淆它们会得出错误结论——

  1. 伪装缺失：值是 0，但这个指标在生理上不可能为 0
     （人活着血液里就有胰岛素，0 只可能是"没测/没录"）
  2. 真实缺失：值本身就是空的（NaN / NA）
  3. 异常值：有值，但超出了生理合理范围

只有第 1 类需要"识别并填充"，第 2 类要单独计数，第 3 类要人工复核。
把它们混在一起统计，是新手最容易犯的错。

运行：python 05_quality_report.py
输出：docs/quality_report.md + 控制台摘要
"""
import os
import pandas as pd
from datetime import datetime

RAW = "data/diabetes.csv"
CLEAN = "data/diabetes_clean.csv"
OUT_DIR = "docs"
OUT = os.path.join(OUT_DIR, "quality_report.md")

# 每列的医学元数据：中文名 / 单位 / 生理上是否可能为 0 / 合理范围
# 范围用于筛"异常值"：超出上下限的记录要人工复核，不能自动填
COLS = [
    ("Pregnancies",                "妊娠次数",           "次",      True,  0,    20),
    ("Glucose",                    "血糖",               "mg/dL",   False, 20,   500),
    ("BloodPressure",              "舒张压",             "mmHg",    False, 20,   200),
    ("SkinThickness",               "三头肌皮褶厚度",     "mm",      False, 1,    100),
    ("Insulin",                    "血清胰岛素",         "μU/mL",   False, 1,    900),
    ("BMI",                         "体质指数",           "kg/m²",   False, 10,   70),
    ("DiabetesPedigreeFunction",   "糖尿病家族史函数",   "—",       False, 0.01, 3),
    ("Age",                        "年龄",               "岁",      False, 1,    120),
    ("Outcome",                    "是否患病",           "0/1",     True,  0,    1),
]

# 为什么这些列不可能为 0 —— 报告里会原样输出，面试时可讲
ZERO_REASON = {
    "Glucose": "人活着血液里就有葡萄糖，为 0 只可能是未检测或未录入",
    "BloodPressure": "血压为 0 意味着临床死亡，不可能出现在门诊样本中",
    "SkinThickness": "皮褶再薄也有厚度，为 0 是测量缺失",
    "Insulin": "正常人血清胰岛素不可能为 0，为 0 代表该项未查",
    "BMI": "体重与身高均不可能为 0",
}


def scan(df):
    """逐列统计三类数据问题"""
    rows = []
    for col, cn, unit, zero_ok, lo, hi in COLS:
        s = df[col]
        n = len(s)
        nulls = int(s.isna().sum())
        zeros = int((s == 0).sum())
        valid = s.dropna()
        # 真实有值部分：把伪装缺失的 0 剔掉再统计
        real = valid[valid != 0] if not zero_ok else valid
        out = int(((real < lo) | (real > hi)).sum())

        if not zero_ok and zeros > 0:
            kind, action = "伪装缺失", "识别为缺失 → 用中位数填充"
        elif nulls > 0:
            kind, action = "真实缺失", "用中位数填充"
        elif out > 0:
            kind, action = "异常值", "标记待人工复核（不自动修改）"
        else:
            kind, action = "正常", "无需处理"

        row = {
            "字段": col, "中文名": cn, "单位": unit,
            "缺失": nulls, "0值": zeros,
            "0值占比": f"{zeros / n * 100:.1f}%",
            "超范围": out,
            "问题类型": kind, "处理动作": action,
            "依据": ZERO_REASON.get(col, "—"),
            "有效样本": int(len(real)),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    if not os.path.exists(RAW):
        print(f"找不到 {RAW}，请先运行 prepare_data.py")
        return
    if not os.path.exists(CLEAN):
        print(f"找不到 {CLEAN}，请先运行 01_clean.py")
        return

    raw = pd.read_csv(RAW)
    clean = pd.read_csv(CLEAN)
    report = scan(raw)
    after = scan(clean)

    os.makedirs(OUT_DIR, exist_ok=True)
    lines = []
    lines.append("# 数据质量报告")
    lines.append("")
    lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}　｜　"
                 f"数据源：`{RAW}`（{len(raw)} 行 × {raw.shape[1]} 列）")
    lines.append("")
    lines.append("## 一、逐列体检")
    lines.append("")
    lines.append("| 字段 | 中文名 | 单位 | 空值 | 0 值 | 0 值占比 | 超出生理范围 | 问题类型 | 处理动作 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in report.iterrows():
        lines.append(f"| `{r['字段']}` | {r['中文名']} | {r['单位']} | {r['缺失']} | "
                   f"{r['0值']} | {r['0值占比']} | {r['超范围']} | "
                   f"{r['问题类型']} | {r['处理动作']} |")
    lines.append("")
    lines.append("## 二、为什么 0 不能当正常值")
    lines.append("")
    lines.append("| 字段 | 生理上为 0 意味着 | 真实成因 |")
    lines.append("|---|---|---|")
    for col, reason in ZERO_REASON.items():
        lines.append(f"| `{col}` | 不可能 | {reason} |")
    lines.append("")
    lines.append("## 三、处理结果与残留问题")
    lines.append("")
    lines.append(f"- 清洗前 0 值合计：**{int((raw[list(ZERO_REASON)].eq(0)).sum().sum())} 个**")
    lines.append(f"- 清洗后剩余缺失值：**{int(clean.isna().sum().sum())} 个**")
    lines.append(f"- 样本量：{len(raw)} → {len(clean)} 条（未删行）")
    lines.append("")
    lines.append("## 四、填充不等于增加信息（自我批评）")
    lines.append("")
    lines.append("这是很多人会忽略的一点：把 0 填成中位数，**表格填满了，但信息并没有变多**。")
    lines.append("")
    lines.append("| 字段 | 原本有值 | 伪装缺失（被填充） | 实际可用于推断的样本 |")
    lines.append("|---|---|---|---|")
    for _, r in report[report["问题类型"] == "伪装缺失"].iterrows():
        lines.append(f"| `{r['字段']}` | {r['有效样本']} 条 | {r['0值']} 条（{r['0值占比']}） "
                   f"| **{r['有效样本']} 条** |")
    lines.append("")
    lines.append("**结论：**胰岛素列名义上有 768 条，但真正观测到的只有 394 条。")
    lines.append("所以本项目**没有把胰岛素列为核心风险指标**，只标注为")
    lines.append("「受数据质量影响的次要变量」——这是有依据的取舍，不是回避。")
    lines.append("")
    lines.append("**如果要正式分析，正确做法是：**")
    lines.append("")
    lines.append("- **多重插补（MICE）**：用其他变量回归预测缺失值、重复多次取平均，不压缩方差")
    lines.append("- **分层填充**：按临床特征分组（如先按血糖分层）再分别取中位数")
    lines.append("- 本项目为演示，采用简单中位数填充并主动说明局限")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("=" * 56)
    print("【数据质量报告】")
    print("=" * 56)
    print(f"样本：{len(raw)} 行 × {raw.shape[1]} 列")
    print(f"患病 {(raw['Outcome'] == 1).sum()} 条 | "
          f"未患病 {(raw['Outcome'] == 0).sum()} 条")
    print()
    print("三类问题分布：")
    for kind, n in report["问题类型"].value_counts().items():
        print(f"  {kind:<6} {n} 列")
    print()
    print("逐列明细：")
    for _, r in report.iterrows():
        print(f"  {r['中文名']:<16} 0 值 {r['0值']:>4} 个（{r['0值占比']:>6}） "
              f"→ {r['问题类型']}")
    print()
    print("填充了多少（= 这些值是猜的，不是测的）：")
    for _, r in report[report["问题类型"] == "伪装缺失"].iterrows():
        print(f"  {r['中文名']:<16} 填充 {r['0值']:>4} 条 / 768 条"
              f"（{r['0值占比']}）→ 实际有效样本 {r['有效样本']} 条")
    print()
    print(f"清洗后剩余缺失值：{int(clean.isna().sum().sum())} 个")
    print(f"报告已输出：{OUT}")


if __name__ == "__main__":
    main()
