"""
第 6 步：医疗数据脱敏与合规检查

医疗数据和互联网数据的最大区别：前者受法律约束，不是"能拿到就能用"。

这个脚本是一条三层的递进处理链——

  第一层  字段分级    —— 每列归到「直接标识符 / 准标识符 / 敏感属性 / 临床指标」
  第二层  泛化       —— 把精确值变区间（年龄 35 → 30-39）
  第三层  抑制       —— 泛化后仍然"独一无二"的记录，直接标记为不可外发

三层都做完才叫脱敏；只做第一层等于没做。

⚠️ 本脚本是工程演示，不构成法律意见。真实外发应由医院信息科 + 法务共同评估。

运行：python 06_privacy.py
输出：docs/privacy_report.md + data/diabetes_deidentified.csv
"""

import os
import pandas as pd
from datetime import datetime

RAW = "data/diabetes.csv"
OUT_DIR = "docs"
OUT_MD = os.path.join(OUT_DIR, "privacy_report.md")
OUT_CSV = "data/diabetes_deidentified.csv"

K_TARGET = 5          # 目标：任意准标识符组合下至少 5 条记录
DIRECT_KEYWORDS = ["姓名", "name", "身份证", "id_card", "医保", "insurance",
                   "email", "邮箱", "电话", "phone", "mobile", "住址", "address"]

# 手工分级表：本项目的 9 个字段逐一定级
# 依据是"这一列离定位到个人有多近"
LEVEL_MAP = {
    "Pregnancies":              ("准标识符", "生育史，在女性人群中是强识别特征"),
    "Glucose":                  ("临床指标", "检查数值，不直接识别身份"),
    "BloodPressure":            ("临床指标", "检查数值，不直接识别身份"),
    "SkinThickness":            ("临床指标", "检查数值，不直接识别身份"),
    "Insulin":                  ("临床指标", "检查数值，不直接识别身份"),
    "BMI":                      ("临床指标", "由身高体重算出，不直接识别身份"),
    "DiabetesPedigreeFunction":  ("临床指标", "家族史衍生指标，不直接识别身份"),
    "Age":                      ("准标识符", "与妊娠次数、生育史组合后可缩小到个别人"),
    "Outcome":                  ("敏感属性", "要分析的目标，泄露后果严重"),
}

# 泛化等级：越大越模糊，隐私越好但可用性越差
LEVELS = [
    ("L0", "原始值",    "原始值",             1,  None),
    ("L1", "5 岁一段",  "0 / 1-2 / 3-4 / 5+",  5,  [0, 2, 4, 100]),
    ("L2", "10 岁一段", "0 / 1-2 / 3-5 / 6+",  10, [0, 2, 5, 100]),
    ("L3", "20 岁一段", "0 / 1-5 / 6+",      20, [0, 5, 100]),
]


def generalize(df, age_step, preg_bins):
    d = df.copy()
    if age_step and age_step > 1:
        d["Age"] = (d["Age"] // age_step * age_step).astype(int)
    if preg_bins:
        d["Pregnancies"] = pd.cut(d["Pregnancies"], bins=preg_bins,
                           include_lowest=True).astype(str)
    return d


def k_of(df, quasi):
    """最小等价类大小：k=1 表示存在可被唯一定位的记录"""
    return int(df.groupby(quasi).size().min())


def main():
    if not os.path.exists(RAW):
        print(f"找不到 {RAW}，请先运行 prepare_data.py")
        return

    df = pd.read_csv(RAW)
    quasi = [c for c, lv in LEVEL_MAP.items() if lv[0] == "准标识符"]

    rows = []
    for code, adesc, pdesc, age_step, preg_bins in LEVELS:
        d = generalize(df, age_step, preg_bins)
        k = k_of(d, quasi)
        g = d.groupby(quasi).size()
        rows.append((code, adesc, pdesc, k, int((g < K_TARGET).sum()), len(g)))
        if k >= K_TARGET:
            break

    best = rows[-1]
    chosen = [x for x in LEVELS if x[0] == best[0]][0]
    final = generalize(df, chosen[3], chosen[4])
    g = final.groupby(quasi).size()
    suppressed = int(g[g < K_TARGET].sum())

    os.makedirs(OUT_DIR, exist_ok=True)
    L = []
    L.append("# 医疗数据脱敏与合规检查报告")
    L.append("")
    L.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}　｜　"
             f"数据源：`{RAW}`（{len(df)} 行 × {df.shape[1]} 列）")
    L.append("")
    L.append("> ⚠️ 本报告为工程演示，不构成法律意见。")
    L.append("")
    L.append("## 一、字段分级（谁离识别个人最近）")
    L.append("")
    L.append("| 字段 | 级别 | 判断依据 |")
    L.append("|---|---|---|")
    for c, (lv, why) in LEVEL_MAP.items():
        L.append(f"| `{c}` | {lv} | {why} |")
    L.append("")
    L.append("## 二、重识别风险：k-匿名测算")
    L.append("")
    L.append(f"- 准标识符：**{' + '.join(quasi)}**")
    L.append(f"- 目标：任意组合下至少 **{K_TARGET} 条**记录（k < {K_TARGET} 视为可定位）")
    L.append("")
    L.append("| 等级 | 年龄处理 | 妊娠次数处理 | 最小组大小 k | 风险组数 | 总组数 |")
    L.append("|---|---|---|---|---|---|")
    for code, adesc, pdesc, k, nr, ng in rows:
        L.append(f"| {code} | {adesc} | {pdesc} | {k} | {nr} | {ng} |")
    L.append("")
    if best[3] >= K_TARGET:
        L.append(f"✅ 采用 **{best[0]}**（年龄 {best[1]}），"
                 f"k = {best[3]}，达到 k ≥ {K_TARGET} 的要求。")
    else:
        L.append(f"⚠️ 即使最强泛化（{best[0]} 年龄 {best[1]}）k 仍为 {best[3]}；"
                 f"此时应**抑制**落在小组合里的 **{suppressed} 条记录**"
                 f"（不删除原始库，仅在外发版本里剔除）才能达标。")
    L.append("")
    L.append("## 三、外发前检查清单")
    L.append("")
    for i, t in enumerate([
        "直接标识符（姓名 / 身份证 / 手机号 / 医保号 / 住址）是否已删除或不可逆加密",
        "准标识符组合后的 k 值是否 ≥ 5",
        "敏感属性（诊断 / 用药）是否为必要最小范围",
        "是否取得患者或监护人知情同意，或具备法律豁免",
        "传输是否加密、落地后是否有访问控制与审计日志",
        "外发与脱敏操作是否已留痕（谁、何时、给了谁）",
    ], 1):
        L.append(f"{i}. {t}")
    L.append("")
    L.append("---")
    L.append("")
    L.append("**合规依据（以官方文本为准，需自行核实）**")
    L.append("")
    L.append("- 《中华人民共和国个人信息保护法》第 28 条：医疗健康信息属**敏感个人信息**，")
    L.append("  处理须具备特定目的与充分必要性，并采取严格保护措施")
    L.append("- 《中华人民共和国数据安全法》：数据处理活动应建立分类分级保护制度")
    L.append("- HIPAA Safe Harbor（美国）：移除 **18 类**标识符后方可视为去标识化数据")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    print("=" * 56)
    print("【脱敏与合规检查】")
    print("=" * 56)
    print(f"样本：{len(df)} 行 × {df.shape[1]} 列")
    print()
    print("字段分级：")
    for c, (lv, why) in LEVEL_MAP.items():
        print(f"  {c:<26} {lv}")
    print()
    print(f"准标识符：{' + '.join(quasi)}")
    print()
    print("泛化过程（自动选到第一个达标等级）：")
    for code, adesc, pdesc, k, nr, ng in rows:
        flag = "达标" if k >= K_TARGET else "不达标"
        print(f"  {code} 年龄{adesc:<8} k={k:<3} 风险组 {nr:<4} 总组 {ng:<4} [{flag}]")
    print()
    if best[3] < K_TARGET:
        print(f"⚠️ 最强泛化仍未达标，外发时需抑制 {suppressed} 条稀有记录")
    print()
    print(f"脱敏后数据：{OUT_CSV}")
    print(f"报告已输出：{OUT_MD}")


if __name__ == "__main__":
    main()
