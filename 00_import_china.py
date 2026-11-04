"""导入中国体检队列数据（瑞慈 2010-2016）。

源文件：RC Health Care Data-20180820.xlsx
来源：Dryad, doi:10.5061/dryad.ft8750v，许可 CC0 1.0（公共领域）
原研究：Chen Y, et al. BMJ Open 2018;8:e021768
        211,833 名中国成年人，11 城 32 站点，2010-2016 体检，中位随访 3.1 年

本脚本只做两件事：
1. 把 27 MB 的 xlsx 转成 CSV（后续分析读 CSV 快得多）
2. 打印真实的数据核查结果——列名、缺失率、结局分布、可疑值

不做任何清洗和建模。清洗要看到核查结果之后再写，避免凭猜测处理。
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "data" / "raw_china" / "RC_Health_Care_Data_20180820.xlsx"
DST = ROOT / "data" / "raw_china" / "rc_china.csv"

# 原始表头带括号和中文逗号，改名成可写代码的形式
RENAME = {
    "id": "pid",
    "Age (y)": "age",
    "Gender(1, male; 2, female)": "gender",
    "site": "site",
    "height(cm)": "height",
    "weight(kg)": "weight",
    "BMI(kg/m2)": "bmi",
    "SBP(mmHg)": "sbp",
    "DBP(mmHg)": "dbp",
    "FPG (mmol/L)": "fpg",
    "Cholesterol(mmol/L)": "tc",
    "Triglyceride(mmol/L)": "tg",
    "HDL-c(mmol/L)": "hdl",
    "LDL(mmol/L)": "ldl",
    "ALT(U/L)": "alt",
    "AST(U/L)": "ast",
    "BUN(mmol/L)": "bun",
    "CCR(umol/L)": "ccr",
    "FPG of final visit(mmol/L)": "fpg_last",
    "Diabetes diagnosed during followup（1,Yes）": "dm_incident",
    "censor of diabetes at followup(1, Yes; 0, No)": "dm_censor",
    "year of followup": "followup_years",
    "smoking status(1,current smoker;2, ever smoker;3,never smoker)": "smoking",
    "drinking status(1,current drinker;2, ever drinker;3,never drinker)": "drinking",
    "family histroy of diabetes(1,Yes;0,No)": "family_history",
}


def convert() -> list[str]:
    """流式转换 xlsx → csv，返回标准列名。"""
    wb = openpyxl.load_workbook(SRC, read_only=True, data_only=True)
    ws = wb["RC"]
    it = ws.iter_rows(values_only=True)
    header = list(next(it))
    cols = [RENAME.get(h, str(h)) for h in header]

    n = 0
    with DST.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for row in it:
            w.writerow(["" if v is None else v for v in row])
            n += 1
            if n % 50000 == 0:
                print(f"  已写 {n:,} 行", flush=True)
    wb.close()
    print(f"转换完成：{n:,} 行 × {len(cols)} 列 → {DST}")
    return cols


def main() -> None:
    if not SRC.exists():
        sys.exit(f"找不到源文件：{SRC}")

    if DST.exists():
        print(f"CSV 已存在，跳过转换：{DST}")
    else:
        print(f"源文件 {SRC.stat().st_size / 1048576:.2f} MB，开始转换 …")
        convert()

    import pandas as pd

    df = pd.read_csv(DST)
    n, p = df.shape
    print()
    print("=" * 70)
    print(f"数据规模：{n:,} 行 × {p} 列")
    print("=" * 70)

    print("\n【列名与缺失率】")
    for c in df.columns:
        miss = df[c].isna().sum()
        print(f"  {c:<14} 缺失 {miss:>7,} ({miss / n * 100:5.1f}%)   "
              f"dtype={str(df[c].dtype):<8} 样例={df[c].dropna().iloc[0] if df[c].notna().any() else '全空'}")

    print("\n【结局变量】")
    if "dm_incident" in df.columns:
        vc = df["dm_incident"].value_counts(dropna=False).to_dict()
        print(f"  dm_incident 分布：{vc}")
    if "dm_censor" in df.columns:
        print(f"  dm_censor 分布：{df['dm_censor'].value_counts(dropna=False).to_dict()}")

    print("\n【关键连续变量描述】")
    keys = ["age", "bmi", "height", "weight", "sbp", "dbp", "fpg", "fpg_last",
            "tc", "tg", "hdl", "ldl", "alt", "ast", "bun", "ccr", "followup_years"]
    print(f"  {'变量':<14}{'有效数':>9}{'均值':>10}{'标准差':>10}{'最小':>9}{'中位':>9}{'最大':>10}")
    for c in keys:
        if c not in df.columns:
            continue
        s = df[c].dropna()
        print(f"  {c:<14}{len(s):>9,}{s.mean():>10.2f}{s.std():>10.2f}"
              f"{s.min():>9.2f}{s.median():>9.2f}{s.max():>10.2f}")

    print("\n【分类变量】")
    for c in ["gender", "smoking", "drinking", "family_history"]:
        if c in df.columns:
            print(f"  {c:<15}{df[c].value_counts(dropna=False).sort_index().to_dict()}")

    print(f"\n【站点】共 {df['site'].nunique()} 个站点，样本量前 10：")
    print("  " + str(df["site"].value_counts().head(10).to_dict()))

    print("\n【可疑值检查】")
    checks = [
        ("bmi", (df["bmi"] < 10) | (df["bmi"] > 60)),
        ("age", (df["age"] < 18) | (df["age"] > 110)),
        ("fpg", (df["fpg"] <= 0) | (df["fpg"] > 40)),
        ("height", (df["height"] < 120) | (df["height"] > 220)),
        ("sbp", (df["sbp"] < 50) | (df["sbp"] > 260)),
    ]
    for c, mask in checks:
        print(f"  {c:<8} 越界 {int(mask.sum()):>6,} 条")


if __name__ == "__main__":
    main()
