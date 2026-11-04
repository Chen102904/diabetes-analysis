# -*- coding: utf-8 -*-
"""
CHARLS 2015 数据导入与体检脚本
================================
作用：把从官网下载的 CHARLS 压缩包/Stata 文件一键整理成可用的中间数据，
      并在动任何分析之前，先把「真实列名、缺失编码、关键变量是否存在」全部打印清楚。

用法：
    python 00_import_charls.py          # 扫描 + 解压 + 体检（不写 CSV）
    python 00_import_charls.py --build  # 额外生成 data/charls_merged.csv 供后续脚本读

设计原则：
    1. 不猜列名。所有变量名从真实 .dta 里读出来再匹配。
    2. 不猜缺失码。CHARLS 用负数（-1 拒答 / -2 不知道 / -3 拒绝等）表示缺失，
       和 Pima 用 0 表示缺失完全相反，这一步必须打印出来人工确认。
    3. 原始数据绝不入库。data/raw_charls/ 已在 .gitignore 中。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------- 路径

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw_charls"
OUT_CSV = ROOT / "data" / "charls_merged.csv"
MANIFEST = ROOT / "data" / "charls_manifest.json"

# 官网下载的压缩包通常落在系统下载目录，一起扫
CANDIDATE_DIRS = [
    RAW_DIR,
    Path.home() / "Downloads",
    Path.home() / "Desktop",
]

# ---------------------------------------------------------------- 关键变量
# 左：逻辑名  右：变量名（2015 码表 VersionID 20190620 已逐一核实）
# 末尾带 * 表示前缀匹配（CHARLS 大量使用数组型变量，如 da007_w2_2_1_ … _14_）
KEY_VARS = {
    # —— Blood Data 数据集（血检，2019 才单独发布）
    "血糖":           ["bl_glu"],
    "糖化血红蛋白":   ["bl_hbalc"],
    "总胆固醇":       ["bl_cho"],
    "甘油三酯_HLD":   ["bl_hdl"],
    "低密度脂蛋白":   ["bl_ldl"],
    "尿酸":           ["bl_ua"],
    "C反应蛋白":      ["bl_crp"],
    # —— Biomarker 数据集（人体测量）
    "身高":           ["qi002"],
    "体重":           ["ql002"],
    "腰围":           ["qm002"],
    "收缩压":         ["bp_systolic*", "qi0*"],
    # —— Demographic_Background
    "性别":           ["ba000_w2_3", "ba000_w2_4"],
    "出生年":         ["ba002_1"],
    # —— Health_Status_and_Functioning
    "糖尿病确诊史":   ["da007_w2_2_*", "da007_w2_1_*"],
    "糖尿病用药":     ["da014s*"],
    # —— Weights / Sample_Info
    "权重":           ["indweight*", "r_indweight*", "weight*"],
    "访谈日期":       ["iwy*", "interviewdate*"],
}

# 这些值在 CHARLS 里是「失效码」而不是真实测量值，混进均值会把结果拉飞
# 例：身高 qi002 的 Max 是 993.0、体重 ql002 的 Max 是 993.0
INVALID_CODES = {993.0, 994.0, 995.0, 996.0, 998.0, 999.0, 9999.0}

DTYPE_HINT = ("*", "float", "int", "str", "object", "category")


def banner(t: str) -> None:
    print("\n" + "=" * 68)
    print(t)
    print("=" * 68)


# ---------------------------------------------------------------- 解压

def extract_archives() -> list[Path]:
    """把下载目录里的 zip 解压到 raw_charls/，返回解压出的文件。"""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []

    seen: set[Path] = set()
    for d in CANDIDATE_DIRS:
        if not d.exists():
            continue
        for z in list(d.glob("*.zip")) + list(d.glob("*.ZIP")):
            resolved = z.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                with zipfile.ZipFile(z) as zf:
                    names = zf.namelist()
                    if not any(n.lower().endswith(".dta") for n in names):
                        print(f"  跳过 {z.name}（里面没有 .dta）")
                        continue
                    print(f"  解压 {z.name} …（{len(names)} 个文件）")
                    zf.extractall(RAW_DIR / z.stem)
                    extracted.append(z)
            except zipfile.BadZipFile:
                print(f"  ⚠ {z.name} 不是有效的 zip，可能没下载完整")

    for d in CANDIDATE_DIRS:
        if not d.exists():
            continue
        for r in list(d.glob("*.rar")) + list(d.glob("*.7z")):
            print(f"  ⚠ {r.name} 是 rar/7z，脚本不能自动解压，请手动解压到 {RAW_DIR}")

    return extracted


def find_dta() -> list[Path]:
    if not RAW_DIR.exists():
        return []
    return sorted(p for p in RAW_DIR.rglob("*.dta") if p.is_file())


# ---------------------------------------------------------------- 读取

def read_dta(path: Path) -> pd.DataFrame | None:
    """读 Stata。先试不做类别转换（保留数字码），失败再退回默认。"""
    for convert in (False, True):
        try:
            df = pd.read_stata(path, convert_categoricals=convert)
            return df
        except Exception as e:  # noqa: BLE001
            last = e
    print(f"  ⚠ 读不了 {path.name}: {type(last).__name__}: {last}")
    return None


def _match(cols: list[str], alias: str) -> list[str]:
    """alias 支持前缀匹配（以 * 结尾）。返回所有命中的真实列名。"""
    low = [c.lower() for c in cols]
    if alias.endswith("*"):
        pre = alias[:-1].lower()
        return [cols[i] for i, c in enumerate(low) if c.startswith(pre)]
    return [cols[low.index(alias.lower())]] if alias.lower() in low else []


def profile(df: pd.DataFrame, path: Path) -> dict:
    """打印单文件的体检结果，返回摘要。"""
    n, p = df.shape
    print(f"\n▸ {path.name}   行 {n:,} × 列 {p}")

    cols = list(df.columns)
    print(f"  前 12 列: {', '.join(cols[:12])}")

    # 负数码（CHARLS 的缺失表示）
    num = df.select_dtypes("number")
    if not num.empty:
        neg_cols = []
        for c in num.columns:
            v = num[c]
            m = (v < 0).sum()
            if m:
                neg_cols.append((c, int(m), sorted({int(x) for x in v[v < 0].dropna().unique()})[:4]))
        if neg_cols:
            print(f"  ⚠ 负数缺失码：{len(neg_cols)} 列，前 5 列：")
            for c, m, codes in neg_cols[:5]:
                print(f"      {c:<22} 负数 {m:>6,} 条   取值 {codes}")

        # 993/999 这类「伪数值」失效码，混进均值会把统计量拉飞
        bad = []
        for c in num.columns:
            v = num[c].dropna()
            m = v.isin(INVALID_CODES).sum()
            if m:
                bad.append((c, int(m), sorted({float(x) for x in v[v.isin(INVALID_CODES)].unique()})[:3]))
        if bad:
            print(f"  ⚠ 发现失效码 993/999 等：{len(bad)} 列，前 5 列：")
            for c, m, codes in bad[:5]:
                print(f"      {c:<22} {m:>6,} 条   取值 {codes}  ← 分析前必须置空")

    # 关键变量命中
    hit = {}
    for label, aliases in KEY_VARS.items():
        for a in aliases:
            got = _match(cols, a)
            if got:
                hit[label] = got if len(got) > 1 else got[0]
                break
    if hit:
        shown = ", ".join(f"{k}={v if isinstance(v, str) else str(v[:3]) + ('…' if len(v) > 3 else '')}"
                          for k, v in hit.items())
        print(f"  ✓ 命中 {len(hit)} 个：{shown}")
    return {"file": path.name, "rows": n, "cols": p, "columns": cols, "key_hits": hit}


# ---------------------------------------------------------------- 合并

def build_merged(frames: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    """按 ID 合并。CHARLS 各表的主键是 ID（跨年不变）。"""
    id_col = None
    for name, df in frames.items():
        for c in df.columns:
            if c.upper() == "ID":
                id_col = c
                break
        if id_col:
            break

    if not id_col:
        print("\n⚠ 没找到 ID 列，无法自动合并。请手动确认各表主键。")
        return None

    usable = {k: v for k, v in frames.items() if id_col in v.columns}
    if len(usable) < 2:
        print(f"\n只有 {len(usable)} 张表含 ID 列，不需要合并。")
        return list(usable.values())[0] if usable else None

    print(f"\n按 {id_col} 合并 {len(usable)} 张表…")
    out = None
    for name, df in usable.items():
        # 去重列时保留左表，避免 _x/_y 满地跑
        if out is None:
            out = df
        else:
            dup = [c for c in df.columns if c in out.columns and c != id_col]
            out = out.merge(df.drop(columns=dup), on=id_col, how="left")
        print(f"  + {name:<34} → {out.shape[0]:,} 行 × {out.shape[1]} 列")
    return out


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="生成合并后的 CSV")
    args = ap.parse_args()

    banner("1. 扫描下载文件")
    if not RAW_DIR.exists():
        RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  扫描目录：")
    for d in CANDIDATE_DIRS:
        print(f"    {'✓' if d.exists() else '✗'} {d}")

    arches = extract_archives()
    dta = find_dta()

    if not dta:
        banner("结果：还没找到数据文件")
        print("  没有在以下位置发现 .dta：")
        for d in CANDIDATE_DIRS:
            print(f"    {d}")
        print("\n  下一步：")
        print("    1. 浏览器登录 CHARLS 后下载 2015 年的数据包（zip）")
        print(f"    2. 把 zip 丢在 下载 文件夹，或解压到 {RAW_DIR}")
        print("    3. 重新运行本脚本")
        return 1

    banner(f"2. 读取 {len(dta)} 个 Stata 文件")
    frames: dict[str, pd.DataFrame] = {}
    summaries = []
    for p in dta:
        df = read_dta(p)
        if df is None:
            continue
        frames[p.stem] = df
        summaries.append(profile(df, p))

    # 全局关键变量覆盖情况
    banner("3. 关键变量覆盖情况")
    covered: dict[str, list[str]] = {k: [] for k in KEY_VARS}
    for s in summaries:
        for label, col in s["key_hits"].items():
            names = col if isinstance(col, list) else [col]
            covered[label].append(f"{s['file'].replace('.dta', '')}::{names[0]}"
                                  + (f"(共{len(names)}列)" if len(names) > 1 else ""))
    for label, where in covered.items():
        mark = "✓" if where else "✗"
        print(f"  {mark} {label:<14} {', '.join(where) if where else '（未找到，需人工核对码表）'}")

    # 缺失情况
    banner("4. 原始数据规模")
    total = max(s["rows"] for s in summaries)
    print(f"  单表最大行数：{total:,}")
    print(f"  文件数：{len(summaries)}")

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        json.dumps({"files": summaries, "generated_by": "00_import_charls.py"},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n  体检报告已写入：{MANIFEST.relative_to(ROOT)}")

    if args.build:
        banner("5. 合并")
        merged = build_merged(frames)
        if merged is not None:
            merged.to_csv(OUT_CSV, index=False)
            print(f"\n  已写出 {OUT_CSV.relative_to(ROOT)}   {merged.shape[0]:,} 行 × {merged.shape[1]} 列")
            print("  ⚠ 这份 CSV 含个体微观数据，不要提交到 GitHub")

    banner("完成")
    print("  下一步：确认上面 ✗ 的变量在真实列名里叫什么，再改 01_clean.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
