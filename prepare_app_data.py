# -*- coding: utf-8 -*-
"""
生成看板用的精简数据文件（data/china_app.parquet）。

为什么不用 29MB 的 china_analysis.csv？
- 看板只用到 13 个列，其余 17 列（hdl/ldl/ast/bun/ccr/fpg_last 等）不需要
- parquet 列式压缩 + float32 降精度，把 29MB 压到 ~2MB，能进 Git 和云端
- 这样 Streamlit Cloud 部署时，GitHub 上有数据文件可读，不会 FileNotFoundError

运行：python prepare_app_data.py
"""
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
SRC = BASE_DIR / "data" / "china_analysis.csv"
DST = BASE_DIR / "data" / "china_app.parquet"

# 看板实际用到的列（与 04_app.py 里 grep 出的一致）
COLS = ["age", "gender", "site", "bmi", "bmi_cat", "age_grp",
        "dm_censor", "followup_years", "fpg", "sbp", "dbp", "tg", "tc"]

df = pd.read_csv(SRC, usecols=COLS)

# 降精度省空间（不影响看板展示，误差在 float32 精度内可忽略）
for c in ["age", "site", "bmi", "fpg", "sbp", "dbp", "tg", "tc", "followup_years"]:
    df[c] = df[c].astype("float32")
df["gender"] = df["gender"].astype("int8")
df["dm_censor"] = df["dm_censor"].astype("int8")

df.to_parquet(DST, index=False, compression="zstd")

size_mb = DST.stat().st_size / 1024 / 1024
print(f"已生成 {DST.name}：{df.shape[0]:,} 行 × {df.shape[1]} 列，{size_mb:.2f} MB")
print(f"（原始 CSV 29MB，压缩至 {size_mb:.2f}MB，比例 {size_mb/29*100:.1f}%）")
