"""
给 Pima 糖尿病原始数据加上列名，生成可直接使用的 CSV。
原始文件无表头，按 UCI 官方定义的 8 个特征 + 1 个标签排列。
"""
import csv

# UCI 官方列名定义（来源：UCI Machine Learning Repository - Pima Indians Diabetes）
COLUMNS = [
    "Pregnancies",              # 怀孕次数
    "Glucose",                  # 口服葡萄糖耐量试验 2 小时血糖
    "BloodPressure",            # 舒张压 (mm Hg)
    "SkinThickness",            # 三头肌皮褶厚度 (mm)
    "Insulin",                  # 2 小时血清胰岛素 (mu U/ml)
    "BMI",                      # 体重指数
    "DiabetesPedigreeFunction", # 糖尿病家族遗传函数
    "Age",                      # 年龄
    "Outcome",                  # 结果 (1=患病, 0=未患病)
]

SRC = "data/pima_diabetes.csv"
DST = "data/diabetes.csv"

with open(SRC, "r", encoding="utf-8") as f_in, \
     open(DST, "w", encoding="utf-8", newline="") as f_out:
    writer = csv.writer(f_out)
    writer.writerow(COLUMNS)
    count = 0
    for line in f_in:
        line = line.strip()
        if not line:
            continue
        writer.writerow(line.split(","))
        count += 1

print(f"[OK] 已生成 {DST}")
print(f"[OK] 数据行数: {count} 条（不含表头）")
print(f"[OK] 列名: {', '.join(COLUMNS)}")
