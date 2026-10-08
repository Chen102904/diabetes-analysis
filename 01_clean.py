"""
第 1 步：数据清洗
把原始数据里的"假 0"识别为缺失值，做填充处理，输出干净数据。

运行：python 01_clean.py
输出：data/diabetes_clean.csv + 控制台清洗报告
"""
import pandas as pd

RAW = "data/diabetes.csv"
OUT = "data/diabetes_clean.csv"

# 这 5 列里，0 在生理上不可能出现，实际代表"缺失"
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def main():
    df = pd.read_csv(RAW)

    print("=" * 50)
    print("【原始数据概况】")
    print(f"样本数：{len(df)} 条，字段数：{df.shape[1]} 个")
    print(f"患病(1)：{(df['Outcome'] == 1).sum()} 条  |  "
          f"未患病(0)：{(df['Outcome'] == 0).sum()} 条")

    print("\n" + "=" * 50)
    print("【缺失值检查（0 值统计）】")
    missing_report = {}
    for col in ZERO_AS_MISSING:
        n = (df[col] == 0).sum()
        pct = n / len(df) * 100
        missing_report[col] = n
        print(f"  {col:<28} 0 值 {n:>4} 个，占 {pct:5.1f}%")

    # 把 0 替换为 NaN（真正的缺失值）
    df[ZERO_AS_MISSING] = df[ZERO_AS_MISSING].replace(0, pd.NA)

    print("\n" + "=" * 50)
    print("【清洗策略】")
    print("  将 0 视为缺失后，用各列【中位数】填充")
    print("  原因：中位数不受极端值影响，比均值更稳健")

    for col in ZERO_AS_MISSING:
        median = df[col].median()
        df[col] = df[col].fillna(median)
        print(f"  {col:<28} 用中位数 {median:.2f} 填充")

    # 保存
    df.to_csv(OUT, index=False)

    print("\n" + "=" * 50)
    print("【清洗完成】")
    print(f"  输出文件：{OUT}")
    print(f"  剩余缺失值：{df.isna().sum().sum()} 个")
    print(f"  最终样本数：{len(df)} 条")


if __name__ == "__main__":
    main()
