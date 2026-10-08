"""
第 2 步：数据分析
对清洗后的数据做统计分析，找出与患病相关的关键指标。

运行：python 02_analyze.py
输出：控制台分析报告
"""
import pandas as pd

CLEAN = "data/diabetes_clean.csv"


def main():
    df = pd.read_csv(CLEAN)

    print("=" * 55)
    print("【一、总体患病情况】")
    total = len(df)
    sick = (df["Outcome"] == 1).sum()
    print(f"  总样本：{total} 人")
    print(f"  患病：{sick} 人（{sick / total * 100:.1f}%）")
    print(f"  未患病：{total - sick} 人（{(total - sick) / total * 100:.1f}%）")

    print("\n" + "=" * 55)
    print("【二、患病组 vs 未患病组 关键指标均值对比】")
    features = ["Glucose", "BloodPressure", "SkinThickness",
                "Insulin", "BMI", "Age", "DiabetesPedigreeFunction"]
    grouped = df.groupby("Outcome")[features].mean()
    print(f"  {'指标':<28}{'未患病':>10}{'患病':>10}{'差值':>10}")
    print("  " + "-" * 58)
    for col in features:
        v0 = grouped.loc[0, col]
        v1 = grouped.loc[1, col]
        print(f"  {col:<28}{v0:>10.2f}{v1:>10.2f}{v1 - v0:>+10.2f}")

    print("\n" + "=" * 55)
    print("【三、年龄分组患病率】")
    bins = [20, 30, 40, 50, 100]
    labels = ["21-30岁", "31-40岁", "41-50岁", "51岁以上"]
    df["年龄组"] = pd.cut(df["Age"], bins=bins, labels=labels, right=False)
    age_stat = df.groupby("年龄组", observed=True)["Outcome"].agg(["count", "sum", "mean"])
    for age_range, row in age_stat.iterrows():
        print(f"  {age_range:<12} 共 {int(row['count']):>3} 人，"
              f"患病 {int(row['sum']):>3} 人，患病率 {row['mean'] * 100:5.1f}%")

    print("\n" + "=" * 55)
    print("【四、各指标与患病的相关性】")
    corr = df[features + ["Outcome"]].corr()["Outcome"].drop("Outcome")
    corr_sorted = corr.sort_values(ascending=False)
    for col, val in corr_sorted.items():
        strength = "强" if abs(val) > 0.3 else ("中" if abs(val) > 0.15 else "弱")
        print(f"  {col:<28} 相关系数 {val:+.3f}  ({strength}相关)")

    print("\n" + "=" * 55)
    print("【五、核心结论】")
    top = corr_sorted.index[0]
    print(f"  1. 与患病相关性最强的指标是「{top}」")
    print(f"  2. 患病组血糖均值 {grouped.loc[1, 'Glucose']:.1f}，"
          f"明显高于未患病组 {grouped.loc[0, 'Glucose']:.1f}")
    print(f"  3. 年龄越大患病率越高（见第三部分）")
    print(f"  4. 注意：本数据集仅限皮马印第安女性，结论不可外推到全部人群")


if __name__ == "__main__":
    main()
