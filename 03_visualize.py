"""
第 3 步：数据可视化
生成 4 张图表，保存到 images/ 目录。

运行：python 03_visualize.py
输出：images/ 下 4 个 png 文件
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

CLEAN = "data/diabetes_clean.csv"
IMG_DIR = "images"

# 中文字体设置（Windows 用 SimHei）
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    df = pd.read_csv(CLEAN)

    # 图1：患病 vs 未患病 饼图
    fig, ax = plt.subplots(figsize=(6, 6))
    counts = df["Outcome"].value_counts()
    ax.pie(counts, labels=["未患病", "患病"], autopct="%1.1f%%",
           colors=["#4A90D9", "#D9534F"], startangle=90,
           textprops={"fontsize": 13})
    ax.set_title("样本患病比例（共 768 人）", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/1_患病比例.png", dpi=150)
    plt.close()
    print(f"[OK] {IMG_DIR}/1_患病比例.png")

    # 图2：患病组 vs 未患病组 血糖分布对比
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df[df["Outcome"] == 0]["Glucose"], bins=25, alpha=0.7,
            label="未患病", color="#4A90D9")
    ax.hist(df[df["Outcome"] == 1]["Glucose"], bins=25, alpha=0.7,
            label="患病", color="#D9534F")
    ax.set_xlabel("血糖 (mg/dL)", fontsize=12)
    ax.set_ylabel("人数", fontsize=12)
    ax.set_title("患病组与未患病组血糖分布对比", fontsize=14)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/2_血糖分布对比.png", dpi=150)
    plt.close()
    print(f"[OK] {IMG_DIR}/2_血糖分布对比.png")

    # 图3：年龄分组患病率柱状图
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = [20, 30, 40, 50, 100]
    labels = ["21-30岁", "31-40岁", "41-50岁", "51岁以上"]
    df["年龄组"] = pd.cut(df["Age"], bins=bins, labels=labels, right=False)
    rate = df.groupby("年龄组", observed=True)["Outcome"].mean() * 100
    bars = ax.bar(rate.index, rate.values, color="#5B9BD5", width=0.6)
    ax.set_ylabel("患病率 (%)", fontsize=12)
    ax.set_title("不同年龄段患病率", fontsize=14)
    for bar, v in zip(bars, rate.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1,
                f"{v:.1f}%", ha="center", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/3_年龄患病率.png", dpi=150)
    plt.close()
    print(f"[OK] {IMG_DIR}/3_年龄患病率.png")

    # 图4：各指标与患病相关性 横向条形图
    fig, ax = plt.subplots(figsize=(8, 5))
    features = ["Glucose", "BMI", "Age", "Pregnancies", "DiabetesPedigreeFunction",
                "Insulin", "SkinThickness", "BloodPressure"]
    corr = df[features + ["Outcome"]].corr()["Outcome"].drop("Outcome")
    corr = corr.sort_values()
    colors = ["#D9534F" if v > 0 else "#5B9BD5" for v in corr.values]
    ax.barh(corr.index, corr.values, color=colors)
    ax.set_xlabel("与患病的相关系数", fontsize=12)
    ax.set_title("各指标与糖尿病的相关性排序", fontsize=14)
    ax.axvline(0, color="gray", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/4_相关性排序.png", dpi=150)
    plt.close()
    print(f"[OK] {IMG_DIR}/4_相关性排序.png")

    print("\n全部图表生成完毕，共 4 张。")


if __name__ == "__main__":
    main()
