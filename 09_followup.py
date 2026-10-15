"""
第 9 步：慢病随访队列与提醒

前面 8 步做的是"分析"——把数据里的规律找出来。
这一步做的是"落地"——把规律变成一份**可执行的随访计划**。

真实医院场景：
  糖尿病是慢病，医院有"慢病管理中心"。筛查出高危人群后，
  不是做完就完了，而是要排随访：谁下个月来、谁三个月后复查、
  谁已经逾期没来了要打电话催。

这份脚本做的事：
  1. 风险评分   —— 用第 7 步拟合出的多因素逻辑回归，给每个患者打分
  2. 风险分层   —— 按分数分成 高危 / 中危 / 低危 三档
  3. 随访排期   —— 每档定不同的复查间隔，算出下次随访日期
  4. 逾期预警   —— 标出已经过了日期还没来的，按逾期天数排序

⚠️ 两点必须说清楚：
  - 随访间隔（1/3/6 个月）是**本项目的演示设定**，
    真实临床应遵循诊疗指南与医生判断，不能照抄。
  - 原始数据里**没有建档日期**这一列，
    脚本按固定随机种子补了一列用于演示，真实系统应直接取库里的日期。

运行：python 09_followup.py
输出：data/followup_plan.csv + docs/followup_report.md + 控制台摘要
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime

try:
    from scipy import stats
except ImportError:
    raise SystemExit(
        "缺少 scipy，请先安装：\n"
        "  pip install scipy -i https://mirrors.aliyun.com/pypi/simple/"
    )

try:
    import statsmodels.api as sm
    HAS_SM = True
except ImportError:
    HAS_SM = False

CLEAN = "data/diabetes_clean.csv"
OUT_DIR = "docs"
OUT_CSV = "data/followup_plan.csv"
OUT_MD = os.path.join(OUT_DIR, "followup_report.md")

FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
CN = {
    "Pregnancies": "妊娠次数", "Glucose": "血糖", "BloodPressure": "舒张压",
    "SkinThickness": "皮褶厚度", "Insulin": "血清胰岛素", "BMI": "体质指数",
    "DiabetesPedigreeFunction": "家族史评分", "Age": "年龄",
}
TARGET = "Outcome"
SEED = 20261014          # 与 seed_demo_data.py 保持一致的固定种子
TODAY = datetime.now()

# 各风险档的随访间隔（单位：天）——演示设定，非临床指南
INTERVALS = {"高危": 30, "中危": 90, "低危": 180}
# 分层阈值：预测概率
THRESHOLDS = {"高危": 0.60, "中危": 0.30}


def risk_level(p):
    if p >= THRESHOLDS["高危"]:
        return "高危"
    if p >= THRESHOLDS["中危"]:
        return "中危"
    return "低危"


def main():
    if not os.path.exists(CLEAN):
        print(f"找不到 {CLEAN}，请先运行 01_clean.py")
        return

    df = pd.read_csv(CLEAN)
    n = len(df)

    # ---------- 1. 风险评分 ----------
    # 优先用多因素逻辑回归的预测概率；
    # 若 statsmodels 不可用，退化成"血糖 + BMI 分档"的简化规则，并在报告里注明。
    if HAS_SM:
        X = df[FEATURES].astype(float)
        Xs = (X - X.mean()) / X.std(ddof=1)
        Xs = sm.add_constant(Xs)
        try:
            res = sm.Logit(df[TARGET].astype(float), Xs).fit(disp=0)
            score = res.predict(Xs)
            method = "多因素逻辑回归预测概率"
        except Exception:
            score, method = None, None
    else:
        score, method = None, None

    if score is None:
        g = df["Glucose"].astype(float)
        b = df["BMI"].astype(float)
        score = ((g - g.mean()) / g.std(ddof=1)
                 + (b - b.mean()) / b.std(ddof=1)) / 2
        # 归一化到 0~1
        score = 1 / (1 + np.exp(-score))
        method = "简化规则（statsmodels 不可用）"

    df = df.assign(risk_score=score.round(4))
    df["risk_level"] = df["risk_score"].apply(risk_level)
    df["followup_days"] = df["risk_level"].map(INTERVALS)

    # ---------- 2. 上次/下次随访日期 ----------
    rng = np.random.default_rng(SEED)
    offsets = rng.integers(0, 366, size=n)
    enroll = pd.Timestamp(TODAY.date()) - pd.to_timedelta(offsets, unit="D")
    df["last_followup"] = enroll
    df["next_followup"] = enroll + pd.to_timedelta(df["followup_days"], unit="D")
    df["overdue_days"] = ((pd.Timestamp(TODAY.date()) - df["next_followup"])
                          .dt.days.clip(lower=0))
    df["is_overdue"] = df["overdue_days"] > 0

    # ---------- 3. 输出 CSV ----------
    os.makedirs(OUT_DIR, exist_ok=True)
    cols = ["risk_level", "risk_score", "Glucose", "BMI", "Age",
            "Pregnancies", "Outcome", "last_followup",
            "next_followup", "overdue_days", "is_overdue"]
    df[cols].to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    # ---------- 4. 汇总 ----------
    order = ["高危", "中危", "低危"]
    counts = df["risk_level"].value_counts().reindex(order).fillna(0).astype(int)
    pos_rate = (df.groupby("risk_level")[TARGET].mean()
                .reindex(order).round(3))

    n_over = int(df["is_overdue"].sum())
    max_over = int(df["overdue_days"].max())

    L = []
    L.append("# 慢病随访队列与提醒")
    L.append("")
    L.append(f"生成时间：{TODAY.strftime('%Y-%m-%d %H:%M')}　｜　"
             f"样本：{n} 人　｜　风险评分方法：{method}")
    L.append("")
    L.append("> ⚠️ 随访间隔（30/90/180 天）为**演示设定**，"
             "真实临床应遵循诊疗指南与医生判断。")
    L.append("")
    L.append("## 一、风险分层结果")
    L.append("")
    L.append("| 风险档 | 人数 | 占比 | 该档实际确诊率 |")
    L.append("|---|---|---|---|")
    for lv in order:
        c = int(counts[lv])
        L.append(f"| {lv} | {c} | {c / n * 100:.1f}% | "
                   f"{pos_rate[lv] * 100:.1f}% |")
    L.append("")
    L.append("## 二、随访排期")
    L.append("")
    L.append("| 风险档 | 复查间隔 |")
    L.append("|---|---|")
    for lv in order:
        L.append(f"| {lv} | 每 {INTERVALS[lv]} 天 |")
    L.append("")
    L.append("## 三、逾期预警")
    L.append("")
    L.append(f"- 当前**已逾期**未随访：**{n_over} 人**（占 {n_over / n * 100:.1f}%）")
    L.append(f"- 最长逾期：**{max_over} 天**")
    L.append("")
    L.append("### 优先联系名单（逾期最久前 10 位）")
    L.append("")
    L.append("| 患者编号 | 风险档 | 上次随访 | 应随访日 | 逾期天数 |")
    L.append("|---|---|---|---|---|")
    top = (df.sort_values("overdue_days", ascending=False)
              .head(10))
    for i, (_, r) in enumerate(top.iterrows(), 1):
        L.append(f"| P{i:03d} | {r['risk_level']} | "
                   f"{r['last_followup'].date()} | "
                   f"{r['next_followup'].date()} | "
                   f"{int(r['overdue_days'])} 天 |")
    L.append("")
    L.append("---")
    L.append("")
    L.append("**说明**")
    L.append("")
    L.append("- 风险评分来自第 7 步拟合的多因素逻辑回归模型，"
             "取值 0~1，越接近 1 表示模型判断的患病可能性越高。")
    L.append("- 该评分是**统计模型的输出，不是临床诊断**；"
             "真实筛查应以医生判断和实验室检查为准。")
    L.append("- 原始数据不含建档日期，脚本以固定随机种子补入用于演示排期逻辑。")
    L.append("- 随访间隔为本项目的演示设定，"
             "实际应参照糖尿病诊疗指南与个体情况确定。")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    # ---------- 控制台摘要 ----------
    print("=" * 60)
    print("【慢病随访队列】")
    print("=" * 60)
    print(f"样本 {n} 人　风险评分方法：{method}")
    print()
    print("风险分层：")
    for lv in order:
        c = int(counts[lv])
        bar = "#" * int(c / n * 40)
        print(f"  {lv}  {c:>4} 人（{c / n * 100:>5.1f}%）{bar}")
    print()
    print("随访间隔：")
    for lv in order:
        print(f"  高危 每 {INTERVALS[lv]} 天")
        break
    for lv in order[1:]:
        print(f"  {lv} 每 {INTERVALS[lv]} 天")
    print()
    print(f"逾期未随访：{n_over} 人，最长逾期 {max_over} 天")
    print()
    print(f"计划已输出：{OUT_CSV}")
    print(f"报告已输出：{OUT_MD}")


if __name__ == "__main__":
    main()
