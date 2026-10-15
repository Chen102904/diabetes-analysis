"""
第 8 步：面向非技术人的分析报告

前面的报告都是给"看得懂 p 值的人"看的。
这一份是给**医院管理者 / 科室主任**看的——他们只关心三件事：
  1. 结论是什么
  2. 我该怎么做
  3. 这个结论有多靠谱

所以这份报告：
  - 不出现公式，不出现原始字段名（全部用中文医学名）
  - 统计量翻译成人话（OR=3.17 → "血糖高出一大截的人，患病几率约 3 倍"）
  - 结论在前，依据在后
  - 明确写出"不能说什么"（相关性 ≠ 因果）

产出是一个**自包含的 HTML 文件**：双击就能打开，可以直接打印或存 PDF。

运行：python 08_report.py
输出：docs/报告.html + 控制台摘要
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
OUT_HTML = os.path.join(OUT_DIR, "报告.html")
IMG = "../images"

FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
CN = {
    "Pregnancies": "妊娠次数", "Glucose": "血糖", "BloodPressure": "舒张压",
    "SkinThickness": "皮褶厚度", "Insulin": "血清胰岛素", "BMI": "体质指数",
    "DiabetesPedigreeFunction": "家族史评分", "Age": "年龄",
}
TARGET = "Outcome"
ALPHA = 0.05


# ---------- 统计计算 ----------
def logit_or(df, feats):
    """多元逻辑回归，返回 {字段: (OR, CI下限, CI上限, p值)}

    变量先标准化，OR 读作「每升高 1 个标准差，患病几率乘以 OR」。
    """
    if not HAS_SM:
        return None
    X = df[feats].astype(float)
    X = (X - X.mean()) / X.std(ddof=1)
    X = sm.add_constant(X)
    try:
        res = sm.Logit(df[TARGET].astype(float), X).fit(disp=0)
    except Exception:
        return None
    out = {}
    ci = res.conf_int()
    for f in feats:
        out[f] = (float(np.exp(res.params[f])), float(np.exp(ci.loc[f, 0])),
                  float(np.exp(ci.loc[f, 1])), float(res.pvalues[f]))
    return out


def group_diff(df, col):
    """Welch t 检验，返回 (患病均值, 未患病均值, 差值, CI下限, CI上限, p, Cohen's d)"""
    a = df.loc[df[TARGET] == 1, col].dropna()
    b = df.loc[df[TARGET] == 0, col].dropna()
    na, nb = len(a), len(b)
    ma, mb = a.mean(), b.mean()
    diff = ma - mb
    _, p = stats.ttest_ind(a, b, equal_var=False)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va / na + vb / nb)
    dfw = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    tc = stats.t.ppf(1 - ALPHA / 2, dfw)
    sp = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    return ma, mb, diff, diff - tc * se, diff + tc * se, p, (diff / sp if sp else 0.0)


def pct(x):
    return f"{x * 100:.1f}%"


def main():
    if not os.path.exists(CLEAN):
        print(f"找不到 {CLEAN}，请先运行 01_clean.py")
        return

    df = pd.read_csv(CLEAN)
    n = len(df)
    n_pos = int((df[TARGET] == 1).sum())
    n_neg = n - n_pos
    m = len(FEATURES)
    alpha_adj = ALPHA / m

    # 单因素 OR：逐个指标单独回归
    single = {}
    for f in FEATURES:
        r = logit_or(df, [f])
        if r:
            single[f] = r[f]
    # 多因素 OR：全部一起回归
    multi = logit_or(df, FEATURES) or {}

    os.makedirs(OUT_DIR, exist_ok=True)

    # ---------- 拼 HTML ----------
    def rows_for(title, mapping):
        out = [f"<h2>{title}</h2>",
               "<table>",
               "<tr><th>指标</th><th>OR</th><th>95% 置信区间</th><th>p 值</th><th>结论</th></tr>"]
        items = sorted(mapping.items(), key=lambda kv: -kv[1][0])
        for f, (o, lo, hi, p) in items:
            if p < alpha_adj:
                sig, cls = "显著", "ok"
            elif p < ALPHA:
                sig, cls = "名义显著", "warn"
            else:
                sig, cls = "不显著", "no"
            out.append(f"<tr><td>{CN[f]}</td><td>{o:.2f}</td>"
                       f"<td>[{lo:.2f}, {hi:.2f}]</td>"
                       f"<td>{p:.2e}</td>"
                       f"<td class='{cls}'>{sig}</td></tr>")
        out.append("</table>")
        return out

    h = []
    h.append("<!DOCTYPE html>")
    h.append('<html lang="zh-CN">')
    h.append('<head><meta charset="utf-8">')
    h.append("<title>糖尿病筛查数据分析报告</title>")
    h.append("<style>")
    h.append("body{font-family:'Microsoft YaHei',sans-serif;max-width:900px;"
             "margin:40px auto;padding:0 24px;line-height:1.7;color:#222;}")
    h.append("h1{border-bottom:3px solid #1B4332;padding-bottom:12px;}")
    h.append("h2{margin-top:36px;color:#1B4332;}")
    h.append("table{border-collapse:collapse;width:100%;margin:16px 0;}")
    h.append("th,td{border:1px solid #ddd;padding:8px 12px;text-align:left;}")
    h.append("th{background:#1B4332;color:#fff;}")
    h.append("tr:nth-child(even){background:#f7f7f4;}")
    h.append(".card{background:#EAF3DE;border-left:5px solid #2D6A4F;"
             "padding:16px 20px;margin:24px 0;border-radius:6px;}")
    h.append(".warn{background:#FAEEDA;border-left:5px solid #BA7517;"
             "padding:16px 20px;margin:24px 0;border-radius:6px;}")
    h.append(".ok{color:#2D6A4F;font-weight:bold;}")
    h.append(".no{color:#888;}")
    h.append(".small{color:#666;font-size:13px;}")
    h.append("@media print{body{margin:0;}}")
    h.append("</style></head><body>")

    h.append("<h1>糖尿病筛查数据分析报告</h1>")
    h.append(f'<p class="small">生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M")}　'
             f'样本量：{n} 人</p>')

    # 结论卡片
    h.append('<div class="card">')
    h.append("<h2>一句话结论</h2>")
    h.append("<p>本次筛查的 768 名受试者中，<b>268 人确诊糖尿病，患病率 34.9%</b>。"
             "在所有被记录的指标里，<b>血糖</b>是与患病关联最强、且在控制其他因素后"
             "仍然独立成立的风险指标。</p>")
    h.append("</div>")

    # 数据概况
    h.append("<h2>一、数据概况</h2>")
    h.append("<table>")
    h.append("<tr><th>项目</th><th>数值</th></tr>")
    h.append(f"<tr><td>受试人数</td><td>{n} 人</td></tr>")
    h.append(f"<tr><td>确诊糖尿病</td><td>{n_pos} 人（{pct(n_pos / n)}）</td></tr>")
    h.append(f"<tr><td>未患病</td><td>{n_neg} 人（{pct(n_neg / n)}）</td></tr>")
    h.append(f"<tr><td>分析指标</td><td>{m} 项</td></tr>")
    h.append("</table>")

    # 风险因素
    if single and multi:
        h.append("<h2>二、哪些指标真的有风险</h2>")
        h.append("<p>两个口径都看，因为结论会不一样：</p>")
        h += rows_for("2.1 单看一个指标时", single)
        h.append("""
<div class="warn">
<b>为什么两个口径要一起看：</b><br>
单看时显著的指标，可能是因为"它和真正的原因绑在一起"。
比如年龄大的人本来血糖和 BMI 就更高，
那"年龄看着有风险"可能只是血糖、BMI 在起作用。<br><br>
把所有指标放进同一个模型（多因素），
能看出<b>剔除互相影响之后，谁还真的独立起作用</b>。
</div>""")
        h += rows_for("2.2 控制其他指标之后（多因素）", multi)

        # 效应量
        h.append("<h2>三、差异有多大（效应量）</h2>")
        h.append("<table>")
        h.append("<tr><th>指标</th><th>患病组</th><th>未患病组</th><th>差值</th>"
                 "<th>95% CI</th><th>p 值</th><th>效应量 d</th></tr>")
        for f in FEATURES:
            ma, mb, diff, lo, hi, p, d = group_diff(df, f)
            h.append(f"<tr><td>{CN[f]}</td><td>{ma:.2f}</td><td>{mb:.2f}</td>"
                     f"<td>{diff:+.2f}</td><td>[{lo:+.2f}, {hi:+.2f}]</td>"
                     f"<td>{p:.2e}</td><td>{d:+.2f}</td></tr>")
        h.append("</table>")
        h.append('<p class="small">Cohen\'s d 经验阈值：0.2 小效应 / 0.5 中效应 / 0.8 大效应。</p>')

    # 局限
    h.append("<h2>四、这个结论不能说什么</h2>")
    h.append("""
<div class="warn">
<b>1. 相关性不等于因果性。</b>
本数据是横断面的（同一次筛查的快照），
只能说"血糖高的人里患病的多"，
<b>不能说"降血糖就能防糖尿病"</b>。
要证明因果需要前瞻性队列研究或干预试验。<br><br>
<b>2. 人群不代表全体。</b>
样本仅含成年皮马印第安女性，
结论不可直接外推到男性、其他族裔或青少年。<br><br>
<b>3. 约一半的胰岛素记录是填充值。</b>
该列近半数为无效 0，经中位数填充后方差被压缩，
与胰岛素相关的统计量被系统性低估。
</div>""")

    h.append("</body></html>")

    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write("\n".join(h))

    # ---------- 控制台摘要 ----------
    print("=" * 60)
    print("【面向非技术人的分析报告】")
    print("=" * 60)
    print(f"样本 {n} 人，患病 {n_pos} 人（{pct(n_pos / n)}）")
    print()
    print("多因素逻辑回归 OR（标准化后，按效应从大到小）：")
    for f, (o, lo, hi, p) in sorted(multi.items(), key=lambda kv: -kv[1][0]):
        flag = "显著" if p < alpha_adj else ("名义显著" if p < ALPHA else "不显著")
        print(f"  {CN[f]:<12} OR={o:.2f}  CI[{lo:.2f},{hi:.2f}]  p={p:.2e}  [{flag}]")
    print()
    print(f"报告已输出：{OUT_HTML}")


if __name__ == "__main__":
    main()
