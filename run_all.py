"""run_all.py —— 一键复跑：从清洗好的数据重新生成所有图表和报告

这是「可复现」的入口。面试时说"改一个参数，一条命令重跑全部分析"，
指的就是这个脚本。它按顺序调用 c4→c7，串起完整分析流水线。

用法：
    python run_all.py

依赖前置：data/china_analysis.csv 已存在（由 c3_clean.py 生成）。
如果要连清洗一起重跑：先 python c3_clean.py，再 python run_all.py。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

# 顺序执行的分析步骤（每个脚本自包含，产出自己的图/报告）
STEPS = [
    ("c4_describe.py",    "第 4 步：描述统计（发病率图 + 年龄分层图）"),
    ("c5_cox.py",         "第 5 步：Cox 回归（M1→M4 + 比例风险检验）"),
    ("c6_effect_mod.py",  "第 6 步：效应修饰（年龄分层 HR + 交互检验）"),
    ("c7_sensitivity.py", "第 7 步：敏感性分析（6 条）"),
]


def main() -> None:
    print("=" * 60)
    print("中国体检队列分析 · 一键复跑")
    print("=" * 60)

    # 前置检查：清洗后的数据在不在
    data = ROOT / "data" / "china_analysis.csv"
    if not data.exists():
        print(f"\n[前置缺失] 找不到 {data.name}")
        print("请先运行：python c3_clean.py 生成清洗后的数据。")
        sys.exit(1)

    print(f"\n数据：{data.name}（{data.stat().st_size / 1e6:.1f} MB）\n")

    failed = []
    for script, desc in STEPS:
        print(f"\n{'─' * 60}")
        print(f"[运行] {script} —— {desc}")
        print("─" * 60)
        r = subprocess.run([PY, str(ROOT / script)], cwd=str(ROOT))
        if r.returncode != 0:
            failed.append(script)
            print(f"[失败] {script} 返回码 {r.returncode}")

    print(f"\n{'=' * 60}")
    if failed:
        print(f"[完成] 有 {len(failed)} 步失败：{', '.join(failed)}")
        sys.exit(1)
    print("[完成] 全部 4 步跑通。")
    print("产出：")
    print("  docs/china_表1.md（基线特征）")
    print("  docs/china_表2_Cox回归.md")
    print("  docs/china_表3_效应修饰.md")
    print("  docs/china_表4_敏感性分析.md")
    print("  images/china_fig1_发病率.png")
    print("  images/china_fig2_年龄分层.png")
    print("  images/china_fig3_年龄分层HR.png")
    print()
    print("下一步：streamlit run 04_app.py 打开看板")
    print("=" * 60)


if __name__ == "__main__":
    main()
