# 糖尿病数据集（Pima Indians Diabetes）说明

## 一、数据来源
- **原始出处**：UCI Machine Learning Repository — *Pima Indians Diabetes Database*
- **数据集编号**：UCI ID 34
- **官方页面**：https://archive.ics.uci.edu/dataset/34/diabetes
- **采集对象**：美国亚利桑那州皮马印第安人（Pima Indian）女性，21 岁以上
- **下载途径**：本次通过 jsDelivr CDN 镜像下载（原始 GitHub 源：jbrownlee/Datasets）
- **下载日期**：2026-10-08

> 注：以上来源信息为公开事实，UCI 数据集是数据科学领域最经典的教学数据集之一。

## 二、数据规模
- **样本数**：768 条（女性患者）
- **字段数**：9 列（8 个特征 + 1 个标签）
- **文件**：`data/diabetes.csv`（UTF-8，带表头）

## 三、字段说明

| 字段名 | 中文含义 | 单位 | 类型 |
|---|---|---|---|
| Pregnancies | 怀孕次数 | 次 | 整数 |
| Glucose | 口服葡萄糖耐量试验 2 小时血糖 | mg/dL | 整数 |
| BloodPressure | 舒张压 | mm Hg | 整数 |
| SkinThickness | 三头肌皮褶厚度 | mm | 整数 |
| Insulin | 2 小时血清胰岛素 | mu U/ml | 整数 |
| BMI | 体重指数 | kg/m² | 小数 |
| DiabetesPedigreeFunction | 糖尿病家族遗传函数（数值越大遗传风险越高） | — | 小数 |
| Age | 年龄 | 岁 | 整数 |
| **Outcome** | **是否患病：1=患病，0=未患病** | — | 0/1 |

## 四、已知数据问题（做分析时必须处理，否则结论不可靠）

1. **缺失值伪装成 0**
   以下字段存在大量 0 值，而生理上不可能为 0，实际是缺失数据：
   - `Glucose`、`BloodPressure`、`SkinThickness`、`Insulin`、`BMI`
   - **处理建议**：将 0 视为缺失，用中位数/均值填充，或删除该行

2. **样本不平衡**
   Outcome=1（患病）约 268 条，Outcome=0（未患病）约 500 条，比例约 1:1.87
   - **影响**：做预测模型时需注意类别不平衡问题

3. **样本代表性有限**
   仅限皮马印第安女性，不能直接外推到其他人群
   - **写分析报告时必须声明此局限**（这是加分项，体现严谨）

## 五、这个数据集能做什么分析（简历可写方向）
- 患病率与各指标的关联分析（如 BMI、血糖与患病的关系）
- 不同年龄段患病率分布
- 数据清洗与缺失值处理的完整流程演示
- 可视化看板（Streamlit）
- （进阶）用逻辑回归预测患病风险，评估模型准确率

## 六、文件清单
```
diabetes_project/
├── data/
│   ├── pima_diabetes.csv   # 原始下载文件（无表头）
│   └── diabetes.csv        # 加工后文件（带表头，可直接用）
├── prepare_data.py         # 加表头的脚本
└── DATA_README.md          # 本说明文件
```
