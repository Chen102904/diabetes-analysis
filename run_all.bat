@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================
echo   糖尿病数据分析项目 - 一键复现
echo ============================================
echo.

echo [1/4] 数据预处理（加表头）...
python prepare_data.py
if errorlevel 1 goto error

echo.
echo [2/4] 数据清洗...
python 01_clean.py
if errorlevel 1 goto error

echo.
echo [3/4] 数据统计分析...
python 02_analyze.py
if errorlevel 1 goto error

echo.
echo [4/4] 生成可视化图表...
python 03_visualize.py
if errorlevel 1 goto error

echo.
echo ============================================
echo   全部完成！图表已生成到 images 文件夹
echo   要看交互看板，请单独运行：
echo   streamlit run 04_app.py
echo ============================================
pause
exit /b 0

:error
echo.
echo ============================================
echo   [错误] 执行失败，请检查：
echo   1. 是否已安装 Python 并加入 PATH
echo   2. 是否已安装依赖：pip install -r requirements.txt
echo ============================================
pause
exit /b 1
