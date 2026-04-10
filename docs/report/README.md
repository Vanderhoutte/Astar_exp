# 实验报告配图与数据说明

对应课程文档《实验一：AStar 算法求解 TSP 问题实验》中的**表 1–3**、**地图图片**与报告插图要求。

## 1. 自动生成（推荐）

在项目根目录执行：

```bash
pip install -r requirements-dev.txt
# 完整实验：n=10,20,50,100，重复次数见 config.report.json，并保存样例地图
python scripts/generate_report_assets.py

# 快速冒烟（小规模、少重复、不保存样例地图）
python scripts/generate_report_assets.py --fast
```

生成内容（默认路径）：

| 内容 | 路径 |
|------|------|
| 表 1–3 CSV + 终端汇总 | `data/report/tables/table1_euclidean.csv` 等、`run_summary.txt` |
| 样例地图 PNG（完整模式） | `data/report/instances/sample_n*_k*_s*.png` |
| 三种启发对比图 | `data/report/figures/heuristic_compare.png` |
| 同一实例三种启发解图 | `data/report/figures/same_instance_three_heuristics.png` |

说明：根目录 `.gitignore` 已忽略整个 `data/`，生成物仅在本地用于写报告；若需归档到仓库，请复制到 `docs/report/` 或取消忽略策略。

### Word 表头「平均运行时间」

CSV 中列名为 `平均运行时间_s`（单位秒）。脚本在生成对比图时会额外写出 `*_for_word.csv`，其中增加一列 `平均运行时间`，数值与 `平均运行时间_s` 相同，便于粘贴到 Word 表格。

## 2. 流程图与 GUI 思维导图 PNG

源文件：

- [`../astar_tsp_flowchart.drawio`](../astar_tsp_flowchart.drawio) — A* 求解 TSP 流程图  
- [`../gui_mindmap.drawio`](../gui_mindmap.drawio) — GUI 设计思维导图  

导出方式（任选其一）：

1. **draw.io 桌面版**：打开 `.drawio` → 文件 → 导出为 → PNG，保存到本目录，例如 `astar_tsp_flowchart.png`、`gui_mindmap.png`。  
2. **命令行**（若已安装 [draw.io CLI](https://github.com/jgraph/drawio-desktop) 且 `drawio` 在 `PATH` 中）：运行 `scripts/generate_report_assets.py` 时会尝试自动导出到本目录。

## 3. 课程实验说明 Word 文档

若需将实验要求一并纳入仓库，可将 `实验一_AStar算法求解TSP问题实验.docx` **自行复制**到 `docs/` 下（注意版权与课程规定）。本仓库默认不附带二进制 docx；正文要求也可通过解压 docx 内 `word/document.xml` 查看。

## 4. 与 `main.py tables` 的关系

`generate_report_assets.py` 内部调用与 `python main.py tables` 相同的 `cmd_tables` 逻辑；报告专用覆盖项见根目录 [`config.report.json`](../../config.report.json)（输出目录指向 `data/report/...`）。
