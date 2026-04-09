# A* 求解 TSP 实验

在 **k 近邻稀疏图**（欧氏平面随机点 + 每点连接 `k` 个最近邻，对称无向）上用 **A\*** 求 **闭合旅行商回路**：从城市 0 出发，沿边访问每个顶点恰一次后回到 0。

- **三种启发**对比、**表 1–3** 批量 CSV + 终端/`run_summary.txt` 汇总  
- **Tk + Matplotlib GUI**：过程展示（满足课程 GUI 要求）  
- **配置**：根目录 [`config.json`](config.json)，可复制 [`config.example.json`](config.example.json)  
- **`demo`**：将步进事件与结果写入 `data/intermediate/`（TSV、摘要、PNG）

实验说明文档见 [`docs/实验一_AStar算法求解TSP问题实验.docx`](docs/实验一_AStar算法求解TSP问题实验.docx)。

## 环境

- Python 3.10+（建议）  
- 依赖：`pip install -r requirements.txt`  
- **GUI** 需要 **tkinter**（多数系统随 Python 提供）

## 安装

```bash
cd Astar_exp
pip install -r requirements.txt
```

## 项目结构

| 路径 | 说明 |
|------|------|
| `config.json` | 运行参数（缺省键见 `main.py` 中 `_DEFAULT`） |
| `main.py` | 子命令：`tables` / `demo` / `gui`；`--config` 指定 JSON |
| `src/map_generator.py` | k-NN 随机图、连通修补、实例 CSV 读写 |
| `src/heuristics.py` | 三种启发（见下节） |
| `src/astar_tsp_solver.py` | `AStarTSPSolver`、`search` / `search_stepwise` |
| `src/visualization.py` | 静态 PNG；`launch_gui`（Tk 演示） |
| `src/mpl_compat.py` | 过滤混装 Matplotlib 的常见警告 |
| `scripts/run_experiments.py` | 等价于 `python main.py tables …` |
| `scripts/launch_gui.py` | 等价于 `python main.py gui`（支持 `--config`，见下） |
| `data/results/` | `table*.csv`、`run_summary.txt` |
| `data/instances/` | 可选：`tables --save-sample-maps` 的示例地图 |
| `data/intermediate/` | `demo`：`events.tsv`、`summary.txt`、`map.png`、`solution.png` |

## 三种启发函数（`euclidean` / `manhattan` / `custom`）

三者**结构相同**：在 **「未访问城市 ∪ {当前城市}」** 上，用 **Prim 型 MST**（在**完全图**上、边权由所选度量给出），再加上 **从当前城市回到起点** 的一程（仍用同一度量）。差别仅在于 **点对距离用哪种度量**。

| 名称 | 度量 | 与 k-NN **图边权**（欧氏）的关系 |
|------|------|----------------------------------|
| **euclidean** | 欧氏 L2 | 启发里用的就是 L2，与边权定义一致，常作基线 |
| **manhattan** | 曼哈顿 L1 | 启发在 **L1 意义下** 估 MST + 回边；真实走路径是 **稀疏图上的欧氏边**，二者**不保证一致**，可采纳性需在报告中讨论 |
| **custom** | **切比雪夫 L∞**（`max(|Δx|,|Δy|)`） | 平面上**任意两点**满足 **d∞ ≤ d₂**（逐边比较成立），但启发仍是 **L∞ 下的 MST + L∞ 回边**，而实际代价沿 **k-NN 欧氏边** 走；**L∞ 的最小生成树与欧氏 MST 可以不是同一棵树**，因此**不能**由「d∞≤d₂」推出对真实剩余欧氏代价一定可采纳或一定更小，仅适合与欧氏、曼哈顿一起做**扩展次数、解质量**的实验对比并在报告中说明 |

对应结果文件：`table1_euclidean.csv`、`table2_manhattan.csv`、`table3_custom.csv`。

## 命令行与 `--config`

使用 **argparse** 时，**`--config` 必须写在子命令之前**，例如：

```bash
python main.py --config config.json tables
python main.py --config config.json demo
python main.py --config config.json gui
```

仅写 `python main.py tables` 则默认读取项目根下 `config.json`（不存在则使用 `main.py` 内置默认）。

`scripts/launch_gui.py` 若带 `--config FILE`，会转换为 `main.py --config FILE gui`。

## 快速运行

### 1. 配置

```bash
cp config.example.json config.json   # 首次可复制模板后编辑
```

常用键：`sizes`、`k`、`repeats`、`seed`、`max_expansions`、`time_limit`、`save_sample_maps`、`auto_time_limit_n_ge_50`；`demo_*`、`intermediate_dir`；`gui_default_*`（GUI 初值）。

### 2. 批量实验（表 1–3）

```bash
python main.py tables
python main.py --config my.json tables
python main.py tables --sizes 10 20 50 100 --k 4 --repeats 10 --seed 42
python main.py tables --save-sample-maps
```

亦可用：`python scripts/run_experiments.py …`（参数传给 `main.py tables`）。

- 生成：`data/results/table1_euclidean.csv` 等。  
- **终端**打印与实验表字段一致的汇总；并写入 **`data/results/run_summary.txt`**（UTF-8）。  
- `max_expansions` / `time_limit` 为 JSON **`null`** 表示不限制。  
- 若 `auto_time_limit_n_ge_50` 为 **true**（默认），且未设上述两项、且 `sizes` 中含 **n≥50**，则每次运行自动 **120s** 时限（stderr 有提示）。

### 3. 图形界面（GUI）

```bash
python main.py gui
python main.py --config my.json gui
python scripts/launch_gui.py
python scripts/launch_gui.py --config my.json
```

功能简述：随机生成 / 加载 CSV；三种启发；后台一次跑完或单步、自动步进；日志含 **f, g, h、扩展次数**；**单步「本步搜索耗时 / 累计搜索耗时」仅统计 `next(生成器)` 内 CPU 时间，不含你在界面上的挂机等待**；结束时有与实验字段对齐的**单次汇总**（单次成功时最优=最差=平均路径长度）。

**多样本、多规模**的最优/最差/平均等仍以 **`main.py tables`** 的 CSV 为准。

### 4. 单次演示（`demo`）

```bash
python main.py demo
python main.py --config my.json demo
```

输出目录由 `intermediate_dir` 决定，默认 `data/intermediate/`：

- `events.tsv`（`save_events`）  
- `summary.txt`（与实验字段对齐，**同时打印到终端**）  
- `map.png`、`solution.png`（成功且 `save_png`）  
- `demo_save_instance` 为 true 时另有 `demo_points.csv` / `demo_edges.csv`

### 5. 在代码中调用

```python
from src.map_generator import ensure_connected_fallback, generate_random_map
from src.astar_tsp_solver import AStarTSPSolver
import numpy as np

g = generate_random_map(12, k_neighbors=4, seed=0)
g = ensure_connected_fallback(g, np.random.default_rng(0))
solver = AStarTSPSolver(g, heuristic="euclidean", start=0)
res = solver.search(max_expansions=500_000, time_limit_sec=60.0)
print(res.success, res.cost, res.expansions)
```

## 为何有时无法完成求解？

1. **图上无哈密顿回路**：算法只在 **当前 k-NN 图** 的边上找回路；k 小或点分布不利时，**可能不存在**合法闭合回路。连通修补 **不保证** 存在回路。失败信息多为 **`no solution (open empty)`**。可尝试 **增大 k（3～5）**、换 **seed** 再生成。  
2. **扩展或时限截断**：达到 **`max_expansions`** 或 **`time_limit_sec`** 会提前结束（`expansion limit` / `time limit`），大 n 时很常见。  
3. 报告需写明：是否使用上限、**失败次数**、表内统计是否**仅针对成功运行**。

## 实验报告提示

- **custom**：写明实现为 **L∞ + MST 型下界**（与课程「自定义距离」对应到切比雪夫度量即可）；**不要**写成「因 d∞≤欧氏故启发一定可采纳」——应写清：**图上是欧氏 k-NN**，启发在 **另一度量下的完全图 MST**，需讨论与真实剩余代价的关系。  
- **manhattan**：同上，L1 与欧氏边权的关系、扩展数与解对比。  
- **n=50/100** 写清时限与扩展策略。  
- 素材：**GUI**、`demo`（PNG、`events.tsv`）、`tables`（CSV、`run_summary.txt`）。

## 故障排除：Matplotlib Axes3D 警告

混装系统包与 pip 的 Matplotlib 时可能出现 **`Unable to import Axes3D`**。本实验仅用 2D 绘图，一般可忽略；[`src/mpl_compat.py`](src/mpl_compat.py) 会过滤该警告。根治：卸系统 `python3-matplotlib` 或单独 **venv** 安装依赖。

## 许可证

课程作业用途，按学校要求使用。
