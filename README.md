# A* 求解 TSP 实验

在 **k 近邻稀疏图**（欧氏平面随机点 + 每点连接 `k` 个最近邻，对称无向）上用 **A\*** 求 **闭合旅行商回路**：从城市 0 出发，沿边访问每个顶点恰一次后回到 0。

- **三种启发**对比、**表 1–3** 批量 CSV + 终端/`run_summary.txt` 汇总  
- **Tk + Matplotlib GUI**：过程展示（满足课程 GUI 要求）  
- **配置**：根目录 [`config.json`](config.json)，可复制 [`config.example.json`](config.example.json)  
- **`demo`**：将步进事件与结果写入 `data/intermediate/`（TSV、摘要、PNG）

实验说明文档见 [`docs/实验一_AStar算法求解TSP问题实验.docx`](docs/实验一_AStar算法求解TSP问题实验.docx)。

## 背景：TSP 与 A\*

### 旅行商问题（TSP）

**经典表述**：给定若干城市及任意两城之间的距离（或代价），求一条**访问每个城市恰好一次**并最终**回到出发点**的闭合路线，使**总路程最小**。

**与本仓库的对应关系**：城市是欧氏平面上的点；**合法移动**仅限于事先生成的 **k 近邻稀疏图**上的边（对称无向），而非在完全图上任意跳。起点固定为城市 **0**，目标是该图上的一条**闭合哈密顿回路**。若图中不存在这样的回路，搜索会以失败结束（见下文「为何有时无法完成求解」）。

### A* 与本项目中的状态

**A\***：维护从起点到当前状态的真实代价 **g**，并用启发函数 **h** 估计剩余代价；按 **f = g + h** 从优先队列（开放表）中取出待扩展状态。当 **h 可采纳**（对任意状态不高估到达目标的真实最优剩余代价）时，首次取出的目标状态对应**最优解**。本实验的 **h** 基于「未访问点集上的 MST 松弛」等构造，与 **k-NN 图**上的真实行走模型未必一致，报告里需讨论可采纳性与扩展次数（见「三种启发函数」与「实验报告提示」）。

**状态表示**（[`src/astar_tsp_solver.py`](src/astar_tsp_solver.py)）：**位掩码 `mask`** 记录已访问过的顶点集合，**`current`** 为当前所在城市；**`phase`** 区分「仍在按边扩展、访问尚未访问的城」与「已访问全集、经边回到起点」的终止阶段，便于正确识别目标。

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

三者**结构相同**（见 [`src/heuristics.py`](src/heuristics.py) 中 `_h_from_metric`）：**尚未访问完所有城市**时，在 **「未访问 ∪ {当前城市}」** 上取 **Prim 型 MST**（在**完全图**上、边权用所选度量）；**掩码已满、只剩闭合回起点**时，启发仅为 **当前 → 起点** 的同度量距离（不再套 MST）。差别仅在于 **点对距离用哪种度量**。

| 名称 | 度量 | 与 k-NN **图边权**（欧氏）的关系 |
|------|------|----------------------------------|
| **euclidean** | 欧氏 L2 | 启发里用的就是 L2，与边权定义一致，常作基线 |
| **manhattan** | 曼哈顿 L1 | 启发在 **L1 意义下** 估 MST / 末段回边；真实走路径是 **稀疏图上的欧氏边**，二者**不保证一致**，可采纳性需在报告中讨论 |
| **custom** | **切比雪夫 L∞**（两坐标分量差的绝对值中取较大者） | 平面上点对满足 **d∞ ≤ d₂**；因而对**同一顶点子集**，L∞ 完全图 MST 的总权 **≤** 欧氏 MST（L∞ 的 MST 最优值不超过「沿用 L2 的 MST 那棵树」在 L∞ 下的权）。故在本实现下 **h_custom ≤ h_euclidean（逐状态）**，扩展数往往更少，但 **f 更紧不等于**对 **k-NN 图上真实剩余最优代价**可采纳——启发仍是 **完全图直线度量下的 MST 松弛**，与只能在 **给定边集** 上行走不是同一模型；报告里应写清 **图约束** 与下界关系，并与欧氏、曼哈顿对比 **扩展次数、解质量** |

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

- **custom**：写明实现为 **L∞ + 与本项目一致的 MST 松弛**（课程「自定义距离」可对切比雪夫度量）；可写 **d∞≤d₂ ⇒ 同子集 MST 权 MST∞≤MST₂ ⇒ h_custom≤h_eucl**。**不要**偷换成「故对 k-NN 图上的真实剩余最优代价一定可采纳」——应写清：**图上是欧氏 k-NN**，启发是 **完全图直线度量下的 MST**，二者模型不同。  
- **manhattan**：同上，L1 与欧氏边权的关系、扩展数与解对比。  
- **n=50/100** 写清时限与扩展策略。  
- 素材：**GUI**、`demo`（PNG、`events.tsv`）、`tables`（CSV、`run_summary.txt`）。

## 故障排除：Matplotlib Axes3D 警告

混装系统包与 pip 的 Matplotlib 时可能出现 **`Unable to import Axes3D`**。本实验仅用 2D 绘图，一般可忽略；[`src/mpl_compat.py`](src/mpl_compat.py) 会过滤该警告。根治：卸系统 `python3-matplotlib` 或单独 **venv** 安装依赖。

## 许可证

课程作业用途，按学校要求使用。
