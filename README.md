# A* 求解 TSP 实验

在 **k 近邻稀疏图**（欧氏平面随机点 + 每点连接 `k` 个最近邻，对称无向）上用 **A\*** 求 **闭合旅行商回路**：从配置指定（或随机抽取）的起点出发，沿边访问每个顶点恰一次后回到该起点。

- **三种启发**对比、**表 1–3** 批量 CSV + 终端/`run_summary.txt` 汇总  
- **Tk + Matplotlib GUI**：过程展示（满足课程 GUI 要求）  
- **配置**：根目录 [`config.json`](config.json)，可复制 [`config.example.json`](config.example.json)  
- **`demo`**：将步进事件与结果写入 `data/intermediate/`（TSV、摘要、PNG）

实验说明文档（课程 Word）可自行复制到 `docs/`；报告配图与表 1–3 生成说明见 [`docs/report/README.md`](docs/report/README.md)。

## 背景：TSP 与 A\*

### 旅行商问题（TSP）

**经典表述**：给定若干城市及任意两城之间的距离（或代价），求一条**访问每个城市恰好一次**并最终**回到出发点**的闭合路线，使**总路程最小**。

**与本仓库的对应关系**：城市是欧氏平面上的点；**合法移动**仅限于事先生成的 **k 近邻稀疏图**上的边（对称无向），而非在完全图上任意跳。起点可在配置中固定（整数）或设为随机（`"random"`），目标是该图上的一条**闭合哈密顿回路**。若图中不存在这样的回路，搜索会以失败结束（见下文「为何有时无法完成求解」）。

### A* 与本项目中的状态

**A\***：维护从起点到当前状态的真实代价 **g**，并用启发函数 **h** 估计剩余代价；按 **f = g + h** 从优先队列（开放表）中取出待扩展状态。当 **h 可采纳**（对任意状态不高估到达目标的真实最优剩余代价）时，首次取出的目标状态对应**最优解**。本实验的 **h** 基于「未访问点集上的 MST 松弛」等构造，与 **k-NN 图**上的真实行走模型未必一致，报告里需讨论可采纳性与扩展次数（见「三种启发函数」与「实验报告提示」）。

**状态表示**（[`src/astar_tsp_solver.py`](src/astar_tsp_solver.py)）：**位掩码 `mask`** 记录已访问过的顶点集合，**`current`** 为当前所在城市；**`phase`** 区分「仍在按边扩展、访问尚未访问的城」与「已访问全集、经边回到起点」的终止阶段，便于正确识别目标。

### 可行解估算（是否存在闭合哈密顿回路）

[`src/feasibility_estimate.py`](src/feasibility_estimate.py) 提供 **`estimate_feasible_closed_tour`**：先做**必要条件**（图是否连通、n≥3 时是否每点度数≥2 等），再用**限步 DFS 回溯 + 随机邻接顺序**做短试探。若试探找到回路，则**一定存在**可行解；若未找到，**不能**据此断定无解（哈密顿回路判定为 NP 完全）。**GUI** 在随机生成或加载实例后会在日志中打印估算；**`demo`** 的 `summary.txt` 中也会写入若干行说明。

### 规模估算与混合求解（`tables` / `demo` / GUI）

位掩码状态的上界量级约为 **2·n·2ⁿ**（见 [`src/solve_policy.py`](src/solve_policy.py) 的 `state_space_upper_bound`）。**n 较大时**在普通 PC 上做**精确最优** A* 不现实，因此默认启用 **`auto_solver`**（见 `config.example.json`）：

- **n ≤ `astar_max_n`（默认 20）**：A*（`f = g + h`, `ε = 1`）；先保证回路合法性，再由小规模精确验证器尝试给出“可证最优”结论。
- **`astar_max_n` < n ≤ `weighted_max_n` 且 `weighted_astar_epsilon` > 1**：**加权 A***，`f = g + ε·h`，更快但**不保证最优**。
- **否则**：在图上构造**可行初解**（**确定性最近邻** → 失败则 **RCL 随机最近邻**多重启 → 仍失败则 **DFS 回溯哈密顿回路**多轮随机邻序），再对初解做 **2-opt 模拟退火**；最终会做严格合法性校验，确保 `success=True` 一定是合法闭合回路。可在 `config` 中调节 `fallback_random_nn_tries`、`fallback_rcl_size`、`fallback_backtrack_restarts` 等。

将 **`auto_solver`** 设为 `false` 时**始终**跑 A*（大 n 可能极慢）。可用 **`force_stronger_solver`** 强制不走 A*：优先使用 Held-Karp 精确 DP（`n <= stronger_exact_max_n` 且未超时），否则回退到贪心+模拟退火；可用 **`force_monster_solver`** 启用 Monster 分支定界（先用贪心+SA 热启动上界，再做精确剪枝；超时可返回当前 best-so-far）；可用 **`force_bruteforce`** 强制纯暴力（受 `bruteforce_max_n` 与 `bruteforce_time_limit_sec` 保护）。GUI 中可勾选 **「强制精确 A*」** 覆盖上述阈值。批量 CSV 中会多出 **状态空间上界、求解方法、是否最优** 等列；其中“是否最优”由小规模精确验证器（Held-Karp）认证，非认证场景默认不宣称最优。

## 环境

- Python 3.10+（建议）  
- 依赖：`pip install -r requirements.txt`  
- 开发/报告自动化：`pip install -r requirements-dev.txt`（含 **pytest**）  
- **GUI** 需要 **tkinter**（多数系统随 Python 提供）

## 安装

```bash
cd Astar_exp
pip install -r requirements.txt
pip install -r requirements-dev.txt   # 可选：跑测试与报告配图脚本
```

## 项目结构

| 路径 | 说明 |
|------|------|
| `config.json` | 运行参数（缺省键见 `main.py` 中 `_DEFAULT`） |
| `main.py` | 子命令：`tables` / `demo` / `gui`；`--config` 指定 JSON |
| `src/map_generator.py` | k-NN 随机图、连通修补、实例 CSV 读写 |
| `src/heuristics.py` | 三种启发（见下节） |
| `src/astar_tsp_solver.py` | `AStarTSPSolver`、`search` / `search_stepwise`（加权 A*：`f = g + ε·h`；可选 `cancel_check` 提前终止） |
| `src/feasibility_estimate.py` | 可行闭合回路是否可能存在的必要条件 + 快速回溯试探（`estimate_feasible_closed_tour`） |
| `src/solve_policy.py` | 按城市数选择精确 A* / 加权 A* / 贪心+模拟退火（`solve_tsp_auto`，含结果验证链路） |
| `src/solution_verifier.py` | 严格校验回路合法性 + 小规模 Held-Karp 最优性认证 |
| `src/tsp_heuristic_solvers.py` | 图上限定贪心最近邻、合法边上 2-opt 模拟退火 |
| `src/visualization.py` | 静态 PNG；`launch_gui`（Tk 演示） |
| `src/mpl_compat.py` | 过滤混装 Matplotlib 的常见警告 |
| `scripts/run_experiments.py` | 等价于 `python main.py tables …` |
| `scripts/launch_gui.py` | 等价于 `python main.py gui`（支持 `--config`，见下） |
| `scripts/generate_report_assets.py` | 生成实验报告用表 1–3、样例地图与对比图（见 `docs/report/README.md`） |
| `config.report.json` | 报告批量实验覆盖配置（输出目录 `data/report/…`） |
| `tests/` | `pytest` 用例（表字段、PNG、drawio 存在性） |
| `requirements-dev.txt` | 开发依赖（含 `pytest`） |
| `docs/report/README.md` | 报告配图与 Word 表头说明 |
| `docs/gui_mindmap.drawio` | GUI 设计思维导图（draw.io 源文件） |
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
仅写 `python main.py` 时，会读取 `config.json` 的 `default_cmd`（默认 `demo`）并执行对应子命令。

`scripts/launch_gui.py` 与 `scripts/run_experiments.py` 若带 `--config FILE`，都会转换为 `main.py --config FILE <子命令>`。

## 快速运行

### 1. 配置

```bash
cp config.example.json config.json   # 首次可复制模板后编辑
```

常用键：`default_cmd`、`sizes`、`k`、`repeats`、`seed`、`max_expansions`、`time_limit`、`save_sample_maps`、`auto_time_limit_n_ge_50`、`auto_time_limit_n_ge_50_sec`（n≥50 自动时限秒数，默认 600）、`tables_start`、`force_exact_astar`、`force_stronger_solver`、`force_monster_solver`、`force_bruteforce`、`monster_max_n`、`monster_time_limit_sec`、`bruteforce_max_n`、`bruteforce_time_limit_sec`、`stronger_exact_max_n`、`stronger_exact_time_limit_sec`、`verify_exact_max_n`、`verify_exact_time_sec`；`demo_*`（含 `demo_start`）；`intermediate_dir`；`gui_default_*`（含 `gui_default_start`）。

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
- 若 `auto_time_limit_n_ge_50` 为 **true**（默认），且未设上述两项、且 `sizes` 中含 **n≥50**，则每次运行自动 **`auto_time_limit_n_ge_50_sec` 秒**时限（默认 **600**，即 10 分钟；stderr 有提示）。

### 3. 图形界面（GUI）

```bash
python main.py gui
python main.py --config my.json gui
python scripts/launch_gui.py
python scripts/launch_gui.py --config my.json
```

功能简述：随机生成 / 加载 CSV；三种启发；**按规模自动选 A* / 加权 A* / 贪心+模拟退火**（与 `config` 中策略一致），可勾选「强制精确 A*（忽略规模估算）」「强制更强算法（禁用A*）」「强制 Monster 分支定界（禁用A*）」「强制纯暴力（小规模）」；并在 GUI 的「高级策略参数」中直接调整阈值、加权参数、SA 参数、回溯参数、Monster 参数、验证器参数等。支持**一键导入/导出 GUI 参数 JSON**；当起点设为 `random` 时，同一张地图只在首次运行随机一次，之后复用该起点（换图后重新随机）。地图初始化完成后会自动计算一条最佳已知回路并以**金色/蓝色**中空高亮（可证最优为金色，未认证 best-known 为蓝色）。**精确/加权 A* 运行中可点「终止 A* 搜索」**（`search` / `search_stepwise` 内 `cancel_check`）；后台一次跑完或单步、自动步进；日志含 **f, g, h、扩展次数**；**单步「本步搜索耗时 / 累计搜索耗时」仅统计 `next(生成器)` 内 CPU 时间，不含你在界面上的挂机等待**；结束时有与实验字段对齐的**单次汇总**（单次成功时最优=最差=平均路径长度）；当规模不大（`n <= verify_exact_max_n`）时会自动做最优性认证。

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
from src.solve_policy import solve_tsp_auto
import numpy as np

g = generate_random_map(12, k_neighbors=4, seed=0)
g = ensure_connected_fallback(g, np.random.default_rng(0))
start = 0  # 也可随机：np.random.default_rng(0).integers(0, g.n)
solver = AStarTSPSolver(g, heuristic="euclidean", start=start)
res = solver.search(max_expansions=500_000, time_limit_sec=60.0)
print(res.success, res.cost, res.expansions, res.method, res.optimal)

# 或按 config 自动选精确 A* / 加权 A* / 贪心+SA（键同 config.example.json）
cfg = {
    "auto_solver": True,
    "astar_max_n": 20,
    "weighted_max_n": None,
    "weighted_astar_epsilon": None,
    "verify_exact_max_n": 14,
    "verify_exact_time_sec": None,
}
res2 = solve_tsp_auto(g, "euclidean", start, cfg, max_expansions=500_000, time_limit_sec=60.0)
```

### 6. 实验报告：测试与配图

```bash
pip install -r requirements-dev.txt
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q    # 若本机 pytest 插件冲突可保留此前缀
python scripts/generate_report_assets.py      # 完整：表1–3 + 样例地图 + 对比图（耗时较长）
python scripts/generate_report_assets.py --fast # 快速冒烟
```

- 自动化测试位于 `tests/`，校验批量表字段、`save_map_png`、流程图与 GUI 思维导图 `.drawio` 存在性。  
- 报告用配置：[`config.report.json`](config.report.json)；生成物默认在 `data/report/`（根目录 `.gitignore` 已忽略 `data/`）。  
- 流程图 / 思维导图 PNG：用 draw.io 从 [`docs/astar_tsp_flowchart.drawio`](docs/astar_tsp_flowchart.drawio)、[`docs/gui_mindmap.drawio`](docs/gui_mindmap.drawio) 导出，详见 [`docs/report/README.md`](docs/report/README.md)。

## 为何有时无法完成求解？

1. **图上无哈密顿回路**：算法只在 **当前 k-NN 图** 的边上找回路；k 小或点分布不利时，**可能不存在**合法闭合回路。连通修补 **不保证** 存在回路。失败信息多为 **`no solution (open empty)`**。可尝试 **增大 k（3～5）**、换 **seed** 再生成。  
2. **扩展或时限截断**：达到 **`max_expansions`** 或 **`time_limit_sec`** 会提前结束（`expansion limit` / `time limit`），大 n 时很常见。  
3. 报告需写明：是否使用上限、**失败次数**、表内统计是否**仅针对成功运行**；并区分“可证最优”与“当前最好但未认证”。

## 实验报告提示

- **custom**：写明实现为 **L∞ + 与本项目一致的 MST 松弛**（课程「自定义距离」可对切比雪夫度量）；可写 **d∞≤d₂ ⇒ 同子集 MST 权 MST∞≤MST₂ ⇒ h_custom≤h_eucl**。**不要**偷换成「故对 k-NN 图上的真实剩余最优代价一定可采纳」——应写清：**图上是欧氏 k-NN**，启发是 **完全图直线度量下的 MST**，二者模型不同。  
- **manhattan**：同上，L1 与欧氏边权的关系、扩展数与解对比。  
- **n=50/100** 写清时限与扩展策略。  
- 素材：**GUI**、`demo`（PNG、`events.tsv`）、`tables`（CSV、`run_summary.txt`）。

## 故障排除：Matplotlib Axes3D 警告

混装系统包与 pip 的 Matplotlib 时可能出现 **`Unable to import Axes3D`**。本实验仅用 2D 绘图，一般可忽略；[`src/mpl_compat.py`](src/mpl_compat.py) 会过滤该警告。根治：卸系统 `python3-matplotlib` 或单独 **venv** 安装依赖。

## 许可证

课程作业用途，按学校要求使用。
