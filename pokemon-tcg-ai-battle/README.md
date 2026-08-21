# Pokémon TCG AI Battle

The Pokémon Company - PTCG AI Battle Challenge Simulation 的归档研究项目。
这里保留能够公开、复现并继续维护的智能体代码、训练框架、本地评测工具和比赛复盘；原始回放、外部智能体、模型缓存、提交归档及认证信息不进入 Git。

## 比赛记录

- [参赛全程复盘](docs/COMPETITION_RETROSPECTIVE_2026.md)
- [四十次提交与全部可恢复方案编年史](docs/COMPETITION_ATTEMPT_CHRONICLE_2026.md)

复盘中的成绩均按文档注明的历史快照解释，不代表仍在变化的实时榜单。

## 代码结构

- `agents/`：保留下来的可审计智能体和冻结模型；
- `rolling_policy/`：回放抽取、特征、模仿模型和可移植树推理；
- `league_selfplay/`：多策略自我对战、PPO、单点干预和统计门禁；
- `scripts/`：数据下载、训练、打包及本地对局入口；
- `tests/`：策略、训练、导出和提交物回归测试。

一次性调查脚本和包含第三方内部细节的工作日志已经清理；能够支持复盘结论的长期实现保留在上述模块中。

## 环境

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-train.txt
```

接受 Kaggle 比赛规则并在本机配置 Kaggle CLI 后，可下载官方数据与模拟引擎：

```bash
./scripts/download_official_data.sh
```

凭据只保存在 Kaggle CLI 的用户配置中，不要复制到项目目录、归档或提交记录。

## 本地评测

单场对局：

```bash
python scripts/run_local_match.py \
  --agent0 agents/candidate_grimmsnarl_imitation_full_v2 \
  --agent1 path/to/opponent
```

交换座位重复评测：

```bash
python scripts/evaluate_local.py \
  --agent-a agents/candidate_grimmsnarl_imitation_full_v2 \
  --agent-b path/to/opponent \
  --games-per-seat 10 \
  --out logs/example.csv
```

多候选、多对手矩阵使用 `scripts/evaluate_matrix.py`。本地模拟依赖官方 `cg` 运行时；缺少该运行时时，纯策略和统计单元测试仍可运行，但完整对局测试不可运行。

## 测试

```bash
python -m pytest tests
```

训练相关测试需要 NumPy、SciPy、scikit-learn、PyTorch 和 pytest，版本下限记录在 `requirements-train.txt`。
