# Kaggle 入门工程：Predicting F1 Pit Stops

我们选的比赛是 Kaggle Playground Series S6E5:

- 比赛链接: https://www.kaggle.com/competitions/playground-series-s6e5
- 任务: 预测 F1 车手下一圈是否会进站
- 类型: Beginner + Tabular，二分类概率预测
- 指标: ROC AUC
- 提交格式: `id,PitNextLap`，第二列是 0 到 1 的概率

我在 2026-05-10 打开页面时看到它还有约 22 天结束，参与人数很多，适合作为第一次正式 Kaggle 练习。

## 1. 安装环境

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. 下载数据

Kaggle 要求登录并接受比赛规则后才能下载数据。你有两种方式：

### 方式 A：网页下载

1. 打开 https://www.kaggle.com/competitions/playground-series-s6e5/data
2. 登录 Kaggle
3. 点 `Join Competition` 并接受规则
4. 点 `Download All`
5. 把下载得到的 zip 放到:

```text
data/raw/
```

脚本会自动解压 zip。

### 方式 B：Kaggle API

在 Kaggle 个人设置里创建 API token，把 `kaggle.json` 放到 `~/.kaggle/kaggle.json`，然后运行：

```bash
chmod 600 ~/.kaggle/kaggle.json
source .venv/bin/activate
kaggle competitions download -c playground-series-s6e5 -p data/raw
```

## 3. 训练 baseline 并生成提交文件

```bash
source .venv/bin/activate
python src/baseline.py --data-dir data/raw --output submissions/submission_baseline.csv
```

运行完成后会看到验证集 ROC AUC，并得到:

```text
submissions/submission_baseline.csv
```

## 4. 提交

网页方式：进入比赛的 Submit 页面，上传 `submissions/submission_baseline.csv`。

API 方式：

```bash
source .venv/bin/activate
kaggle competitions submit \
  -c playground-series-s6e5 \
  -f submissions/submission_baseline.csv \
  -m "beginner baseline: hist gradient boosting"
```

## 我们的第一阶段目标

先跑通一次完整流程：

1. 下载数据
2. 训练 baseline
3. 生成 `submission.csv`
4. 提交到 Kaggle
5. 记录 Public LB 分数

之后再开始一点点优化，比如特征工程、交叉验证、LightGBM/CatBoost、模型融合。

## 当前最佳本地方案

这版使用比赛允许的原始 F1 strategy dataset 做外部训练样本增强，并在多组 OOF 预测上做外部 key prior、SGD meta stack、LightGBM meta stack 和固定 alpha blend。

> 注意：这个 public repo 只提交代码和说明，不提交 Kaggle 原始数据、外部数据、OOF 文件或 submission CSV。数据和输出都被 `.gitignore` 排除了。

```bash
mkdir -p data/external
curl -L -o data/external/f1-strategy-dataset-pit-stop-prediction.zip \
  "https://www.kaggle.com/api/v1/datasets/download/aadigupta1601/f1-strategy-dataset-pit-stop-prediction"
mkdir -p data/external/f1_strategy
unzip -o data/external/f1-strategy-dataset-pit-stop-prediction.zip -d data/external/f1_strategy
```

核心脚本：

```bash
# 基础模型和早期外部数据增强
python src/simple_external_cv.py --models catboost,lgbm --n-splits 5 --catboost-iterations 2500

# 外部 key prior 融合和 residual/meta 探索
python src/external_greedy_prior_blend.py
python src/external_prior_meta_stack.py --suffix external_prior_meta_stack_v1

# LightGBM meta stack
python src/external_prior_lgb_stack.py \
  --base-oof submissions/oof_external_prior_meta_stack_v1.csv \
  --base-submission submissions/submission_external_prior_meta_stack_v1.csv \
  --suffix external_prior_lgb_stack_v1

# 最终提交版固定 alpha blend
python src/fixed_alpha_blend.py --alpha 0.767
```

已提交最好版本：

- OOF AUC: `0.9541005311`
- Kaggle Public Score: `0.95346`
- Public LB rank at submission time: `194 / 1328`

已提交文件:

```text
submissions/submission_external_prior_lgb_stack_v1_alpha0767.csv
```

当前本地最好版本还没有提交到 Kaggle：

- OOF AUC: `0.9541801786`
- 相比已提交 OOF: `+0.0000796475`
- 方法: 先加入外部 F1 strategy dataset 的低权重 nearest-neighbor 校准信号，再用 regularized logistic meta blend 做二次校准
- 本地候选文件:

```text
submissions/submission_regularized_nn_base_logreg_seedmean_v1.csv
```

## P0 序列特征实验记录

新增脚本。下面这条命令复现当前 `external_nn_greedy_v1` 候选：

```bash
python src/sequence_features_cv.py --diagnose-only
python src/sequence_features_cv.py --suffix sequence_features_lgbm_v1
python src/sequence_features_cv.py --include-lead --suffix sequence_features_lgbm_lead_v1
python src/sequence_features_cv.py --target-encoding --suffix sequence_features_lgbm_te_v1
```

验证结果：

- 外部精确 lookup 命中率最高的 key 不是 Driver，而是 `Year/Race/LapNumber/Position/Stint` 一类模板字段；train/test 命中率约 `77%`。
- 单独的序列 LGBM 不强：lag/rolling 版 OOF `0.9440807760`。
- 加 `lead1` 未来圈特征没有起飞：OOF `0.9440214278`。
- fold-safe target encoding 版更差：OOF `0.9315159607`。
- 直接把外部 lookup prior 微量 blend 进当前 best，最多只涨约 `+0.0000027` OOF，说明当前 external prior/meta stack 已经吃掉了大部分这类信号。

结论：简单序列特征不是独立突破口。下一步应该把序列特征作为补充输入接入现有 external-train CatBoost/LGBM 管线，或者转向更低相关性的表格 NN/多 seed CatBoost。

## 外部近邻校准实验记录

新增脚本:

```bash
python src/external_nn_probe.py \
  --configs yr_race_knn,yr_race_lap_knn,yr_race_lap_compound_knn,yr_race_phase_knn \
  --ks 1,3,5,10,25 \
  --suffix external_nn_greedy_v1 \
  --greedy-steps 6 \
  --greedy-max-alpha 0.04 \
  --greedy-grid-size 31
```

验证结果：

- 单个最强近邻信号是 `Year/Race` 分桶、`k=5`，OOF 从 `0.9541005311` 到 `0.9541245164`。
- 贪心融合后继续选中 `Year/Race/Stint/Compound k=3`，最终 OOF `0.9541353975`。
- 完整 probe 里额外跑过的 `Year/Race/LapNumber/Position` 分桶过稀疏，候选集平均只有 1 条，单独 AUC 约 `0.7032`，不应继续投入。
- 更精确的 `LapNumber` / `Compound` lookup 没有起飞，说明外部数据不是可直接查表的隐藏真值，只能当很低权重的分布校准。

当前判断：外部近邻能带来稳定小增益，但不是冲前十的主钥匙。后续更值得做的是低相关性模型，比如更强多 seed CatBoost、表格 NN，或者把近邻信号作为 feature 接入 meta stack，而不是继续手工 lookup。

## 正则 meta 校准实验记录

新增和更新：

- `src/external_nn_meta_lgb_stack.py`: 把 KNN greedy 预测作为特征接入 LightGBM 二层。
- `src/regularized_meta_blend.py`: 支持选择 `--base-name nn_greedy`，并加入 current best / KNN greedy 预测特征和 seed-mean ensemble。

LightGBM 二层没有继续提升：

```bash
python src/external_nn_meta_lgb_stack.py \
  --suffix external_nn_meta_lgb_stack_v1 \
  --seeds 101,202 \
  --leaves 7,15
```

结果：4 个 LGB meta 成员与 `external_nn_greedy_v1` 融合时最佳 alpha 都为 `0`，说明树模型二层已经饱和。

正则 logistic 校准有稳定小增益：

```bash
python src/regularized_meta_blend.py \
  --suffix regularized_nn_base_logreg_seedmean_v1 \
  --base-name nn_greedy \
  --model logreg \
  --seeds 17,29,41 \
  --c-grid 0.03 \
  --max-alpha 0.8 \
  --include-risky
```

最佳结果：

- `seed=17, C=0.03, class_weight=balanced`
- Meta AUC: `0.9541211039`
- Blend alpha: `0.396`
- Blend OOF AUC: `0.9541801786`
- 相比 `external_nn_greedy_v1`: `+0.0000447810`
- 相比已提交 `external_prior_lgb_stack_v1_alpha0767`: `+0.0000796475`

Seed mean ensemble 为 `0.9541793602`，略低于 seed 17 单模。当前判断：正则线性校准是目前最有效的局部提分路线，但增益仍是 `1e-5` 量级；要冲前十仍需要新的低相关强成员。

## Submission 对齐说明

`fixed_alpha_blend.py` 现在支持显式恢复“已融合 meta submission”里的原始 meta 预测，避免 OOF 公式和 submission 公式不一致：

```bash
python src/fixed_alpha_blend.py \
  --alpha 0.767 \
  --suffix external_prior_lgb_stack_v1_alpha0767_rawmeta \
  --meta-submission submissions/submission_external_prior_lgb_stack_v1.csv \
  --meta-submission-blend-base submissions/submission_external_prior_meta_stack_v1.csv \
  --meta-submission-blend-alpha 0.7
```

实测这个显式恢复版与当前 `submission_external_prior_lgb_stack_v1_alpha0767.csv` 完全一致；这次修改主要是把可复现逻辑固化到代码里。
