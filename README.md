# Kaggle Carry

这是一个总的 Kaggle 实验仓库，用来沉淀我们用 LLM 协作打 Kaggle 的代码、复盘和可复现流程。

仓库只放适合公开的内容：代码、说明、依赖清单和空目录占位文件。Kaggle 原始数据、外部数据、OOF、submission CSV、模型文件和本地虚拟环境都不会提交。

## Projects

- `playground-series-s6e5-f1-pit-stops/`: Kaggle Playground Series S6E5：Predicting F1 Pit Stops。当前最好提交 Public Score `0.95346`，提交时排名 `194 / 1328`。
- `pokemon-tcg-ai-battle/`: The Pokémon Company - PTCG AI Battle Challenge Simulation。已归档完整研发代码、40 次正式提交复盘和实验编年史。

## Layout

每个比赛一个独立目录，尽量保持同样结构：

```text
competition-name/
  README.md
  requirements.txt
  src/
  data/
  submissions/
  leaderboard/
```

其中 `data/`、`submissions/`、`leaderboard/` 只提交 `.gitkeep`，真实文件留在本地或从 Kaggle 重新下载。
