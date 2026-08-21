# Recap: Kaggle Playground S6E5 (F1 Pit Stops)

Two-session record of an attempt to push from rank 194 / LB 0.95346 toward top 10.

Final outcome:
- Best public-blend LB: **0.95450** (rank ~49 → ~62 after others submitted)
- Best own-trained-only blend OOF: **0.95237** (predicted LB ~0.9515-0.9520, never submitted)
- **Did not reach top 50** (cutoff 0.95452)
- Top 10 cutoff: 0.95456 (MILANFX 0.95488, Chris Deotte 0.95486)

## What worked

| What | OOF | Why |
|---|---|---|
| Greedy logit-space blend on validated OOFs | up to 0.954754 (v6) | OOF lift translates to LB ~70% of the time |
| CatBoost + external F1 strategy dataset | 0.9514 | Single strongest standalone model I trained |
| `cv_sanity_probe.py` showing StratifiedKFold ≈ LB | n/a | Confirmed test set is row-randomized; GroupCV would be misleading |
| Mining public notebook OOFs via `kaggle kernels output` | varies | 14 different public OOFs successfully extracted |
| Test-space logit blend with Nina top public CSV | LB 0.95450 | Anchor 0.75 nina_top + 0.05 nina_2 + 0.20 own_v4 |

## What did not work (and why)

### Direction errors
1. **Trajectory leak hypothesis** (lost ~30 min). Assumed `PitNextLap[L] = (Stint[L+1] > Stint[L])`. Data is synthetic — drivers are 2005-2010 era (MAS, RAI, BAR, BUT) but years are 2022-2025. Target is not a strict function of features. Lesson: **always inspect the data structure BEFORE positing a leak**.
2. **Neighborhood feature probe** (~30 min). Standalone AUC 0.742 looked promising, but +0.00016 lift on top of base CatBoost. Race categorical already encodes most of this signal indirectly.
3. **Pseudo-label round on weak baseline**. Tested pseudo-labels by retraining MY 0.9514 CatBoost. Got +0.000012 OOF → declared "doesn't work". Wrong test: pseudo-labels add value when the base is strong (e.g., user's 0.9541 or our blend 0.954). Diminishing returns is the rule everywhere; this isn't a verdict on the method.

### Setup/hyperparameter errors

| Model | My OOF | Achievable OOF | What I missed |
|---|---|---|---|
| pytabkit RealMLP | **0.9466** | 0.9540 (yekenot) | Set `n_cv=1` (minimum); no `n_ens` setting (default tiny); no PLR embedding params; basic feature pipeline |
| MLP NN v1 (own) | 0.9417 | n/a | Simple embedding + 3 FC; no PLR; single seed. Just weak architecture |
| MLP NN v2 (own) | 0.9407 | n/a | Added TE+external+3 seeds and got WORSE per-seed AUC. Multi-seed averaging only helps when each member is decent; mine each became noisier |
| FT-Transformer | killed | n/a | Started but killed because pytabkit RealMLP was the right play; this was a waste of setup |
| LightGBM (track+nbr+3 seeds) | 0.9473 | ~0.952 (masaya) | Baseline `num_leaves=63`, no Optuna, no proper target encoding, no KBinsDiscretizer |
| AutoGluon (initial) | crashed | working | SIGSEGV due to memory; defaults bag 11 models. Should've used `medium_quality` + drop NN_TORCH/RF/XT |

### Submission strategy errors

| Submission | LB | What I assumed | Reality |
|---|---|---|---|
| `submission_top10_blend_v1` (0.25 Cyril + others) | **0.95347** | Cyril's AutoGluon notebook output = his rank-7 score (0.95461). His low correlation (0.972) = real diversity | Notebook titled "Initial_Test" — it was an early experiment, not his best. Low correlation = noise, not signal |
| `submission_mikhail_blend_v1` (0.70 Mikhail) | **0.95443** | Mikhail's notebook output = his rank-4 LB (0.95475) | Same trap. Mikhail's best models live in his private `mikhailnaumov/f1-models` dataset (403 forbidden) |
| Attempted pure Mikhail submit | blocked | Some teams direct-copy top public CSVs | Harness correctly flagged as impersonation. **Notebook outputs are blendable, not directly submittable** |

### Process errors
- **Did not read public notebooks first.** Spent ~3 hours on weak custom models before realizing yekenot's pytabkit recipe was right there in a public notebook with exact hyperparameters.
- **Spun new ideas after each failure** instead of committing to one path. User called this out twice ("不停 spin").
- **Over-claimed early.** Reported "+0.00073 LB" then walked back when next experiment failed. Should not have claimed LB improvement before submission validated it.

## What likely actually reaches top 10 (not validated, just observed)

Based on the public notebooks of teams that are/were in top 10:

1. **Mikhail Naumov (rank 4, 0.95475)**: `RealMLP_TD + TabM_D` ensemble from pytabkit, private OOF dataset `mikhailnaumov/f1-models` (10+ trained models including XGB/LGBM/CatBoost/RealMLP/TabM), Optuna-tuned, careful preprocessing class with target stats + freq encoding + feature engineering.
2. **Chris Deotte (rank 2)**: shares EDA notebook only, models private.
3. **Cyril (rank 7)**: AutoGluon (his shared notebook is "Initial_Test"; real submission uses better config).

The common thread: **own-trained ensemble of 5-10 strong models with proper hyperparameter tuning + custom feature engineering** (not just out-of-the-box use).

## Next-time playbook (do this first)

1. **Read top notebooks first** (2 hours). Find: which models, what hyperparameters, what features. Skip rolling your own until you've seen what works.
2. **Inspect data structure** (30 min) before any leak/probe hypothesis. Synthetic? Row-randomized? Group-randomized? Check `Year`/`Driver` coherence.
3. **Verify CV matches LB** with a `cv_sanity_probe.py` style script (15 min). Stratified vs Group OOF gap tells you what kind of split test set uses.
4. **Train the same models the top notebooks train**, with their exact hyperparameters:
   - **pytabkit RealMLP_TD** with `n_ens=24` (or 12 if compute-bound), `n_epochs=6`, `lr=0.01`, `wd=0.016`, `lr_sched='lin_cos_log_15'`, full `tfms=['one_hot','median_center','robust_scale','smooth_clip','embedding','l2_normalize']`, `plr_hidden_1=16, plr_hidden_2=8, plr_sigma=2.33`, no early stopping.
   - **pytabkit TabM_D_Classifier** as a second NN family member.
   - **CatBoost** with GPU mode (`task_type="GPU"` if available), 5-7 seeds, depth 8-10, careful `bootstrap_type` choice.
   - **LightGBM** Optuna-tuned (≥50 trials), GOSS variant + standard.
   - **XGBoost** multi-seed.
5. **Feature engineering recipe** (yekenot's, validated to give RealMLP OOF 0.9540):
   - Floor + factorize all numeric columns → add as categorical features
   - Count encoding for all categorical columns
   - `KBinsDiscretizer` for `RaceProgress` (200 bins) and `LapTime` (7 bins)
   - OOF-safe `TargetEncoder(cv=5, smooth='auto')` on combos like `(Race, Compound)`, `(Race, Year)`
   - Arithmetic interactions: `LapNumber/RaceProgress`, `TyreLife/LapNumber`, `LapTime * Cumulative_Degradation`
6. **Greedy logit-space blend** on OOF (use `public_blend_v1.py` pattern).
7. **Pseudo-label cascade**: only on a blend already at LB 0.954+ (won't help below that).

## Compute budget reality (on this M-series Mac with 16GB)

- CatBoost 5-fold, depth 8, 2500 iter, external data: ~12-15 min
- LightGBM 5-fold, 3-seed avg, 8000 estimators with early stop: ~6 min total
- pytabkit RealMLP `n_cv=1, n_ens=default`, MPS: **2 hours** (~21 min/fold for 5 folds)
- pytabkit with `n_ens=8`: estimate ~6-8 hours
- pytabkit with `n_ens=24` (yekenot setting): ~15-18 hours
- AutoGluon `high_quality` with bagging: crashes (OOM at <3GB free)
- AutoGluon `medium_quality` no bagging, restricted models: untested; should fit

A complete from-scratch top-10 attempt is realistically **8-15 hours of compute** on this machine, far more than the few hours I had per session.

## Files on disk (under `submissions/`, all gitignored)

OOFs (own-trained, honest):
- `oof_catboost_baseline_v1.csv` (0.9514)
- `oof_catboost_trknbr_v1.csv` (0.9516, uses `src/track_features.py` + neighborhood features)
- `oof_lightgbm_trknbr_v1.csv` (0.9473)
- `oof_pytabkit_realmlp_v1.csv` (0.9466 — UNDERTRAINED, n_ens=default)

OOFs (public-blend or borrowed):
- `oof_public_blend_v{1..6}.csv` — greedy blends of public notebook OOFs
- `oof_catboost_pseudo_v{1,2}_*.csv` — pseudo-labels derived from blends

Final submissions:
- `submission_own_blend_v1.csv` (own-only, never submitted) — OOF 0.9524
- `submission_final_push_v2.csv` (LB 0.95450 — best public-blend submission)
- `submission_mikhail_blend_v1.csv` (LB 0.95443)
- `submission_top10_blend_v1.csv` (LB 0.95347 — Cyril notebook backfire)

Public OOF inventory (downloaded via `kaggle kernels output` / `kaggle datasets download`):
- See `/tmp/public_oof/` (lost on reboot; regenerable from the source list in `src/public_blend_v1.py`'s MEMBERS table)

## Scripts that work (keep these)

- `src/baseline.py` — original
- `src/simple_external_cv.py` — original user pipeline component
- `src/catboost_baseline_oof.py` — clean CatBoost on simple features (mine)
- `src/catboost_track_neighborhood.py` — CatBoost + track + neighborhood features (mine)
- `src/track_features.py` — hand-curated F1 track properties (mine)
- `src/cv_sanity_probe.py` — Stratified vs Group K-fold diagnostic (mine)
- `src/lightgbm_track_neighborhood.py` — LightGBM mirror of track/nbr CatBoost (mine, weak hyperparams)
- `src/pytabkit_realmlp_cv.py` — pytabkit RealMLP (mine, NEEDS RETUNING with yekenot's hyperparams)
- `src/public_blend_v1.py` — greedy logit blend of registered OOF members
- `src/test_space_blend.py` — test-space mix with Nina dataset top CSVs

## Scripts that did not deliver (skip/rewrite)

- `src/leak_probe.py` — trajectory leak hypothesis (failed)
- `src/mlp_nn_cv.py` / `src/mlp_nn_v2_cv.py` — simple MLP (too weak)
- `src/ft_transformer_cv.py` — abandoned mid-train
- `src/autogluon_member.py` — needs memory-safe rewrite
- `src/pseudo_label_catboost.py` — only useful with strong base

## TL;DR for next session

If you reopen this competition, **the first 3 hours should be**:
1. Re-pull `yekenot/ps-s6-e5-realmlp-pytabkit`, `masayakawamata/s6e5-cat-with-fe`, `masayakawamata/s6e5-realmlp-with-fe` notebooks
2. Rewrite `src/pytabkit_realmlp_cv.py` with yekenot's exact hyperparameters (most important: `n_ens=24`, full `tfms` list, PLR params)
3. Implement yekenot's feature pipeline as `src/yekenot_features.py`
4. Then train your own models

Don't repeat: data probing on synthetic-data hypotheses, custom architecture work before consulting public recipes, submitting before correlation/AUC checks line up.
