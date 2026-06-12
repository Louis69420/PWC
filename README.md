# GameVault — Acquisition Radar: sentiment engine + game scorecard

## 1. The problem (in my own words)

GameVault acquires already-proved games and republishes them on new platforms, they are a small company with low risk tolerance and their biggest risk is launching a game that doesn't perform well. At the moment they rely on surface level metrics to decide on which games to launch and slow inconsitent manual game reviews. Most of their data is decentralized and scattered.

### Hints from spec
1. Trend-driven games with enough moment is also valid even though they crash and burn quickly.
2. Since different team members use different standards, a non-black box solution should be applied to ensure explainablity.
3. Since they have low risk tolerance false positives should be minimum.

## 2. Approach

Audited the scattered review data, surfaced three trajectory insights, then split the problem in two: an ML sentiment engine where the data is deep (14,610 labelled reviews) and a transparent weighted scorecard where it is thin (only 23 games). Reviews get scored, games get ranked on a confidence-flagged leaderboard, and the dashboard lets the team tune weights and drill into any title. Diagram in [`docs/solution_design.md`](docs/solution_design.md).

**Repo map**

| Path | What it is |
|---|---|
| `steam_reviews 1.csv` | The dataset (did not push the clients dataset to github)|
| `notebooks/eda.ipynb` | Data audit and exploratory analysis |
| `notebooks/model.ipynb` | Feature decisions, leakage audit, models, evaluation |
| `dashboard/app.py` | Streamlit dashboard (`streamlit run dashboard/app.py`) |
| `outputs/` | Scored records and exported charts |
| `AI_USAGE.md` | Full AI-use disclosure log |

## 3. Key findings from the data

1. **GTA5 went from 68% positive reviews to 25% in quarterly reviews before bouncing back to 65%** — lifetime averages hide sentiment crashes
2. **PUBG early-access sentiment was 80% positive vs 57% post release** — early-access hype can evaporate at launch
3. **Players with less than 10 hours are 47% likely to not recommend while players with more than 10 hours are 70%+ likely to recommend** — hours-played is a clean engagement signal for the recommendation rating


**Data quality findings** (these matter to the client as much as the model):
- 7 reviews have missing text → dropped (negligible, <0.05% of rows).
- 5 of 23 titles have <100 reviews → flagged `thin_coverage` so the scorecard doesn't over-trust sparse games.


## 4. Leakage decisions

Prediction moment = the instant a review is posted. Excluded: **`funny`/`helpful`** (vote counts accumulate *after* posting — zero on a fresh review) and **`title`** (the engine must generalise to candidate games it has never seen, not memorise the catalogue).

## 5. Model results

One model: TF-IDF (uni+bigrams) + log-scaled playtime + early-access flag → Logistic Regression, `class_weight='balanced'`. Metrics are for the minority **Not Recommended** class (29%), not accuracy.

| Evaluation (Not Recommended class) | Precision | Recall | PR-AUC |
|---|---|---|---|
| Random baseline (prevalence) | — | — | 0.29 |
| In-catalogue holdout | 0.685 | 0.810 | 0.827 |
| Unseen games (GroupKFold) | — | — | 0.750 ± 0.053 |

**In business terms:** the engine catches **81% of negative reviews** at 0.69 precision and holds **PR-AUC 0.75 on games it has never seen** — so it can screen review text for candidate titles outside the catalogue, which is the real acquisition use case. The scorecard then ranks the 23 titles: durable-community games (Terraria, Factorio, RimWorld) top the board, while the high-volume giants (GTA V, PUBG, Rust) land mid-table — huge review counts but middling satisfaction and momentum.

**Why-flagged explainability:** LogReg coefficients read directly — strongest negative drivers are *worst, servers, banned, boring, crashes*; strongest positive are *best, amazing, great, love*. See `outputs/feature_importance.png` and the dashboard's "Why flagged" view.

## 6. Dashboard

`streamlit run dashboard/app.py` — three views: Overview (KPIs + headline patterns), Triage queue (scored, filterable), Why flagged (per-record explanation). Screenshots in `dashboard/screenshots/`.

## 7. How I used AI

I used Claude as a programmer: it scaffolded the EDA and model notebooks and constructed graphs of my choosing. It also constructed the dashboard skeleton and helped debug, check the @CLAUDE.md file. All framing, insight selection, leakage rulings, model/metric choices and verification of every number are my own.


## How to run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/ 
streamlit run dashboard/app.py
```
