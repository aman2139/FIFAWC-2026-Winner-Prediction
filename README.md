# ⚽ 2026 FIFA World Cup Prediction Model

> A machine learning + Monte Carlo simulation project built to predict the 2026 FIFA World Cup — purely for fun and learning.

---

## ⚠️ Disclaimer

This project is built **entirely for personal education and entertainment**. It does not:
- Promote, endorse, or facilitate gambling or betting of any kind
- Represent the views of FIFA, any football federation, or any club
- Claim to provide accurate or reliable predictions for real-world use
- Constitute financial, sporting, or advisory advice of any kind

All predictions are probabilistic outputs of a statistical model trained on historical data. Football is inherently unpredictable. This is a coding and data science exercise — nothing more.

---

## 📌 Project Overview

This notebook simulates the entire 2026 FIFA World Cup — all 48 teams, 12 groups, and every knockout round — using a trained machine learning ensemble and 10,000 Monte Carlo iterations. The goal was to explore how far you can push a prediction system using only publicly available data and standard ML tooling.

**Predicted Champion:** 🏆 Spain

---

## 🧠 What It Does

| Stage | Description |
|---|---|
| **Data Ingestion** | Loads historical match results, ELO ratings, goalscorer events, penalty shootout records, and team metadata |
| **Feature Engineering** | Builds 19 per-team features including decay-weighted ELO, market value, clutch factor, form metrics, and environmental modifiers |
| **ML Training** | Trains a soft-voting ensemble (HGB + GB + Random Forest) on 15,000+ historical matches with recency-weighted sample decay |
| **Group Stage Simulation** | Simulates all 12 groups with FIFA tie-breaker rules (pts → GD → GF → fair play) |
| **Knockout Engine** | Simulates Ro32 → Ro16 → QF → SF → Final with shootout clutch-factor for drawn matches |
| **Monte Carlo** | Runs 10,000 full tournament iterations to compute win probabilities for all 48 teams |
| **Backtest** | Validates model against the 2022 World Cup — Argentina correctly ranked top-3 |
| **Visualisations** | 6 charts: winner probabilities, heatmap, convergence plot, group advancement, backtest, sensitivity analysis |

---

## 📊 Model Performance

| Metric | v1 (Baseline) | v2 (Improved) |
|---|---|---|
| 5-Fold CV Accuracy | 0.561 ± 0.009 | **0.587 ± 0.009** |
| Train Accuracy | 0.587 | **0.603** |
| Train-CV Gap | 0.026 | **0.016** |
| Training Samples | ~6,000 | **15,632** |
| Features | 15 | **19** |

The practical ceiling for 3-class football outcome prediction (Home Win / Draw / Away Win) is approximately 62–65% due to the sport's inherent unpredictability. This model sits at ~75–80% of the achievable ceiling given the available data.

---

## 🔧 Feature Set

### Core Features
| Feature | Description |
|---|---|
| `ELO` | Exponential decay-weighted ELO rating (365-day half-life) |
| `MV_m` | Total squad market value (€M) |
| `RSI` | Relative Strength Index — market value vs ELO delta (over/underperformer signal) |
| `SO_rate` | All-time penalty shootout win rate |
| `LateGoal_rate` | % of goals scored in 80th minute or later |
| `Clutch` | Composite: 60% shootout rate + 40% late goal rate |
| `Mgr_weight` | log(WC wins + 1) — institutional pedigree |
| `PPG` | Points per game (last 10 matches) |
| `xG / xGA` | Expected goals for/against (last 10 matches) |
| `CS%` | Clean sheet percentage (last 10 matches) |
| `GD` | Goal difference (last 10 matches) |

### Derived Features (v2)
| Feature | Description |
|---|---|
| `WinRate` | W / (W+D+L) in last 10 matches |
| `DefSolidity` | CS% / xGA — defensive consistency relative to chances conceded |
| `AttEff` | GF / (xG × 10) — finishing efficiency vs expectation |
| `xG_consistency` | Inverse instability signal from losses and draws |

### Environmental Modifiers
| Modifier | Description |
|---|---|
| `Alt_mod` | −3.1% goal expectancy per 1,000m altitude above team's home elevation |
| `Fatigue_mod` | −1% per 1,000km cumulative intra-group travel |
| `Heat penalty` | −0.5% per degree above 30°C at venue |

---

## 🏗️ Architecture

```
data/
├── results.csv              # 49,287 historical international match results
├── eloratings.csv           # ELO ratings per team over time
├── goalscorers.csv          # 47,601 individual goal events
├── shootouts.csv            # 675 penalty shootout records
├── former_names.csv         # Historical team name aliases
└── World_Cup_Project_Misc_data.xlsx
    ├── team_data            # Squad values, WC participations
    ├── past_world_cup_winners
    ├── last10_match_stats   # Form data: xG, xGA, CS%, cards
    ├── most_goals_l10m
    ├── most_assists_l10m
    ├── most_cleansheets_l10m
    └── fixtures             # 2026 group assignments and venues

Code_World_Cup_Project_v2.ipynb   # Main notebook (30 cells)
README.md
```

---

## 🚀 How to Run

### Requirements
```bash
pip install numpy pandas matplotlib seaborn scikit-learn openpyxl
# Optional (will fall back to sklearn if unavailable):
pip install xgboost lightgbm
```

### Steps
1. Clone the repository
2. Place all data files in a `data/` folder in the same directory as the notebook
3. Open `Code_World_Cup_Project_v2.ipynb` in Jupyter
4. Run all cells top to bottom — each cell is labelled with its purpose

> **Note:** Cells must be run sequentially. Variables from earlier cells are used throughout. Runtime is approximately 3–5 minutes depending on hardware; the Monte Carlo loop (Cell 20) runs ~4,800 simulations/second.

---

## 📈 Key Results

### Top 10 Championship Probabilities (Monte Carlo)
| Rank | Team | Win Probability |
|---|---|---|
| 1 | Argentina | 14.13% |
| 2 | Spain | 13.73% |
| 3 | France | 11.42% |
| 4 | Portugal | 8.13% |
| 5 | Brazil | 7.69% |
| 6 | Germany | 5.81% |
| 7 | England | 5.69% |
| 8 | Colombia | 3.40% |
| 9 | Croatia | 3.24% |
| 10 | Belgium | 3.19% |

### Deterministic Bracket Winner
**Spain** — via SF win over Brazil (50%), Final win over France (47%)

### Why Argentina leads Monte Carlo but Spain wins the bracket
These answer different questions. The Monte Carlo randomises match outcomes across 10,000 bracket configurations — Argentina edges ahead because their group (J: Algeria, Austria, Jordan) is easier, giving them marginally more paths to the final across random simulations. Argentina and Spain are co-favourites separated by 0.4 percentage points, which is within statistical noise.

The deterministic bracket always selects the match-by-match favourite — Spain's ELO (2131, highest of all 48 teams) wins every head-to-head calculation in its specific path. Argentina is eliminated by France in the Ro16 in this fixed bracket.

### 2022 Backtest Validation
Argentina ranked **#2** (14.64%) using pre-2022 data only — confirming the model correctly identifies the actual champion as a top-3 favourite.

---

## 🔬 Sensitivity Analysis

Running 2,000 simulations without altitude/heat modifiers shows France benefits most from environmental features (+1.57% with modifiers vs without), confirming the environmental adjustments add genuine signal for teams playing at high-altitude venues like Mexico City (2,250m) and Guadalajara (1,566m).

---

## 📉 Known Limitations

- **No real-time lineup data** — injuries, suspensions, and squad rotation are not modelled
- **No in-match event data** — shots on target, possession phases, pressing metrics unavailable
- **Market value as proxy** — squad value is a noisy signal for current form
- **ELO recency** — even decay-weighted ELO lags behind very recent form changes
- **Draw prediction** — draws are inherently the hardest outcome to classify; the model uses a pair-level draw propensity feature as a partial fix
- **Theoretical CV ceiling ~62–65%** — the remaining gap is locked behind data this project does not have access to

---

## 🛠️ Improvements Made (v1 → v2)

| Cell | Change | Impact |
|---|---|---|
| 5 | Extended training cutoff 2018 → 2010 | +~0.010 CV |
| 6 | Decay-weighted ELO (365-day half-life) | +~0.005 CV |
| 9 | Added `WinRate`, `DefSolidity`, `AttEff`, `xG_consistency` | +~0.006 CV |
| 10 | Updated `FEATURE_COLS`, rebalanced strength weights | Structural |
| 13 | `pair_draw_rate` feature + `sample_weights` (2-yr decay) | +~0.005 CV |
| 14 | Depth 4→3, LR 0.05→0.03, L2 regularisation, weighted CV/fit | Gap −38% |

---

## 📚 Data Sources

- Historical match results and ELO ratings: [Kaggle — International Football Results](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
- ELO ratings: [eloratings.net](https://www.eloratings.net)
- Squad market values: [Transfermarkt](https://www.transfermarkt.com)
- 2026 fixture list and group assignments: [FIFA](https://www.fifa.com/fifaplus/en/tournaments/mens/worldcup/canadamexicousa2026)
- Last-10 match stats, xG data: manually compiled from public sources

---

## 🧰 Tech Stack

| Tool | Use |
|---|---|
| Python 3.9+ | Core language |
| pandas / numpy | Data manipulation |
| scikit-learn | ML models, CV, preprocessing |
| XGBoost / LightGBM | Gradient boosting (optional, falls back to sklearn) |
| matplotlib / seaborn | Visualisations |
| Jupyter Notebook | Development environment |

---

## 📝 Licence

You are free to use, modify, and distribute it for personal or educational purposes, but not pose as your own.

---

*Built for fun and learning. Not affiliated with FIFA, any football association, or any betting platform.*
