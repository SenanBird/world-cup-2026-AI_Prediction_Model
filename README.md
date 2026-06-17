# 🏆 2026 FIFA World Cup Prediction & Simulation Pipeline

A high-performance machine learning and data simulation engine designed to project group-stage outcomes for the 2026 FIFA World Cup. The system combines a **PyTorch deep learning brain** for match-level xG (expected goals) prediction with a **C++ Monte Carlo simulation engine** that performs 1,000,000 tournament iterations to deliver granular placement probabilities.

---

## 🧠 System Architecture

The pipeline operates in two modes to provide flexibility throughout the tournament:

1.  **Pre-Tournament Prediction**: Forecasts all 48 group-stage matches based on historical data.
2.  **Mid-Tournament Hybrid Update**: Injects actual match results as they happen, then re-predicts remaining fixtures to maintain live, accurate standings.

### AI Brain (`/ai_brain`)
* **Neural Network Architecture**: 
    * **Transformer Encoder**: A 2-layer transformer processing the last 10 matches to capture recent form, including goals, opponent Elo, and home/away dynamics.
    * **Static Feature Branch**: A feed-forward network ingesting FIFA rankings, current Elo differences, and proprietary squad valuation metrics.
    * **Poisson Output**: Dual Softplus units predicting home/away xG.
* **Loss Function**: Custom weighted Poisson loss, minimizing `(xG – actual_goals * log(xG))` with temporal decay to prioritize recent form.

### Simulation Engine (`/sim_engine`)
* **C++ Monte Carlo Engine**: 
    * Reads xG parameters (or live hybrid data).
    * Computes exact scoreline probabilities using the Poisson distribution.
    * Performs **1,000,000 independent tournament simulations** per group to output stable, statistical probabilities for final group standings (1st, 2nd, 3rd, 4th).

---

## 📂 Repository Layout

```text
world-cup-2026-simulator/
├── ai_brain/
│   ├── Ultra_Prediction.py              # Pre-tournament full prediction script
│   └── predict_round_2.py               # Mid-tournament hybrid update script
├── sim_engine/
│   └── simulate_mid_tournament.cpp      # C++ Monte Carlo simulation engine
├── Training Data/
│   ├── results.csv                      # Historical international matches
│   ├── players.csv                      # Transfermarkt player registry
│   ├── player_valuations.csv            # Historical market values
│   ├── fifa_ranking.csv                 # FIFA ranking history
│   └── update_datasets.py               # Automated Kaggle data sync script
└── data/
    ├── mid_tournament_predictions.csv   # Hybrid match xG predictions
    ├── simulation_matrices_mid_tournament.json # C++ Monte Carlo output
    └── team_features_debug.txt          # Per-team feature audit trail


