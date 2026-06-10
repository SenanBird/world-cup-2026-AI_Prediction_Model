# 🏆 2026 Football World Cup Prediction & Simulation Pipeline

A high-performance machine learning and data simulation engine built to project group-stage advancements and matrix probabilities for the 2026 World Cup. 

This project integrates a **PyTorch Deep Learning Brain** with a **High-Performance C++ Monte Carlo Simulation Engine**.

---

## 🧠 System Architecture Overview

1. **AI Brain (`/ai_brain`)**: A neural network implemented in PyTorch utilizing a Transformer Encoder layer to ingest temporal team forms (historical matches), national squad valuations (scraped Transfermarkt data), and rolling dynamic ELO trends to predict Poisson xG parameters for any match matchup.
2. **Simulation Engine (`/sim_engine`)**: A ultra-fast, multi-threaded C++ engine that ingests predicted match xG parameters, computes exact scoreline probabilities using a Poisson distribution matrix, and performs **1,000,000 tournament simulations** per group to generate precise placement probabilities.

---

## 📂 Repository Layout

```text
world-cup-2026-simulator/
├── ai_brain/
│   └── Ultra_Prediction.py    # PyTorch Model training & prediction
├── sim_engine/
│   └── main.cpp               # 1,000,000 iteration C++ Monte Carlo Engine
├── Training Data/
│   ├── results.csv            # Historical international fixtures
│   ├── players.csv            # Transfermarkt player registries
│   ├── player_valuations.csv  # Historical market values
│   ├── fifa_ranking.csv       # Dynamic FIFA ranking index
│   └── update_datasets.py     # Automated Kaggle data sync script
└── data/
    ├── group_stage_predictions.csv  # Predicted xG intermediate table
    └── simulation_matrices.json     # Final C++ JSON output file
