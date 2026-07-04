#!/usr/bin/env python3
"""
FIFA World Cup 2026 – Group Stage xG Predictor (Symmetric, 7‑feature, Transfermarkt)
====================================================================================
- Same model structure & training as knockout stage (Transformer + embeddings).
- 7 features: Elo, Squad Sum/Median/Var, Count>50M, FIFA, Weighted Margin.
- Mirroring ensures symmetry; asymmetric dropout (Elo drop 30%, others 10%).
- Transfermarkt actual squad values override current estimates.
- Outputs:
  - mid_tournament_predictions.csv (for C++ simulator)
  - prediction_debug.csv (feature diffs + xG + Poisson probabilities)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os, sys, time, pickle, warnings
from collections import defaultdict
from itertools import product
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import multiprocessing as mp
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')
pd.set_option('future.no_silent_downcasting', True)

# =========================================================================
# CONFIGURATION
# =========================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../Training Data")) + "/"
OUTPUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../data")) + "/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FORCE_FRESH = True
MODEL_PATH = f"{OUTPUT_DIR}best_model_group.pt"
DEBUG_FILE = f"{OUTPUT_DIR}team_features_group_debug.txt"
ELO_CACHE_FILE = f"{OUTPUT_DIR}elo_state_cache.pkl"
SQUAD_CACHE_FILE = f"{OUTPUT_DIR}squad_cache.pkl"
TRANSFERMARKT_FILE = f"{DATA_DIR}transfermarkt_squad_values_2026.csv"

HISTORY_LEN = 10
EMBEDDING_DIM = 16
MIN_YEAR = 2000
CURRENT_YEAR = 2026
REF_DATE = pd.Timestamp('2026-06-01')
TEMPORAL_DECAY_LAMBDA = 0.15
EPOCHS = 100
LEARNING_RATE = 0.0005
BATCH_SIZE = 256
N_JOBS = -1
NUM_CORES = mp.cpu_count()
POISSON_SIMULATIONS = 50000      # for debug probabilities
POISSON_RANDOM_SEED = 42

if FORCE_FRESH:
    for f in [MODEL_PATH, DEBUG_FILE]:
        if os.path.exists(f):
            os.remove(f)
    print("Forced fresh training – group stage caches deleted.")

# =========================================================================
# NAME MAP
# =========================================================================
name_map = {
    'USA': 'United States', 'IR Iran': 'Iran', 'Korea Republic': 'South Korea',
    'Congo DR': 'DR Congo', 'Curacao': 'Curaçao', 'Bosnia-Herzegovina': 'Bosnia and Herzegovina',
    'Côte d\'Ivoire': 'Ivory Coast', 'Korea DPR': 'North Korea', 'RCS': 'Czech Republic',
    'Zaire': 'DR Congo', 'Yugoslavia': 'Serbia', 'Netherlands Antilles': 'Curaçao',
    'Türkiye': 'Turkey', 'Korea, South': 'South Korea', 'Cote d\'Ivoire': 'Ivory Coast',
    'Ivory Coast': 'Ivory Coast', 'Iran, Islamic Republic of': 'Iran',
}

# =========================================================================
# TOURNAMENT GROUPS (2026)
# =========================================================================
GROUPS = {
    'A': ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    'B': ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    'C': ["Brazil", "Morocco", "Haiti", "Scotland"],
    'D': ["United States", "Paraguay", "Australia", "Turkey"],
    'E': ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    'F': ["Netherlands", "Japan", "Sweden", "Tunisia"],
    'G': ["Belgium", "Egypt", "Iran", "New Zealand"],
    'H': ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    'I': ["France", "Senegal", "Iraq", "Norway"],
    'J': ["Argentina", "Algeria", "Austria", "Jordan"],
    'K': ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    'L': ["England", "Croatia", "Ghana", "Panama"]
}
team_to_group = {team: grp for grp, teams in GROUPS.items() for team in teams}
ALL_TEAMS = sorted(team_to_group.keys())

# =========================================================================
# 1. LOAD & PREPARE DATA
# =========================================================================
t_start = time.time()
print("Loading data...")
raw_results = pd.read_csv(f"{DATA_DIR}results.csv")
players_df   = pd.read_csv(f"{DATA_DIR}players.csv")
valuations_df = pd.read_csv(f"{DATA_DIR}player_valuations.csv")
fifa_df = pd.read_csv(f"{DATA_DIR}fifa_ranking.csv")

valuations_df['date'] = pd.to_datetime(valuations_df['date'])
raw_results.rename(str.title, axis='columns', inplace=True)
raw_results.rename(columns={'Home_Team': 'Home Team', 'Away_Team': 'Away Team',
                            'Home_Score': 'Home Score', 'Away_Score': 'Away Score'}, inplace=True)
raw_results['Date'] = pd.to_datetime(raw_results['Date'])
raw_results['Home Team'] = raw_results['Home Team'].replace(name_map)
raw_results['Away Team'] = raw_results['Away Team'].replace(name_map)
players_df['country_of_citizenship'] = players_df['country_of_citizenship'].astype(str).str.strip()
players_df['country_of_citizenship'] = players_df['country_of_citizenship'].replace(name_map)

results_df = raw_results[raw_results['Date'].dt.year >= MIN_YEAR].reset_index(drop=True)
results_df = results_df.dropna(subset=['Home Score', 'Away Score'])
results_df['Year'] = results_df['Date'].dt.year
results_df['Weight'] = np.exp(-TEMPORAL_DECAY_LAMBDA * (CURRENT_YEAR - results_df['Year']))
print(f"  Loaded {len(results_df):,} historical matches. ({time.time()-t_start:.1f}s)")

# =========================================================================
# 2. FIFA RANKINGS
# =========================================================================
print("Processing FIFA rankings...")
fifa_df = fifa_df[['rank_date', 'country_full', 'total_points']].copy()
fifa_df['rank_date'] = pd.to_datetime(fifa_df['rank_date'])
fifa_df.rename(columns={'country_full': 'Team', 'total_points': 'fifa_points'}, inplace=True)
fifa_df['Team'] = fifa_df['Team'].replace(name_map)
fifa_df = fifa_df.dropna(subset=['fifa_points'])
fifa_df = fifa_df.sort_values(['Team', 'rank_date']).drop_duplicates(['Team', 'rank_date'], keep='last')
fifa_dict = {}
for team, group in fifa_df.groupby('Team'):
    fifa_dict[team] = group.sort_values('rank_date')

def get_fifa_points(team, match_date):
    if team not in fifa_dict:
        return 1500.0
    prior = fifa_dict[team][fifa_dict[team]['rank_date'] <= match_date]
    if len(prior) == 0:
        return 1500.0
    return prior.iloc[-1]['fifa_points']

# =========================================================================
# 3. MARKET VALUE PREDICTOR
# =========================================================================
print("Training market value predictor...")
value_predictor = None
latest_val_all = (valuations_df[valuations_df['date'] <= REF_DATE]
                  .sort_values('date')
                  .groupby('player_id')
                  .last()
                  .reset_index())
latest_val_clean = latest_val_all[['player_id', 'market_value_in_eur']].copy()
players_for_merge = players_df.copy()
if 'market_value_in_eur' in players_for_merge.columns:
    players_for_merge = players_for_merge.drop(columns=['market_value_in_eur'])
players_with_val = players_for_merge.merge(latest_val_clean, on='player_id', how='left')
has_value = players_with_val['market_value_in_eur'].notna() & (players_with_val['market_value_in_eur'] > 0)
train_data = players_with_val[has_value].copy()

if len(train_data) >= 1000:
    train_data['age'] = (REF_DATE - pd.to_datetime(train_data['date_of_birth'])).dt.days / 365.25
    train_data['position'] = train_data['position'].fillna('Missing')
    le_pos = LabelEncoder()
    train_data['pos_code'] = le_pos.fit_transform(train_data['position'])
    train_data['league_id'] = train_data['current_club_domestic_competition_id'].fillna('Unknown')
    le_league = LabelEncoder()
    train_data['league_code'] = le_league.fit_transform(train_data['league_id'].astype(str))
    train_data['caps'] = train_data['international_caps'].fillna(0).clip(0, 150)
    train_data['height'] = train_data['height_in_cm'].fillna(180).clip(150, 210)
    X_train = train_data[['age', 'pos_code', 'league_code', 'caps', 'height']].fillna(0)
    y_train = np.log1p(train_data['market_value_in_eur'])
    rf_model = RandomForestRegressor(n_estimators=80, max_depth=12, n_jobs=-1, random_state=42)
    rf_model.fit(X_train, y_train)
    value_predictor = {'model': rf_model, 'le_pos': le_pos, 'le_league': le_league}
    print(f"  ✅ Value predictor trained on {len(train_data):,} players")
else:
    print("  ⚠️  Not enough data, predictor disabled")

def batch_predict_market_values(missing_players_df, predictor, ref_date):
    if missing_players_df.empty or predictor is None:
        return np.array([])
    df = missing_players_df.copy()
    df['age'] = (ref_date - pd.to_datetime(df['date_of_birth'])).dt.days / 365.25
    df['position'] = df['position'].fillna('Missing')
    known_pos = set(predictor['le_pos'].classes_)
    df['pos_code'] = df['position'].apply(
        lambda x: predictor['le_pos'].transform(['Missing'])[0] if x not in known_pos
        else predictor['le_pos'].transform([x])[0]
    )
    df['league_id'] = df['current_club_domestic_competition_id'].fillna('Unknown').astype(str)
    known_league = set(predictor['le_league'].classes_)
    df['league_code'] = df['league_id'].apply(
        lambda x: predictor['le_league'].transform(['Unknown'])[0] if x not in known_league
        else predictor['le_league'].transform([x])[0]
    )
    df['caps'] = df['international_caps'].fillna(0).clip(0, 150)
    df['height'] = df['height_in_cm'].fillna(180).clip(150, 210)
    X = df[['age', 'pos_code', 'league_code', 'caps', 'height']].fillna(0)
    return np.expm1(predictor['model'].predict(X))

# =========================================================================
# 4. SQUAD VALUATIONS (with disk caching, parallel build if needed)
# =========================================================================
print("Preparing squad data...")
valuations_with_country = valuations_df.merge(
    players_df[['player_id', 'country_of_citizenship']], on='player_id', how='left'
).dropna(subset=['country_of_citizenship'])
valuations_with_country['date'] = pd.to_datetime(valuations_with_country['date'])
valuations_with_country['year'] = valuations_with_country['date'].dt.year

def squad_for_country_up_to_year(country, ref_year):
    """Fallback non-parallel version used only if cache missing."""
    country_players = players_df[players_df['country_of_citizenship'] == country]
    if country_players.empty:
        fifa_val = get_fifa_points(country, pd.Timestamp(f'{ref_year}-06-01'))
        estimated_sum = 50.0 + (fifa_val - 1500) * 0.02
        return {'sum': max(30.0, estimated_sum), 'median': max(1.5, estimated_sum/20),
                'var': 0.0, 'max': max(2.0, estimated_sum/20), 'count_above_50M': 0}
    player_ids = country_players['player_id'].unique()
    val_subset = valuations_df[
        (valuations_df['player_id'].isin(player_ids)) &
        (valuations_df['date'] <= pd.Timestamp(f'{ref_year}-06-01'))
    ]
    latest_vals = (val_subset.sort_values('date').groupby('player_id').last().reset_index()
                   if not val_subset.empty else pd.DataFrame(columns=['player_id','market_value_in_eur']))
    squad_df = country_players[['player_id']].merge(latest_vals, on='player_id', how='left')
    squad_df['market_value_in_eur'] = pd.to_numeric(squad_df['market_value_in_eur'], errors='coerce')
    market_vals = squad_df['market_value_in_eur'].fillna(0.0) / 1_000_000.0
    if (market_vals <= 0).any() and value_predictor is not None:
        missing_mask = market_vals <= 0
        missing_players = country_players[country_players['player_id'].isin(
            squad_df.loc[missing_mask, 'player_id'])]
        if not missing_players.empty:
            predicted = batch_predict_market_values(missing_players, value_predictor,
                                                    pd.Timestamp(f'{ref_year}-06-01'))
            pred_map = dict(zip(missing_players['player_id'], predicted / 1_000_000.0))
            for pid, val in pred_map.items():
                idx = squad_df[squad_df['player_id'] == pid].index
                if len(idx) > 0:
                    market_vals.loc[idx[0]] = val
    known_vals = market_vals[market_vals > 0]
    if len(known_vals) == 0:
        fifa_val = get_fifa_points(country, pd.Timestamp(f'{ref_year}-06-01'))
        estimated_sum = 50.0 + (fifa_val - 1500) * 0.02
        return {'sum': max(30.0, estimated_sum), 'median': max(1.5, estimated_sum/20),
                'var': 0.0, 'max': max(2.0, estimated_sum/20), 'count_above_50M': 0}
    if (market_vals <= 0).any():
        market_vals = market_vals.where(market_vals > 0, known_vals.median())
    top23 = market_vals.nlargest(23)
    if len(top23) < 23:
        pad_val = known_vals.median() if len(known_vals) > 0 else 0.5
        top23 = pd.concat([top23, pd.Series([pad_val] * (23 - len(top23)))])
    return {'sum': float(top23.sum()), 'median': float(top23.median()),
            'var': float(top23.var()) if len(top23) > 1 else 0.0,
            'max': float(top23.max()), 'count_above_50M': int((top23 > 50).sum())}

def squad_for_country_fast(country, ref_year):
    """Fast version using pre-filtered team_valuations for parallel build."""
    country_players = players_df[players_df['country_of_citizenship'] == country]
    if country_players.empty:
        fifa_val = get_fifa_points(country, pd.Timestamp(f'{ref_year}-06-01'))
        estimated_sum = 50.0 + (fifa_val - 1500) * 0.02
        return {'sum': max(30.0, estimated_sum), 'median': max(1.5, estimated_sum/20),
                'var': 0.0, 'max': max(2.0, estimated_sum/20), 'count_above_50M': 0}
    val_subset = team_valuations[country][
        team_valuations[country]['date'] <= pd.Timestamp(f'{ref_year}-06-01')
    ]
    latest_vals = (val_subset.sort_values('date').groupby('player_id').last().reset_index()
                   if not val_subset.empty else pd.DataFrame(columns=['player_id','market_value_in_eur']))
    squad_df = country_players[['player_id']].merge(latest_vals, on='player_id', how='left')
    squad_df['market_value_in_eur'] = pd.to_numeric(squad_df['market_value_in_eur'], errors='coerce')
    market_vals = squad_df['market_value_in_eur'].fillna(0.0) / 1_000_000.0
    if (market_vals <= 0).any() and value_predictor is not None:
        missing_mask = market_vals <= 0
        missing_players = country_players[country_players['player_id'].isin(
            squad_df.loc[missing_mask, 'player_id'])]
        if not missing_players.empty:
            predicted = batch_predict_market_values(missing_players, value_predictor,
                                                    pd.Timestamp(f'{ref_year}-06-01'))
            pred_map = dict(zip(missing_players['player_id'], predicted / 1_000_000.0))
            for pid, val in pred_map.items():
                idx = squad_df[squad_df['player_id'] == pid].index
                if len(idx) > 0:
                    market_vals.loc[idx[0]] = val
    known_vals = market_vals[market_vals > 0]
    if len(known_vals) == 0:
        fifa_val = get_fifa_points(country, pd.Timestamp(f'{ref_year}-06-01'))
        estimated_sum = 50.0 + (fifa_val - 1500) * 0.02
        return {'sum': max(30.0, estimated_sum), 'median': max(1.5, estimated_sum/20),
                'var': 0.0, 'max': max(2.0, estimated_sum/20), 'count_above_50M': 0}
    if (market_vals <= 0).any():
        market_vals = market_vals.where(market_vals > 0, known_vals.median())
    top23 = market_vals.nlargest(23)
    if len(top23) < 23:
        pad_val = known_vals.median() if len(known_vals) > 0 else 0.5
        top23 = pd.concat([top23, pd.Series([pad_val] * (23 - len(top23)))])
    return {'sum': float(top23.sum()), 'median': float(top23.median()),
            'var': float(top23.var()) if len(top23) > 1 else 0.0,
            'max': float(top23.max()), 'count_above_50M': int((top23 > 50).sum())}

if os.path.exists(SQUAD_CACHE_FILE):
    print("Loading squad cache from disk...")
    with open(SQUAD_CACHE_FILE, 'rb') as f:
        squad_cache = pickle.load(f)
    print(f"  Loaded {len(squad_cache)} cached squad entries.")
else:
    print("Precomputing squad values for all (team, year) pairs in parallel...")
    unique_teams = sorted(set(results_df['Home Team'].unique()) | set(results_df['Away Team'].unique()))
    unique_years = sorted(results_df['Year'].unique())
    pairs = list(product(unique_teams, unique_years))

    print("  Pre-grouping player data by team...")
    team_player_ids = {team: players_df[players_df['country_of_citizenship'] == team]['player_id'].unique()
                       if not players_df[players_df['country_of_citizenship'] == team].empty
                       else np.array([]) for team in unique_teams}

    print("  Pre-filtering valuations...")
    team_valuations = {}
    for team, pids in team_player_ids.items():
        team_valuations[team] = valuations_df[valuations_df['player_id'].isin(pids)].copy() if len(pids) > 0 \
            else pd.DataFrame(columns=valuations_df.columns)

    t0 = time.time()
    results = Parallel(n_jobs=12, backend='loky', verbose=10, batch_size=50)(
        delayed(squad_for_country_fast)(team, year) for team, year in pairs
    )
    squad_cache = {p: r for p, r in zip(pairs, results)}
    with open(SQUAD_CACHE_FILE, 'wb') as f:
        pickle.dump(squad_cache, f)
    print(f"\n  Squad cache built in {time.time()-t0:.1f}s and saved to {SQUAD_CACHE_FILE}")

current_squad = {team: squad_cache.get((team, CURRENT_YEAR), squad_for_country_up_to_year(team, CURRENT_YEAR)) for team in ALL_TEAMS}
print("  Current squad data prepared.")

# -------------------------------------------------------------------------
# OVERRIDE current_squad with Transfermarkt actual values
# -------------------------------------------------------------------------
print("Loading Transfermarkt squad values for 2026...")
if os.path.exists(TRANSFERMARKT_FILE):
    tm_df = pd.read_csv(TRANSFERMARKT_FILE)
    tm_df['Team'] = tm_df['Team'].replace(name_map)
    tm_df['TotalValueM'] = tm_df['TotalValueEUR'] / 1_000_000.0
    tm_df['AvgValueM'] = tm_df['AvgValueEUR'] / 1_000_000.0
    updated = 0
    for _, row in tm_df.iterrows():
        team = row['Team']
        if team in current_squad:
            current_squad[team]['sum'] = row['TotalValueM']
            current_squad[team]['median'] = row['AvgValueM']
            updated += 1
        else:
            print(f"  ⚠️  Team '{team}' not found in current_squad list – skipping")
    print(f"  Overwritten {updated} teams with Transfermarkt values.")
else:
    print(f"  ⚠️  {TRANSFERMARKT_FILE} not found. Using estimated squad values.")

# =========================================================================
# 5. MODEL DEFINITION (static_dim = 7)
# =========================================================================
class PoissonLoss(nn.Module):
    def forward(self, pred, target, weights=None):
        pred = torch.clamp(pred, min=0.1, max=6.0)
        loss = pred - target * torch.log(pred + 1e-8)
        loss = loss.sum(dim=1)
        if weights is not None:
            loss = loss * weights
        return loss.mean()

class OrderInvariantPredictor(nn.Module):
    def __init__(self, num_teams, embed_dim=16, hist_len=10, hist_input_dim=4, static_dim=7):
        super().__init__()
        self.team_embedding = nn.Embedding(num_teams, embed_dim)
        self.emb_dropout = nn.Dropout(0.1)
        self.hist_proj = nn.Linear(hist_input_dim, 32)
        encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, batch_first=True, dropout=0.2)
        self.hist_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.static_branch = nn.Sequential(
            nn.Linear(static_dim, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16), nn.ReLU()
        )
        fusion_dim = embed_dim * 2 + 32 * 2 + 16
        self.final_mlp = nn.Sequential(
            nn.Linear(fusion_dim, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(32, 2), nn.Softplus()
        )

    def forward(self, teamA_id, teamB_id, teamA_seq, teamB_seq, static):
        emb_A = self.emb_dropout(self.team_embedding(teamA_id))
        emb_B = self.emb_dropout(self.team_embedding(teamB_id))
        state_A = self.hist_encoder(self.hist_proj(teamA_seq))[:, -1, :]
        state_B = self.hist_encoder(self.hist_proj(teamB_seq))[:, -1, :]
        static_out = self.static_branch(static)
        combined = torch.cat([emb_A, emb_B, state_A, state_B, static_out], dim=1)
        return self.final_mlp(combined)

# =========================================================================
# 6. WEIGHTED MARGIN HELPER
# =========================================================================
def compute_weighted_margin(team, team_history_local, decay=0.8):
    hist = team_history_local.get(team, [])
    if not hist:
        return 0.0
    recent = hist[-HISTORY_LEN:]
    weights = np.exp(-decay * np.arange(len(recent))[::-1])
    weights /= weights.sum()
    margins = []
    for m in recent:
        margin = (m['goals_for'] - m['goals_against']) * (m['opponent_elo'] / 1500.0)
        margins.append(margin)
    return float(np.average(margins, weights=weights))

# =========================================================================
# 7. TRAIN / LOAD ELO & MATCH DATA (reuse knockout cache if exists)
# =========================================================================
if os.path.exists(ELO_CACHE_FILE):
    print("Loading Elo & match data from cache...")
    with open(ELO_CACHE_FILE, 'rb') as f:
        cache = pickle.load(f)
        elo = defaultdict(lambda: 1500.0, cache['elo'])
        team_history = defaultdict(list, cache['team_history'])
        match_data = cache['match_data']
    print(f"  Loaded {len(match_data):,} matches from cache.")
else:
    print("No Elo cache found. Computing historical Elo from scratch (this may take several minutes)...")
    all_teams_set = sorted(set(results_df['Home Team'].unique()) | set(results_df['Away Team'].unique()))
    team_to_idx_local = {team: i for i, team in enumerate(all_teams_set)}
    for team in team_to_group:
        if team not in team_to_idx_local:
            all_teams_set.append(team)
            team_to_idx_local[team] = len(all_teams_set) - 1
    num_teams_local = len(all_teams_set)

    elo = defaultdict(lambda: 1500.0)
    team_history = defaultdict(list)
    match_data = []
    K = 32
    total = len(results_df)
    start_t = time.time()
    for idx in range(total):
        if idx % 5000 == 0:
            print(f"  Elo: {idx}/{total}...")
        row = results_df.iloc[idx]
        h, a = row['Home Team'], row['Away Team']
        year = row['Year']
        match_date = row['Date']
        h_elo_before = elo[h]
        a_elo_before = elo[a]
        home_fifa = get_fifa_points(h, match_date)
        away_fifa = get_fifa_points(a, match_date)

        def make_seq(team):
            hist = team_history[team][-HISTORY_LEN:] if team in team_history else []
            seq = []
            for m in hist:
                seq.append([m['goals_for'], m['goals_against'], m['opponent_elo'],
                            1.0 if m['was_home'] else 0.0])
            while len(seq) < HISTORY_LEN:
                seq.insert(0, [0.0, 0.0, 1500.0, 0.0])
            return seq[-HISTORY_LEN:]
        h_seq = make_seq(h)
        a_seq = make_seq(a)

        h_squad = squad_cache[(h, year)]
        a_squad = squad_cache[(a, year)]
        h_margin = compute_weighted_margin(h, team_history)
        a_margin = compute_weighted_margin(a, team_history)

        feat = [
            h_elo_before - a_elo_before,
            (h_squad['sum'] - a_squad['sum']) / 100.0,
            h_squad['median'] - a_squad['median'],
            np.log1p(h_squad['var'] + 1) - np.log1p(a_squad['var'] + 1),
            h_squad['count_above_50M'] - a_squad['count_above_50M'],
            home_fifa - away_fifa,
            h_margin - a_margin,
        ]
        match_info = {
            'teamA_seq': np.array(h_seq, dtype=np.float32),
            'teamB_seq': np.array(a_seq, dtype=np.float32),
            'features': np.array(feat, dtype=np.float32),
            'target_A': row['Home Score'],
            'target_B': row['Away Score'],
            'weight': row['Weight'],
            'year': year,
            'teamA_name': h,
            'teamB_name': a,
        }
        match_data.append(match_info)

        h_score, a_score = row['Home Score'], row['Away Score']
        if h_score > a_score: h_res, a_res = 1, 0
        elif h_score < a_score: h_res, a_res = 0, 1
        else: h_res, a_res = 0.5, 0.5
        h_exp = 1 / (1 + 10**((a_elo_before - h_elo_before)/400))
        a_exp = 1 / (1 + 10**((h_elo_before - a_elo_before)/400))
        K_adj = K * (1 + min(abs(h_score-a_score), 4)/10)
        elo[h] += K_adj * (h_res - h_exp)
        elo[a] += K_adj * (a_res - a_exp)
        team_history[h].append({'goals_for': h_score, 'goals_against': a_score,
                                'opponent_elo': a_elo_before, 'was_home': False})
        team_history[a].append({'goals_for': a_score, 'goals_against': h_score,
                                'opponent_elo': h_elo_before, 'was_home': False})
    with open(ELO_CACHE_FILE, 'wb') as f:
        pickle.dump({'elo': dict(elo), 'team_history': dict(team_history),
                     'match_data': match_data, 'last_index': total}, f)
    print(f"  Elo computation finished in {time.time()-start_t:.1f}s and cached.")

# =========================================================================
# 8. BUILD TEAM INDEX FOR MODEL
# =========================================================================
all_teams = sorted(set(results_df['Home Team'].unique()) | set(results_df['Away Team'].unique()))
for team in team_to_group:
    if team not in all_teams:
        all_teams.append(team)
team_to_idx_model = {team: i for i, team in enumerate(all_teams)}
num_teams = len(all_teams)

# =========================================================================
# 9. BUILD MIRRORED DATASET & SCALER
# =========================================================================
print("Building mirrored dataset...")
features_orig, targets_orig, weights_orig, years_orig = [], [], [], []
seqA_orig, seqB_orig, idxA_orig, idxB_orig = [], [], [], []
for m in match_data:
    features_orig.append(m['features'])
    targets_orig.append([m['target_A'], m['target_B']])
    weights_orig.append(m['weight'])
    years_orig.append(m['year'])
    seqA_orig.append(m['teamA_seq'])
    seqB_orig.append(m['teamB_seq'])
    idxA_orig.append(team_to_idx_model[m['teamA_name']])
    idxB_orig.append(team_to_idx_model[m['teamB_name']])

features_mirror, targets_mirror, weights_mirror, years_mirror = [], [], [], []
seqA_mirror, seqB_mirror, idxA_mirror, idxB_mirror = [], [], [], []
for i, m in enumerate(match_data):
    feat = -m['features']
    features_mirror.append(feat)
    targets_mirror.append([m['target_B'], m['target_A']])
    weights_mirror.append(m['weight'])
    years_mirror.append(m['year'])
    seqA_mirror.append(m['teamB_seq'])
    seqB_mirror.append(m['teamA_seq'])
    idxA_mirror.append(team_to_idx_model[m['teamB_name']])
    idxB_mirror.append(team_to_idx_model[m['teamA_name']])

features_all = np.array(features_orig + features_mirror, dtype=np.float32)
targets_all = np.array(targets_orig + targets_mirror, dtype=np.float32)
weights_all = np.array(weights_orig + weights_mirror, dtype=np.float32)
years_all = np.array(years_orig + years_mirror)
seqA_all = np.stack(seqA_orig + seqA_mirror)
seqB_all = np.stack(seqB_orig + seqB_mirror)
idxA_all = np.array(idxA_orig + idxA_mirror)
idxB_all = np.array(idxB_orig + idxB_mirror)

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_all)

train_mask = years_all < 2022
val_mask   = (years_all >= 2022) & (years_all < 2025)
print(f"Training samples: {train_mask.sum()}, Validation samples: {val_mask.sum()}")

# =========================================================================
# 10. TRAINING WITH ASYMMETRIC DROPOUT
# =========================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = OrderInvariantPredictor(num_teams, EMBEDDING_DIM, HISTORY_LEN, 4,
                                static_dim=features_scaled.shape[1]).to(device)
criterion = PoissonLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

def create_dataloader(idA, idB, sA, sB, static, y, w, bs, shuffle=True):
    dataset = TensorDataset(
        torch.tensor(idA, dtype=torch.long), torch.tensor(idB, dtype=torch.long),
        torch.tensor(sA, dtype=torch.float32), torch.tensor(sB, dtype=torch.float32),
        torch.tensor(static, dtype=torch.float32), torch.tensor(y, dtype=torch.float32),
        torch.tensor(w, dtype=torch.float32))
    return DataLoader(dataset, batch_size=bs, shuffle=shuffle, pin_memory=False, num_workers=0)

train_loader = create_dataloader(
    idxA_all[train_mask], idxB_all[train_mask], seqA_all[train_mask], seqB_all[train_mask],
    features_scaled[train_mask], targets_all[train_mask], weights_all[train_mask], BATCH_SIZE, True)
val_loader = create_dataloader(
    idxA_all[val_mask], idxB_all[val_mask], seqA_all[val_mask], seqB_all[val_mask],
    features_scaled[val_mask], targets_all[val_mask], weights_all[val_mask], BATCH_SIZE, False)

def feature_mask_batch_asymmetric(x, mask_probs):
    mask = torch.ones_like(x)
    for i, p in enumerate(mask_probs):
        keep = torch.rand(x.shape[0], device=x.device) > p
        mask[:, i] = keep.float()
    return x * mask

mask_probs = [0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

best_val_loss = float('inf')
patience_counter = 0
patience = 15
best_model_state = None
print("Training model with asymmetric dropout (Elo drop 0.3)...")
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        idA, idB, sA, sB, stat, yb, wb = [x.to(device) for x in batch]
        stat = feature_mask_batch_asymmetric(stat, mask_probs)
        optimizer.zero_grad()
        pred = model(idA, idB, sA, sB, stat)
        loss = criterion(pred, yb, wb)
        if torch.isnan(loss) or torch.isinf(loss): continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            idA, idB, sA, sB, stat, yb, wb = [x.to(device) for x in batch]
            val_loss += criterion(model(idA, idB, sA, sB, stat), yb, wb).item()
    val_loss /= len(val_loader)
    train_loss /= len(train_loader)
    if epoch % 10 == 0:
        print(f"  Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict().copy()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    scheduler.step(val_loss)

model.load_state_dict(best_model_state)
torch.save(best_model_state, MODEL_PATH)
print(f"  Training complete. Best val loss: {best_val_loss:.4f}")

# Feature importance (same as before)
feature_names = [
    "Elo Diff", "Squad Sum Diff", "Squad Median Diff", "Squad Var Diff",
    "Count >50M Diff", "FIFA Diff", "Weighted Margin Diff"
]
model.eval()
with torch.no_grad():
    base_loss = 0.0
    for batch in val_loader:
        idA, idB, sA, sB, stat, yb, wb = [x.to(device) for x in batch]
        base_loss += criterion(model(idA, idB, sA, sB, stat), yb, wb).item()
    base_loss /= len(val_loader)

importances = {}
static_val = features_scaled[val_mask].copy()
for i, fname in enumerate(feature_names):
    static_shuf = static_val.copy()
    np.random.shuffle(static_shuf[:, i])
    temp_dataset = TensorDataset(
        torch.tensor(idxA_all[val_mask], dtype=torch.long),
        torch.tensor(idxB_all[val_mask], dtype=torch.long),
        torch.tensor(seqA_all[val_mask], dtype=torch.float32),
        torch.tensor(seqB_all[val_mask], dtype=torch.float32),
        torch.tensor(static_shuf, dtype=torch.float32),
        torch.tensor(targets_all[val_mask], dtype=torch.float32),
        torch.tensor(weights_all[val_mask], dtype=torch.float32))
    temp_loader = DataLoader(temp_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False, num_workers=0)
    perm_loss = 0.0
    with torch.no_grad():
        for batch in temp_loader:
            idA, idB, sA, sB, stat, yb, wb = [x.to(device) for x in batch]
            perm_loss += criterion(model(idA, idB, sA, sB, stat), yb, wb).item()
    perm_loss /= len(temp_loader)
    importances[fname] = max(0.0, perm_loss - base_loss)
    print(f"    {fname:25s} importance: {importances[fname]:.6f}")

imp_file = f"{OUTPUT_DIR}feature_importance_group.txt"
with open(imp_file, 'w') as f:
    f.write("============================================================\n")
    f.write("FEATURE IMPORTANCE (Group Stage Model)\n")
    f.write("============================================================\n")
    f.write(f"Baseline validation loss: {base_loss:.6f}\n\n")
    for fname, imp in sorted(importances.items(), key=lambda x: x[1]):
        f.write(f"{fname:30s} {imp:.6f}\n")
print(f"  Feature importance saved to {imp_file}")

# =========================================================================
# 11. DEBUG OUTPUT (ELO & SQUAD)
# =========================================================================
with open(DEBUG_FILE, 'w') as f:
    f.write("ELO RATINGS (as of 2026-06-01)\n")
    for team in sorted(team_to_group.keys()):
        f.write(f"{team}: {elo.get(team, 1500.0):.2f}\n")
    f.write("\nCURRENT SQUAD SUMS & MEDIANS (M€)\n")
    for team in sorted(team_to_group.keys()):
        s = current_squad[team]
        f.write(f"{team}: Sum = {s['sum']:.2f} | Median = {s['median']:.2f}\n")
print("Debug file generated.")

# =========================================================================
# 12. GENERATE GROUP STAGE PREDICTIONS + DEBUG CSV
# =========================================================================
print("\nGenerating group stage xG predictions...")

def encode_history_batch(teams):
    seqs = []
    for team in teams:
        hist = team_history.get(team, [])
        seq = []
        for m in hist[-HISTORY_LEN:]:
            seq.append([m['goals_for'], m['goals_against'], m['opponent_elo'], 1.0 if m['was_home'] else 0.0])
        while len(seq) < HISTORY_LEN:
            seq.insert(0, [0.0, 0.0, 1500.0, 0.0])
        seqs.append(seq[-HISTORY_LEN:])
    return np.array(seqs, dtype=np.float32)

def build_features_batch(team_a_list, team_b_list):
    feats = []
    for team_a, team_b in zip(team_a_list, team_b_list):
        squad_a = current_squad[team_a]; squad_b = current_squad[team_b]
        fifa_a = get_fifa_points(team_a, REF_DATE); fifa_b = get_fifa_points(team_b, REF_DATE)
        margin_a = compute_weighted_margin(team_a, team_history)
        margin_b = compute_weighted_margin(team_b, team_history)
        feat = [
            elo[team_a] - elo[team_b],
            (squad_a['sum'] - squad_b['sum']) / 100.0,
            squad_a['median'] - squad_b['median'],
            np.log1p(squad_a['var']+1) - np.log1p(squad_b['var']+1),
            squad_a['count_above_50M'] - squad_b['count_above_50M'],
            fifa_a - fifa_b,
            margin_a - margin_b,
        ]
        feats.append(feat)
    return np.array(feats, dtype=np.float32)

# Determine already played group stage matches
wc_2026_mask = (raw_results['Tournament'].str.contains('World Cup', case=False, na=False)) & \
               (~raw_results['Tournament'].str.contains('qualification', case=False, na=False)) & \
               (raw_results['Date'].dt.year == 2026)
wc_fixtures = raw_results[wc_2026_mask].copy()
played_scores = {}
for _, row in wc_fixtures.iterrows():
    h, a = row['Home Team'], row['Away Team']
    if pd.notna(row['Home Score']) and pd.notna(row['Away Score']):
        if h in team_to_group and a in team_to_group and team_to_group[h] == team_to_group[a]:
            played_scores[(h, a)] = (int(row['Home Score']), int(row['Away Score']))

# Build fixture list: all 6 pairs per group
fixtures = []
for grp, teams in GROUPS.items():
    for i in range(4):
        for j in range(i+1, 4):
            t1, t2 = teams[i], teams[j]
            fixtures.append((grp, t1, t2))

predictions = []
debug_rows = []
rng_debug = np.random.RandomState(POISSON_RANDOM_SEED)

model.eval()
with torch.no_grad():
    for grp, team_a, team_b in fixtures:
        # Features from A vs B perspective (team_a = home)
        feat_AB = build_features_batch([team_a], [team_b])[0]  # raw diff array

        # Check if played
        if (team_a, team_b) in played_scores:
            hg, ag = played_scores[(team_a, team_b)]
            home_xg, away_xg = hg, ag
            is_played = 1
            p_home, p_draw, p_away = (1.0, 0.0, 0.0) if hg > ag else ((0.0, 0.0, 1.0) if ag > hg else (0.0, 1.0, 0.0))
        elif (team_b, team_a) in played_scores:
            hg, ag = played_scores[(team_b, team_a)]
            home_xg, away_xg = ag, hg
            is_played = 1
            p_home, p_draw, p_away = (1.0, 0.0, 0.0) if ag > hg else ((0.0, 0.0, 1.0) if hg > ag else (0.0, 1.0, 0.0))
        else:
            # Symmetric xG via two-pass averaging
            feat_BA = build_features_batch([team_b], [team_a])
            feat_AB_scaled = scaler.transform(feat_AB.reshape(1, -1))
            feat_BA_scaled = scaler.transform(feat_BA.reshape(1, -1))
            seqA_AB = encode_history_batch([team_a])
            seqB_AB = encode_history_batch([team_b])
            seqA_BA = encode_history_batch([team_b])
            seqB_BA = encode_history_batch([team_a])

            idA_AB = torch.tensor([team_to_idx_model[team_a]], dtype=torch.long, device=device)
            idB_AB = torch.tensor([team_to_idx_model[team_b]], dtype=torch.long, device=device)
            idA_BA = torch.tensor([team_to_idx_model[team_b]], dtype=torch.long, device=device)
            idB_BA = torch.tensor([team_to_idx_model[team_a]], dtype=torch.long, device=device)

            pred_AB = model(idA_AB, idB_AB,
                            torch.tensor(seqA_AB, dtype=torch.float32, device=device),
                            torch.tensor(seqB_AB, dtype=torch.float32, device=device),
                            torch.tensor(feat_AB_scaled, dtype=torch.float32, device=device)).cpu().numpy()[0]
            pred_BA = model(idA_BA, idB_BA,
                            torch.tensor(seqA_BA, dtype=torch.float32, device=device),
                            torch.tensor(seqB_BA, dtype=torch.float32, device=device),
                            torch.tensor(feat_BA_scaled, dtype=torch.float32, device=device)).cpu().numpy()[0]
            home_xg = (pred_AB[0] + pred_BA[1]) / 2.0
            away_xg = (pred_AB[1] + pred_BA[0]) / 2.0
            is_played = 0

            # Poisson debug probabilities
            goalsA = rng_debug.poisson(home_xg, POISSON_SIMULATIONS)
            goalsB = rng_debug.poisson(away_xg, POISSON_SIMULATIONS)
            home_win = np.sum(goalsA > goalsB)
            draw = np.sum(goalsA == goalsB)
            away_win = np.sum(goalsA < goalsB)
            total_sim = POISSON_SIMULATIONS
            p_home = home_win / total_sim
            p_draw = draw / total_sim
            p_away = away_win / total_sim

        # Append to mid_tournament CSV
        predictions.append({
            'Group': grp,
            'Home_Team': team_a,
            'Away_Team': team_b,
            'Home_xG': round(home_xg, 2),
            'Away_xG': round(away_xg, 2),
            'Is_Played': is_played
        })

        # Append to debug CSV (features are team_a minus team_b)
        debug_rows.append({
            'Team_A': team_a,
            'Team_B': team_b,
            'Elo_Diff': feat_AB[0],
            'Squad_Sum_Diff': feat_AB[1],
            'Squad_Median_Diff': feat_AB[2],
            'Squad_Var_Diff': feat_AB[3],
            'Count_50M_Diff': feat_AB[4],
            'FIFA_Diff': feat_AB[5],
            'Margin_Diff': feat_AB[6],
            'xG_A_final': float(home_xg),
            'xG_B_final': float(away_xg),
            'p_Home_Win': p_home,
            'p_Draw': p_draw,
            'p_Away_Win': p_away,
        })

# Save main predictions
out_df = pd.DataFrame(predictions)
out_csv = f"{OUTPUT_DIR}mid_tournament_predictions.csv"
out_df.to_csv(out_csv, index=False)
print(f"✅ Group stage predictions saved -> {out_csv}")

# Save debug CSV
debug_df = pd.DataFrame(debug_rows)
debug_csv = f"{OUTPUT_DIR}prediction_debug.csv"
debug_df.to_csv(debug_csv, index=False)
print(f"✅ Prediction debug CSV saved -> {debug_csv}")

print(f"Total runtime: {time.time()-t_start:.1f}s")