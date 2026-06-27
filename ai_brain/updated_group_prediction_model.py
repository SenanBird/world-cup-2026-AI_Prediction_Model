import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import os
import warnings
from collections import defaultdict
import gc
import pickle
import multiprocessing as mp
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../Training Data")) + "/"
OUTPUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../data")) + "/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

SQUAD_CACHE = f"{OUTPUT_DIR}squad_cache.pkl"
ELO_CACHE = f"{OUTPUT_DIR}elo_cache.pkl"

# Force fresh recomputation
FORCE_FRESH = True
if FORCE_FRESH:
    for cache_file in [SQUAD_CACHE, ELO_CACHE]:
        if os.path.exists(cache_file):
            os.remove(cache_file)
    model_file = f"{OUTPUT_DIR}best_model.pt"
    if os.path.exists(model_file):
        os.remove(model_file)

EPOCHS = 100
LEARNING_RATE = 0.0005
BATCH_SIZE = 256
EMBEDDING_DIM = 16          # team embedding size
HISTORY_LEN = 10
TEMPORAL_DECAY_LAMBDA = 0.1
MIN_YEAR = 2000
TEST_START_YEAR = 2023
VAL_START_YEAR = 2019

N_JOBS = -1
NUM_CORES = mp.cpu_count()

print(f"System: {NUM_CORES} CPU cores | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# TOURNAMENT GROUP MAPPINGS
# ==========================================
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

# ==========================================
# LOAD & FILTER DATA
# ==========================================
print("\n[1/5] Loading data...")
raw_results = pd.read_csv(f"{DATA_DIR}results.csv")
players_df = pd.read_csv(f"{DATA_DIR}players.csv")
valuations_df = pd.read_csv(f"{DATA_DIR}player_valuations.csv")
fifa_df = pd.read_csv(f"{DATA_DIR}fifa_ranking.csv")

# ⚠️ Critical fix: convert the date column to datetime immediately
valuations_df['date'] = pd.to_datetime(valuations_df['date'])

raw_results.rename(str.title, axis='columns', inplace=True)
raw_results.rename(columns={'Home_Team': 'Home Team', 'Away_Team': 'Away Team',
                            'Home_Score': 'Home Score', 'Away_Score': 'Away Score'}, inplace=True)
raw_results['Date'] = pd.to_datetime(raw_results['Date'])

# Extended name map that includes the fixes from the diagnostic
name_map = {
    'USA': 'United States', 'IR Iran': 'Iran', 'Korea Republic': 'South Korea',
    'Congo DR': 'DR Congo', 'Curacao': 'Curaçao', 'Bosnia-Herzegovina': 'Bosnia and Herzegovina',
    'Côte d\'Ivoire': 'Ivory Coast', 'Korea DPR': 'North Korea', 'RCS': 'Czech Republic',
    'Zaire': 'DR Congo', 'Yugoslavia': 'Serbia', 'Netherlands Antilles': 'Curaçao',
    'Türkiye': 'Turkey', 'Korea, South': 'South Korea', 'Cote d\'Ivoire': 'Ivory Coast',
    'Ivory Coast': 'Ivory Coast', 'Iran, Islamic Republic of': 'Iran',
}
raw_results['Home Team'] = raw_results['Home Team'].replace(name_map)
raw_results['Away Team'] = raw_results['Away Team'].replace(name_map)

# Also apply to players_df for consistent country names in squad processing
players_df['country_of_citizenship'] = players_df['country_of_citizenship'].astype(str).str.strip()
players_df['country_of_citizenship'] = players_df['country_of_citizenship'].replace(name_map)

wc_2026_mask = (raw_results['Tournament'].str.contains('World Cup', case=False, na=False)) & \
               (~raw_results['Tournament'].str.contains('qualification', case=False, na=False)) & \
               (raw_results['Date'].dt.year == 2026)
wc_fixtures = raw_results[wc_2026_mask].copy()

results_df = raw_results[raw_results['Date'].dt.year >= MIN_YEAR].reset_index(drop=True)
results_df = results_df.dropna(subset=['Home Score', 'Away Score'])
current_year = 2026
results_df['Year'] = results_df['Date'].dt.year
results_df['Weight'] = np.exp(-TEMPORAL_DECAY_LAMBDA * (current_year - results_df['Year']))
print(f"   Loaded {len(results_df):,} historical matches for training.")

# ==========================================
# PREPROCESS FIFA RANKINGS
# ==========================================
print("[2/5] Processing FIFA rankings...")
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
    if team not in fifa_dict: return 1500.0
    prior = fifa_dict[team][fifa_dict[team]['rank_date'] <= match_date]
    if len(prior) == 0: return 1500.0
    return prior.iloc[-1]['fifa_points']

# ==========================================
# BUILD VALUE PREDICTOR (RandomForest)
# ==========================================
print("[3/5] Building market value predictor for missing players...")
value_predictor = None

latest_val_all = (valuations_df[valuations_df['date'] <= pd.Timestamp('2026-06-01')]
                  .sort_values('date')
                  .groupby('player_id')
                  .last()
                  .reset_index())
latest_val_clean = latest_val_all[['player_id', 'market_value_in_eur']].copy()

# Drop players_df's own market_value_in_eur column to avoid conflicts
players_for_merge = players_df.copy()
if 'market_value_in_eur' in players_for_merge.columns:
    players_for_merge = players_for_merge.drop(columns=['market_value_in_eur'])

players_with_val = players_for_merge.merge(latest_val_clean, on='player_id', how='left')
has_value = players_with_val['market_value_in_eur'].notna() & (players_with_val['market_value_in_eur'] > 0)
train_data = players_with_val[has_value].copy()

if len(train_data) >= 1000:
    ref_date = pd.Timestamp('2026-06-01')
    train_data['age'] = (ref_date - pd.to_datetime(train_data['date_of_birth'])).dt.days / 365.25
    train_data['position'] = train_data['position'].fillna('Missing')
    le_pos = LabelEncoder()
    train_data['pos_code'] = le_pos.fit_transform(train_data['position'])
    train_data['league_id'] = train_data['current_club_domestic_competition_id'].fillna('Unknown')
    le_league = LabelEncoder()
    train_data['league_code'] = le_league.fit_transform(train_data['league_id'].astype(str))
    train_data['caps'] = train_data['international_caps'].fillna(0).clip(0, 150)
    train_data['height'] = train_data['height_in_cm'].fillna(180).clip(150, 210)

    features = ['age', 'pos_code', 'league_code', 'caps', 'height']
    X_train = train_data[features].fillna(0)
    y_train = np.log1p(train_data['market_value_in_eur'])

    rf_model = RandomForestRegressor(n_estimators=80, max_depth=12, n_jobs=-1, random_state=42)
    rf_model.fit(X_train, y_train)
    value_predictor = {
        'model': rf_model,
        'le_pos': le_pos,
        'le_league': le_league
    }
    print(f"   ✅ Value predictor trained on {len(train_data):,} players")
else:
    print("   ⚠️  Not enough training data, predictor disabled")


def batch_predict_market_values(missing_players_df, predictor, ref_date):
    if missing_players_df.empty or predictor is None:
        return np.array([])
    df = missing_players_df.copy()
    df['age'] = (ref_date - pd.to_datetime(df['date_of_birth'])).dt.days / 365.25
    df['position'] = df['position'].fillna('Missing')
    # Safe transform for unseen classes
    known_pos = set(predictor['le_pos'].classes_)
    df['pos_code'] = df['position'].apply(
        lambda x: predictor['le_pos'].transform(['Missing'])[0] if x not in known_pos else predictor['le_pos'].transform([x])[0]
    )
    df['league_id'] = df['current_club_domestic_competition_id'].fillna('Unknown').astype(str)
    known_league = set(predictor['le_league'].classes_)
    df['league_code'] = df['league_id'].apply(
        lambda x: predictor['le_league'].transform(['Unknown'])[0] if x not in known_league else predictor['le_league'].transform([x])[0]
    )
    df['caps'] = df['international_caps'].fillna(0).clip(0, 150)
    df['height'] = df['height_in_cm'].fillna(180).clip(150, 210)
    X = df[['age', 'pos_code', 'league_code', 'caps', 'height']].fillna(0)
    return np.expm1(predictor['model'].predict(X))

# ==========================================
# SQUAD VALUATIONS – ML‑ENHANCED
# ==========================================
print("[4/5] Preparing squad data (ML‑enhanced)...")

valuations_with_country = valuations_df.merge(
    players_df[['player_id', 'country_of_citizenship']], on='player_id', how='left'
)
valuations_with_country = valuations_with_country.dropna(subset=['country_of_citizenship'])
# Already converted to datetime, but this line is harmless and kept for safety
valuations_with_country['date'] = pd.to_datetime(valuations_with_country['date'])
valuations_with_country['year'] = valuations_with_country['date'].dt.year

def squad_for_country_up_to_year(country, ref_year):
    """Build squad using all players of that nation, imputing missing values."""
    # Get all players of this country from players_df
    country_players = players_df[players_df['country_of_citizenship'] == country]
    if country_players.empty:
        # No players at all – FIFA estimate
        fifa_val = get_fifa_points(country, pd.Timestamp(f'{ref_year}-06-01'))
        estimated_sum = 50.0 + (fifa_val - 1500) * 0.02
        return {
            'sum': max(30.0, estimated_sum),
            'median': max(1.5, estimated_sum / 20),
            'var': 0.0,
            'max': max(2.0, estimated_sum / 20),
            'count_above_50M': 0
        }

    # Get latest valuations for those players up to ref_year
    player_ids = country_players['player_id'].unique()
    val_subset = valuations_df[
        (valuations_df['player_id'].isin(player_ids)) &
        (valuations_df['date'] <= pd.Timestamp(f'{ref_year}-06-01'))
    ]
    if not val_subset.empty:
        latest_vals = val_subset.sort_values('date').groupby('player_id').last().reset_index()
        latest_vals = latest_vals[['player_id', 'market_value_in_eur']]
    else:
        latest_vals = pd.DataFrame(columns=['player_id', 'market_value_in_eur'])

    # Merge valuations into the player list
    squad_df = country_players[['player_id']].merge(
        latest_vals, on='player_id', how='left'
    )
    # Convert to million euros, fill missing with 0 initially
    market_vals = squad_df['market_value_in_eur'].fillna(0).astype(float) / 1_000_000.0

    # ML imputation for players with missing/zero values
    if (market_vals <= 0).any() and value_predictor is not None:
        missing_mask = market_vals <= 0
        missing_players = country_players[country_players['player_id'].isin(
            squad_df.loc[missing_mask, 'player_id']
        )]
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
        # Still no values – FIFA fallback
        fifa_val = get_fifa_points(country, pd.Timestamp(f'{ref_year}-06-01'))
        estimated_sum = 50.0 + (fifa_val - 1500) * 0.02
        return {
            'sum': max(30.0, estimated_sum),
            'median': max(1.5, estimated_sum / 20),
            'var': 0.0,
            'max': max(2.0, estimated_sum / 20),
            'count_above_50M': 0
        }

    # Impute any remaining zeros with median
    if (market_vals <= 0).any():
        market_vals = market_vals.where(market_vals > 0, known_vals.median())

    top23 = market_vals.nlargest(23)
    if len(top23) < 23:
        pad_val = known_vals.median() if len(known_vals) > 0 else 0.5
        missing = 23 - len(top23)
        top23 = pd.concat([top23, pd.Series([pad_val] * missing)])

    return {
        'sum': float(top23.sum()),
        'median': float(top23.median()),
        'var': float(top23.var()) if len(top23) > 1 else 0.0,
        'max': float(top23.max()),
        'count_above_50M': int((top23 > 50).sum())
    }

squad_cache = {}
def get_squad_historical(team, year):
    key = (team, year)
    if key not in squad_cache:
        res = squad_for_country_up_to_year(team, year)
        # The function now never returns None, but keep fallback for safety
        if res is None:
            res = {'sum': 50.0, 'median': 2.0, 'var': 0.0, 'max': 50.0, 'count_above_50M': 0}
        squad_cache[key] = res
    return squad_cache[key]

current_squad = {}
for team in team_to_group:
    current_squad[team] = squad_for_country_up_to_year(team, current_year)

# ==========================================
# ELO AND FEATURES (symmetric, no home/away)
# ==========================================
all_teams = sorted(set(results_df['Home Team'].unique()) | set(results_df['Away Team'].unique()))
team_to_idx = {team: i for i, team in enumerate(all_teams)}
num_teams = len(all_teams)

missing = [team for team in team_to_group if team not in team_to_idx]
if missing:
    print(f"\n⚠️  WARNING: Adding {missing} with default features.\n")
    for team in missing:
        all_teams.append(team)
        team_to_idx[team] = len(all_teams) - 1
    num_teams = len(all_teams)

print("[DEBUG] Writing Elo & squad facts to team_features_debug.txt")
with open(f"{OUTPUT_DIR}team_features_debug.txt", 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("ELO & SQUAD FACTS FOR ALL WORLD CUP 2026 PARTICIPANTS\n")
    f.write("=" * 80 + "\n\n")

if os.path.exists(ELO_CACHE):
    print("[5/6] Loading cached Elo data...")
    with open(ELO_CACHE, 'rb') as f:
        cached = pickle.load(f)
        elo = defaultdict(lambda: 1500.0, cached['elo_dict'])
        team_history = defaultdict(list, cached['team_history_dict'])
        match_data = cached['match_data']
    print(f"   Loaded {len(match_data):,} pre-processed matches from cache.")
else:
    print("[5/6] Computing Elo ratings & features (sequential)...")
    elo = defaultdict(lambda: 1500.0)
    K = 32
    team_history = defaultdict(list)
    match_data = []

    total_matches = len(results_df)

    for idx in range(total_matches):
        if idx % 5000 == 0:
            print(f"   Processing match {idx}/{total_matches}...")
        row = results_df.iloc[idx]
        h, a = row['Home Team'], row['Away Team']
        year = row['Year']
        weight = row['Weight']
        match_date = row['Date']

        h_elo_before = elo[h]
        a_elo_before = elo[a]

        home_fifa = get_fifa_points(h, match_date)
        away_fifa = get_fifa_points(a, match_date)

        h_hist = team_history[h][-HISTORY_LEN:] if team_history[h] else []
        a_hist = team_history[a][-HISTORY_LEN:] if team_history[a] else []

        def encode_history(hist):
            seq = []
            for m in hist:
                seq.append([m['goals_for'], m['goals_against'], m['opponent_elo'], 1.0 if m['was_home'] else 0.0])
            if len(seq) < HISTORY_LEN:
                pad = [[0.0, 0.0, 1500.0, 0.0]] * (HISTORY_LEN - len(seq))
                seq = pad + seq
            return seq[-HISTORY_LEN:]

        h_seq = encode_history(h_hist)
        a_seq = encode_history(a_hist)

        # Head‑to‑head
        h2h_gd = 0
        h2h_matches = 0
        for prev_idx in range(max(0, idx-200), idx):
            prev_row = results_df.iloc[prev_idx]
            prev_h, prev_a = prev_row['Home Team'], prev_row['Away Team']
            if prev_h == h and prev_a == a:
                gd = prev_row['Home Score'] - prev_row['Away Score']
                h2h_gd += gd
                h2h_matches += 1
            elif prev_h == a and prev_a == h:
                gd = prev_row['Away Score'] - prev_row['Home Score']
                h2h_gd += gd
                h2h_matches += 1
            if h2h_matches >= 5:
                break

        h2h_gd = max(-10, min(10, h2h_gd / max(1, h2h_matches)))

        def get_goal_diff_stats(team):
            gd_list = [m['goals_for'] - m['goals_against'] for m in team_history[team][-10:]] if team_history[team] else [0]
            if not gd_list:
                return 0.0, 0.0
            return float(np.median(gd_list)), float(np.percentile(gd_list, 95))

        h_gd_med, h_gd_p95 = get_goal_diff_stats(h)
        a_gd_med, a_gd_p95 = get_goal_diff_stats(a)

        h_squad = get_squad_historical(h, year)
        a_squad = get_squad_historical(a, year)

        comp = row.get('Tournament', 'Friendly')
        tourney_mult = 2.0 if 'World Cup' in str(comp) else (1.5 if 'qualification' in str(comp).lower() else 1.0)

        # Features – all differences Team A (home) minus Team B (away)
        squad_sum_diff = (h_squad['sum'] - a_squad['sum']) / 100.0
        squad_median_diff = h_squad['median'] - a_squad['median']
        squad_var_diff = np.log1p(h_squad['var'] + 1) - np.log1p(a_squad['var'] + 1)
        squad_max_diff = (h_squad['max'] - a_squad['max']) / 100.0
        count_50M_diff = h_squad['count_above_50M'] - a_squad['count_above_50M']
        gd_median_diff = h_gd_med - a_gd_med
        gd_p95_diff = h_gd_p95 - a_gd_p95
        fifa_diff = home_fifa - away_fifa

        match_info = {
            'teamA_idx': team_to_idx[h],
            'teamB_idx': team_to_idx[a],
            'teamA_seq': np.array(h_seq, dtype=np.float32),
            'teamB_seq': np.array(a_seq, dtype=np.float32),
            'elo_diff': h_elo_before - a_elo_before,
            'squad_sum_diff': squad_sum_diff,
            'squad_median_diff': squad_median_diff,
            'squad_var_diff': squad_var_diff,
            'squad_max_diff': squad_max_diff,
            'count_50M_diff': count_50M_diff,
            'h2h_goal_diff': h2h_gd,
            'gd_median_diff': gd_median_diff,
            'gd_p95_diff': gd_p95_diff,
            'tourney_mult': tourney_mult,
            'fifa_diff': fifa_diff,
            'target_A_goals': row['Home Score'],
            'target_B_goals': row['Away Score'],
            'weight': weight,
            'year': year,
        }
        match_data.append(match_info)

        # Update Elo (neutral)
        h_score, a_score = row['Home Score'], row['Away Score']
        if h_score > a_score:
            h_res, a_res = 1, 0
        elif h_score < a_score:
            h_res, a_res = 0, 1
        else:
            h_res, a_res = 0.5, 0.5
        h_exp = 1 / (1 + 10**((a_elo_before - h_elo_before)/400))
        a_exp = 1 / (1 + 10**((h_elo_before - a_elo_before)/400))
        gd_abs = abs(h_score - a_score)
        K_adj = K * (1 + min(gd_abs, 4)/10)
        elo[h] = elo[h] + K_adj * (h_res - h_exp)
        elo[a] = elo[a] + K_adj * (a_res - a_exp)

        team_history[h].append({'goals_for': h_score, 'goals_against': a_score, 'opponent_elo': a_elo_before, 'was_home': False})
        team_history[a].append({'goals_for': a_score, 'goals_against': h_score, 'opponent_elo': h_elo_before, 'was_home': False})

    with open(ELO_CACHE, 'wb') as f:
        pickle.dump({
            'elo_dict': dict(elo),
            'team_history_dict': {k: list(v) for k, v in team_history.items()},
            'match_data': match_data,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"   Elo ratings calculated fresh and cached ({len(match_data):,} matches).")

# Debug output for Elo and squad sums
with open(f"{OUTPUT_DIR}team_features_debug.txt", 'a', encoding='utf-8') as f:
    f.write("\n" + "=" * 80 + "\n")
    f.write("ELO RATINGS (as of 2026-06-01)\n")
    f.write("=" * 80 + "\n")
    for team in sorted(team_to_group.keys()):
        f.write(f"{team}: {elo.get(team, 1500.0):.2f}\n")
    f.write("\n" + "=" * 80 + "\n")
    f.write("CURRENT SQUAD SUMS & MEDIANS (M€)\n")
    f.write("=" * 80 + "\n")
    for team in sorted(team_to_group.keys()):
        f.write(f"{team}: Sum = {current_squad[team]['sum']:.2f} | Median = {current_squad[team]['median']:.2f}\n")

# ==========================================
# BUILD FEATURE MATRICES (11 static features)
# ==========================================
print("[6/7] Building symmetric feature matrices...")

features_orig = []
targets_orig = []
weights_orig = []
years_orig = []
seqA_orig = []
seqB_orig = []
idxA_orig = []
idxB_orig = []

for m in match_data:
    feat = [
        m['elo_diff'],
        m['squad_sum_diff'],
        m['squad_median_diff'],
        m['squad_var_diff'],
        m['squad_max_diff'],
        m['count_50M_diff'],
        m['h2h_goal_diff'],
        m['gd_median_diff'],
        m['gd_p95_diff'],
        m['tourney_mult'],
        m['fifa_diff']
    ]
    features_orig.append(feat)
    targets_orig.append([m['target_A_goals'], m['target_B_goals']])
    weights_orig.append(m['weight'])
    years_orig.append(m['year'])
    seqA_orig.append(m['teamA_seq'])
    seqB_orig.append(m['teamB_seq'])
    idxA_orig.append(m['teamA_idx'])
    idxB_orig.append(m['teamB_idx'])

# Mirrored matches
features_mirror = []
targets_mirror = []
weights_mirror = []
years_mirror = []
seqA_mirror = []
seqB_mirror = []
idxA_mirror = []
idxB_mirror = []

for i, m in enumerate(match_data):
    feat = [
        -m['elo_diff'],
        -m['squad_sum_diff'],
        -m['squad_median_diff'],
        -m['squad_var_diff'],
        -m['squad_max_diff'],
        -m['count_50M_diff'],
        -m['h2h_goal_diff'],
        -m['gd_median_diff'],
        -m['gd_p95_diff'],
        m['tourney_mult'],           # not mirrored
        -m['fifa_diff']
    ]
    features_mirror.append(feat)
    targets_mirror.append([m['target_B_goals'], m['target_A_goals']])
    weights_mirror.append(m['weight'])
    years_mirror.append(m['year'])
    seqA_mirror.append(m['teamB_seq'])
    seqB_mirror.append(m['teamA_seq'])
    idxA_mirror.append(m['teamB_idx'])
    idxB_mirror.append(m['teamA_idx'])

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

train_mask = years_all < VAL_START_YEAR
val_mask = (years_all >= VAL_START_YEAR) & (years_all < TEST_START_YEAR)
test_mask = years_all >= TEST_START_YEAR

print(f"   Total samples after mirroring: {len(features_all)} (train: {train_mask.sum()}, val: {val_mask.sum()}, test: {test_mask.sum()})")

# ==========================================
# MODEL DEFINITION (Team embeddings + Transformer)
# ==========================================
class PoissonLoss(nn.Module):
    def forward(self, pred, target, weights=None):
        pred = torch.clamp(pred, min=0.1, max=6.0)
        loss = pred - target * torch.log(pred + 1e-8)
        loss = loss.sum(dim=1)
        if weights is not None:
            loss = loss * weights
        return loss.mean()

class SymmetricPredictor(nn.Module):
    def __init__(self, num_teams, embed_dim=16, hist_len=10, hist_input_dim=4, static_dim=11):
        super().__init__()
        self.team_embedding = nn.Embedding(num_teams, embed_dim)
        self.hist_proj = nn.Linear(hist_input_dim, 32)
        encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, batch_first=True, dropout=0.2)
        self.hist_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.static_branch = nn.Sequential(
            nn.Linear(static_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU()
        )

        fusion_dim = embed_dim * 2 + 32 * 2 + 32
        self.final_mlp = nn.Sequential(
            nn.Linear(fusion_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 2), nn.Softplus()
        )

    def forward(self, teamA_id, teamB_id, teamA_seq, teamB_seq, static):
        emb_A = self.team_embedding(teamA_id)
        emb_B = self.team_embedding(teamB_id)

        proj_A = self.hist_proj(teamA_seq)
        state_A = self.hist_encoder(proj_A)[:, -1, :]

        proj_B = self.hist_proj(teamB_seq)
        state_B = self.hist_encoder(proj_B)[:, -1, :]

        static_out = self.static_branch(static)

        combined = torch.cat([emb_A, emb_B, state_A, state_B, static_out], dim=1)
        goals = self.final_mlp(combined)
        return goals

model = SymmetricPredictor(num_teams, EMBEDDING_DIM, HISTORY_LEN, 4, features_scaled.shape[1]).to(device)
criterion = PoissonLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

# ==========================================
# TRAINING
# ==========================================
print("[7/7] Training symmetric model...")

def create_dataloader(idxA, idxB, seqA, seqB, static, y, w, batch_size, shuffle=True):
    dataset = TensorDataset(
        torch.tensor(idxA, dtype=torch.long),
        torch.tensor(idxB, dtype=torch.long),
        torch.tensor(seqA, dtype=torch.float32),
        torch.tensor(seqB, dtype=torch.float32),
        torch.tensor(static, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        torch.tensor(w, dtype=torch.float32)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=False, num_workers=0)

train_loader = create_dataloader(
    idxA_all[train_mask], idxB_all[train_mask],
    seqA_all[train_mask], seqB_all[train_mask],
    features_scaled[train_mask], targets_all[train_mask], weights_all[train_mask],
    BATCH_SIZE, True
)
val_loader = create_dataloader(
    idxA_all[val_mask], idxB_all[val_mask],
    seqA_all[val_mask], seqB_all[val_mask],
    features_scaled[val_mask], targets_all[val_mask], weights_all[val_mask],
    BATCH_SIZE, False
)

best_val_loss = float('inf')
best_state = None

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        idA, idB, sA, sB, stat, yb, wb = [x.to(device) for x in batch]
        optimizer.zero_grad()
        pred = model(idA, idB, sA, sB, stat)
        loss = criterion(pred, yb, wb)
        if torch.isnan(loss) or torch.isinf(loss):
            continue
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
        best_state = model.state_dict().copy()
        torch.save(best_state, f"{OUTPUT_DIR}best_model.pt")

    scheduler.step(val_loss)

model.load_state_dict(best_state)
print(f"  Training complete. Best val loss: {best_val_loss:.4f}")

# ==========================================
# HYBRID MID-TOURNAMENT PREDICTIONS
# ==========================================
print("\nGenerating Hybrid Predictions (Actual Results + Future xG)...")

def build_team_features(team):
    hist = team_history.get(team, [])
    seq = []
    for m in hist[-HISTORY_LEN:]:
        seq.append([m['goals_for'], m['goals_against'], m['opponent_elo'], 1.0 if m['was_home'] else 0.0])
    if len(seq) < HISTORY_LEN:
        seq = [[0.0, 0.0, 1500.0, 0.0]] * (HISTORY_LEN - len(seq)) + seq
    seq = np.array(seq[-HISTORY_LEN:], dtype=np.float32)
    elo_val = elo.get(team, 1500.0)
    squad = current_squad[team]
    pred_date = pd.Timestamp('2026-06-01')
    fifa_pts = get_fifa_points(team, pred_date)
    return elo_val, seq, squad, fifa_pts

team_data = {team: build_team_features(team) for team in team_to_group}

predictions = []
model.eval()
skipped_count = 0

with torch.no_grad():
    for _, row in wc_fixtures.iterrows():
        home = row['Home Team']
        away = row['Away Team']

        if home not in team_to_group or away not in team_to_group:
            skipped_count += 1
            continue
        if team_to_group[home] != team_to_group[away]:
            skipped_count += 1
            continue

        h_score_raw = row['Home Score']
        a_score_raw = row['Away Score']
        group = team_to_group[home]
        is_played = pd.notna(h_score_raw) and pd.notna(a_score_raw)

        if is_played:
            predictions.append({
                'Group': group, 'Home_Team': home, 'Away_Team': away,
                'Home_xG': float(h_score_raw), 'Away_xG': float(a_score_raw), 'Is_Played': 1
            })
        else:
            h_elo, h_seq, h_squad, h_fifa = team_data[home]
            a_elo, a_seq, a_squad, a_fifa = team_data[away]

            feat = np.array([
                h_elo - a_elo,
                (h_squad['sum'] - a_squad['sum']) / 100.0,
                h_squad['median'] - a_squad['median'],
                np.log1p(h_squad['var'] + 1) - np.log1p(a_squad['var'] + 1),
                (h_squad['max'] - a_squad['max']) / 100.0,
                h_squad['count_above_50M'] - a_squad['count_above_50M'],
                0.0,   # h2h goal diff
                0.0,   # gd median diff
                0.0,   # gd p95 diff
                2.0,   # tournament multiplier (World Cup)
                h_fifa - a_fifa
            ], dtype=np.float32)

            feat_scaled = scaler.transform(feat.reshape(1, -1))

            h_seq_t = torch.tensor(h_seq.reshape(1, HISTORY_LEN, 4), device=device)
            a_seq_t = torch.tensor(a_seq.reshape(1, HISTORY_LEN, 4), device=device)
            hid_t = torch.tensor([team_to_idx[home]], device=device)
            aid_t = torch.tensor([team_to_idx[away]], device=device)
            stat_t = torch.tensor(feat_scaled, device=device)

            home_xg, away_xg = model(hid_t, aid_t, h_seq_t, a_seq_t, stat_t).cpu().numpy()[0]

            predictions.append({
                'Group': group, 'Home_Team': home, 'Away_Team': away,
                'Home_xG': round(float(home_xg), 2), 'Away_xG': round(float(away_xg), 2), 'Is_Played': 0
            })

            if home == 'Ecuador' and away == 'Germany':
                print(f"\n[DEBUG] {home} vs {away}")
                print(f"  Elo diff: {h_elo - a_elo:.2f}")
                print(f"  Squad sum diff (scaled): {(h_squad['sum'] - a_squad['sum']) / 100.0:.4f}")
                print(f"  Squad median diff: {h_squad['median'] - a_squad['median']:.4f}")
                print(f"  Predicted xG: {home_xg:.3f} - {away_xg:.3f}")

print(f"   Extracted {len(predictions)} legitimate group stage matches.")
if skipped_count > 0:
    print(f"   Filtered out {skipped_count} irrelevant records.")

out_csv = f"{OUTPUT_DIR}mid_tournament_predictions.csv"
pd.DataFrame(predictions).to_csv(out_csv, index=False)

print("\n[VERIFICATION] Key predictions:")
for pred in predictions:
    if pred['Is_Played'] == 0:
        if pred['Home_Team'] in ['Ecuador', 'Germany'] and pred['Away_Team'] in ['Ecuador', 'Germany']:
            print(f"  {pred['Home_Team']} {pred['Home_xG']} - {pred['Away_xG']} {pred['Away_Team']}")

print(f"\n{'='*50}")
print(f"✅ HYBRID MANIFEST COMPLETE!")
print(f"{'='*50}")
print(f"File saved to: {out_csv}")

# ==========================================
# DIAGNOSTIC: FEATURE IMPORTANCE ANALYSIS
# ==========================================
print("\n[Diagnostic] Calculating Permutation Feature Importance...")

feature_names = [
    "Elo Diff", "Squad Sum Diff", "Squad Median Diff", "Squad Var Diff",
    "Squad Max Diff", "Count >50M Diff", "H2H Goal Diff", "GD Median Diff",
    "GD P95 Diff", "Tournament Mult", "FIFA Diff"
]

model.eval()
baseline_loss = 0.0
with torch.no_grad():
    for batch in val_loader:
        idA, idB, sA, sB, stat, yb, wb = [x.to(device) for x in batch]
        baseline_loss += criterion(model(idA, idB, sA, sB, stat), yb, wb).item()
baseline_loss /= len(val_loader)

importance_scores = []
for i in range(features_scaled.shape[1]):
    shuffled_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            idA, idB, sA, sB, stat, yb, wb = [x.to(device) for x in batch]
            stat_shuffled = stat.clone()
            perm = torch.randperm(stat_shuffled.size(0))
            stat_shuffled[:, i] = stat_shuffled[perm, i]
            shuffled_loss += criterion(model(idA, idB, sA, sB, stat_shuffled), yb, wb).item()
    shuffled_loss /= len(val_loader)
    importance_scores.append(max(0.0, shuffled_loss - baseline_loss))

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance_scores
}).sort_values(by='Importance', ascending=True)

txt_path = f"{OUTPUT_DIR}feature_importance.txt"
with open(txt_path, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("FEATURE IMPORTANCE (Permutation Loss Increase)\n")
    f.write("=" * 60 + "\n")
    f.write(f"Baseline validation loss: {baseline_loss:.6f}\n\n")
    f.write(f"{'Feature':<30} {'Importance':>12}\n")
    f.write("-" * 42 + "\n")
    for _, row in importance_df.iterrows():
        f.write(f"{row['Feature']:<30} {row['Importance']:>12.6f}\n")
    f.write("-" * 42 + "\n")
    f.write("\nSorted by importance (lowest to highest).\n")
print(f"   Feature importance text saved to: {txt_path}")

plt.figure(figsize=(12, 8))
bars = plt.barh(importance_df['Feature'], importance_df['Importance'], color='#1f77b4', edgecolor='#1d3557', height=0.6)
plt.axvline(x=0, color='#6c757d', linestyle='--', alpha=0.5)
plt.title("Feature Importance (Permutation Loss Increase)", fontsize=14, fontweight='black')
plt.xlabel("Loss Increase", fontsize=11)
plt.grid(axis='x', linestyle=':', alpha=0.6)
for bar in bars:
    width = bar.get_width()
    if width > 0.001:
        plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, f"+{width:.4f}", va='center', fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}feature_importance_diagnostics.png", dpi=150)
plt.close()
print(f"   Feature importance chart saved to: {OUTPUT_DIR}feature_importance_diagnostics.png")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()