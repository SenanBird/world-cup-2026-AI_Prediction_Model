import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd
import os
import warnings
from collections import defaultdict
from tqdm import tqdm
import gc
import pickle
from joblib import Parallel, delayed
import multiprocessing as mp

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
# Dynamically find the root '2026' directory relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Points to 'Training Data' folder (handles the space safely)
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../Training Data")) + "/"

# Points to 'data' folder for models and caches
OUTPUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../data")) + "/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

EPOCHS = 80
LEARNING_RATE = 0.0005
BATCH_SIZE = 256
EMBEDDING_DIM = 16
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
# LOAD & FILTER DATA
# ==========================================
print("\n[1/5] Loading data...")
results_df = pd.read_csv(f"{DATA_DIR}results.csv")
players_df = pd.read_csv(f"{DATA_DIR}players.csv")
valuations_df = pd.read_csv(f"{DATA_DIR}player_valuations.csv")
fifa_df = pd.read_csv(f"{DATA_DIR}fifa_ranking.csv")

print(f"   Loaded FIFA rankings: {len(fifa_df)} rows")

# Clean and rename columns
results_df.rename(str.title, axis='columns', inplace=True)
results_df.rename(columns={'Home_Team': 'Home Team', 'Away_Team': 'Away Team', 
                          'Home_Score': 'Home Score', 'Away_Score': 'Away Score'}, inplace=True)
results_df['Date'] = pd.to_datetime(results_df['Date'])

# Team name normalization (unify names across all data sources)
name_map = {
    'USA': 'United States', 'IR Iran': 'Iran', 'Korea Republic': 'South Korea', 
    'Congo DR': 'DR Congo', 'Curacao': 'Curaçao', 'Bosnia-Herzegovina': 'Bosnia and Herzegovina',
    'Côte d\'Ivoire': 'Ivory Coast', 'Korea DPR': 'North Korea', 'RCS': 'Czech Republic',
    'Zaire': 'DR Congo', 'Yugoslavia': 'Serbia', 'Netherlands Antilles': 'Curaçao'
}
results_df['Home Team'] = results_df['Home Team'].replace(name_map)
results_df['Away Team'] = results_df['Away Team'].replace(name_map)

# Filter by year
results_df = results_df[results_df['Date'].dt.year >= MIN_YEAR].reset_index(drop=True)
# Remove rows with missing scores
results_df = results_df.dropna(subset=['Home Score', 'Away Score'])
current_year = 2026
results_df['Year'] = results_df['Date'].dt.year
results_df['Weight'] = np.exp(-TEMPORAL_DECAY_LAMBDA * (current_year - results_df['Year']))
print(f"   Loaded {len(results_df):,} matches ({MIN_YEAR}-{current_year})")

# ==========================================
# PREPROCESS FIFA RANKINGS
# ==========================================
print("[2/5] Processing FIFA rankings...")
# Keep only relevant columns: rank_date, country_full, total_points
fifa_df = fifa_df[['rank_date', 'country_full', 'total_points']].copy()
fifa_df['rank_date'] = pd.to_datetime(fifa_df['rank_date'])
fifa_df.rename(columns={'country_full': 'Team', 'total_points': 'fifa_points'}, inplace=True)
fifa_df['Team'] = fifa_df['Team'].replace(name_map)
# Remove rows with missing points
fifa_df = fifa_df.dropna(subset=['fifa_points'])

# Keep only the latest ranking per team per day (some days may have multiple entries)
fifa_df = fifa_df.sort_values(['Team', 'rank_date']).drop_duplicates(['Team', 'rank_date'], keep='last')

# Build fast lookup dict: team -> sorted DataFrame of rankings
fifa_dict = {}
for team, group in fifa_df.groupby('Team'):
    fifa_dict[team] = group.sort_values('rank_date')
print(f"   Loaded FIFA rankings for {len(fifa_dict)} teams")

def get_fifa_points(team, match_date):
    """Return the most recent FIFA points before match_date, or 1500.0 if missing."""
    if team not in fifa_dict:
        return 1500.0
    prior = fifa_dict[team][fifa_dict[team]['rank_date'] <= match_date]
    if len(prior) == 0:
        return 1500.0
    return prior.iloc[-1]['fifa_points']

# ==========================================
# SQUAD VALUATIONS (OPTIMIZED & CACHED)
# ==========================================
SQUAD_CACHE = f"{OUTPUT_DIR}squad_cache.pkl"

print("[3/5] Pre-processing squad data...")
valuations_with_country = valuations_df.merge(
    players_df[['player_id', 'country_of_citizenship']], 
    on='player_id', 
    how='left'
)
valuations_with_country = valuations_with_country.dropna(subset=['country_of_citizenship'])
valuations_with_country['year'] = pd.to_datetime(valuations_with_country['date']).dt.year
print(f"   Pre-processed {len(valuations_with_country):,} valuation records")

def compute_squad_for_country_year_optimized(country, year, valuations_preprocessed):
    try:
        val_data = valuations_preprocessed[
            (valuations_preprocessed['country_of_citizenship'] == country) & 
            (valuations_preprocessed['year'] == year)
        ]
        if len(val_data) == 0:
            return None
        latest = val_data.sort_values('date').groupby('player_id').last().reset_index()
        market_vals = latest['market_value_in_eur'].fillna(0) / 1_000_000
        top23 = market_vals.nlargest(23)
        if len(top23) == 0:
            return None
        return {
            'country': country,
            'year': year,
            'sum': top23.sum(),
            'var': top23.var() if len(top23) > 1 else 0.0,
            'max': top23.max(),
            'count_above_50M': (top23 > 50).sum()
        }
    except:
        return None

if os.path.exists(SQUAD_CACHE):
    print("[3/5] Loading cached squad data...")
    with open(SQUAD_CACHE, 'rb') as f:
        historical_squad = pickle.load(f)
    print(f"   Loaded {len(historical_squad)} countries from cache")
else:
    print("[3/5] Computing squad valuations (parallel, first time only)...")
    all_countries = valuations_with_country['country_of_citizenship'].unique()
    years_range = range(MIN_YEAR, current_year + 1)
    tasks = [(country, year) for country in all_countries for year in years_range]
    total_tasks = len(tasks)
    print(f"   Total tasks: {total_tasks:,} (countries: {len(all_countries)}, years: {len(years_range)})")
    
    results = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(compute_squad_for_country_year_optimized)(country, year, valuations_with_country)
        for country, year in tasks
    )
    
    historical_squad = defaultdict(dict)
    for res in results:
        if res is not None:
            historical_squad[res['country']][res['year']] = {
                'sum': res['sum'],
                'var': res['var'],
                'max': res['max'],
                'count_above_50M': res['count_above_50M']
            }
    
    with open(SQUAD_CACHE, 'wb') as f:
        pickle.dump(historical_squad, f)
    print(f"   Squad data cached ({len(historical_squad)} countries)")

def get_squad_features(country, year):
    default = {'sum': 50.0, 'var': 0.0, 'max': 50.0, 'count_above_50M': 0}
    try:
        if country in historical_squad:
            if year in historical_squad[country]:
                val = historical_squad[country][year]
                if isinstance(val, dict):
                    return val
            years = sorted([y for y in historical_squad[country].keys() if y <= year])
            if years:
                val = historical_squad[country][max(years)]
                if isinstance(val, dict):
                    return val
    except:
        pass
    return default

# ==========================================
# ELO AND FEATURES (WITH NEUTRAL FLAG & FIFA)
# ==========================================
ELO_CACHE = f"{OUTPUT_DIR}elo_cache.pkl"
all_teams = sorted(set(results_df['Home Team'].unique()) | set(results_df['Away Team'].unique()))
team_to_idx = {team: i for i, team in enumerate(all_teams)}
num_teams = len(all_teams)

if os.path.exists(ELO_CACHE):
    print("[4/5] Loading cached Elo data...")
    with open(ELO_CACHE, 'rb') as f:
        cached = pickle.load(f)
        elo = defaultdict(lambda: 1500.0, cached['elo_dict'])
        team_history = defaultdict(list, cached['team_history_dict'])
        match_data = cached['match_data']
    print(f"   Loaded {len(match_data):,} pre-processed matches")
else:
    print("[4/5] Computing Elo ratings & features (sequential, 5-8 min)...")
    elo = defaultdict(lambda: 1500.0)
    K = 32
    HOME_ADV = 35
    team_history = defaultdict(list)
    match_data = []
    
    total_matches = len(results_df)
    next_print = 0.05
    
    for idx in range(total_matches):
        progress = idx / total_matches
        if progress >= next_print:
            print(f"   Progress: {int(progress*100)}% ({idx}/{total_matches})")
            next_print += 0.05
        
        row = results_df.iloc[idx]
        h, a = row['Home Team'], row['Away Team']
        year = row['Year']
        weight = row['Weight']
        match_date = row['Date']
        
        # Neutral flag
        neutral = row.get('Neutral', False)
        if isinstance(neutral, str):
            neutral = neutral.upper() == 'TRUE'
        
        if neutral:
            h_elo_before = elo[h]
            a_elo_before = elo[a]
        else:
            h_elo_before = elo[h] + HOME_ADV
            a_elo_before = elo[a]
        
        # FIFA points
        home_fifa = get_fifa_points(h, match_date)
        away_fifa = get_fifa_points(a, match_date)
        
        # Build history sequences
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
        
        # Head-to-head stats
        h2h_gd = 0
        h2h_wins_home = 0
        h2h_matches = 0
        for prev_idx in range(max(0, idx-200), idx):
            prev_row = results_df.iloc[prev_idx]
            prev_h = prev_row['Home Team']
            prev_a = prev_row['Away Team']
            if prev_h == h and prev_a == a:
                gd = prev_row['Home Score'] - prev_row['Away Score']
                h2h_gd += gd
                if gd > 0: h2h_wins_home += 1
                h2h_matches += 1
            elif prev_h == a and prev_a == h:
                gd = prev_row['Away Score'] - prev_row['Home Score']
                h2h_gd += gd
                if gd > 0: h2h_wins_home += 1
                h2h_matches += 1
            if h2h_matches >= 5: break
        
        h2h_gd = max(-10, min(10, h2h_gd))
        h2h_wins_home = h2h_wins_home / 5.0 if h2h_matches > 0 else 0.0
        
        # Goal difference stats
        def get_goal_diff_stats(team):
            gd_list = [m['goals_for'] - m['goals_against'] for m in team_history[team][-10:]] if team_history[team] else [0]
            if not gd_list:
                return 0.0, 0.0
            return float(np.median(gd_list)), float(np.percentile(gd_list, 95))
        
        h_gd_med, h_gd_p95 = get_goal_diff_stats(h)
        a_gd_med, a_gd_p95 = get_goal_diff_stats(a)
        
        # Squad features
        h_squad = get_squad_features(h, year)
        a_squad = get_squad_features(a, year)
        if not isinstance(h_squad, dict):
            h_squad = {'sum': 50.0, 'var': 0.0, 'max': 50.0, 'count_above_50M': 0}
        if not isinstance(a_squad, dict):
            a_squad = {'sum': 50.0, 'var': 0.0, 'max': 50.0, 'count_above_50M': 0}
        
        # Tournament multiplier
        comp = row.get('Tournament', 'Friendly')
        tourney_mult = 2.0 if 'World Cup' in str(comp) else (1.5 if 'qualification' in str(comp).lower() else 1.0)
        
        match_info = {
            'home_idx': team_to_idx[h],
            'away_idx': team_to_idx[a],
            'home_elo': h_elo_before,
            'away_elo': a_elo_before,
            'home_seq': np.array(h_seq, dtype=np.float32),
            'away_seq': np.array(a_seq, dtype=np.float32),
            'home_squad_sum': np.log1p(h_squad['sum']),
            'away_squad_sum': np.log1p(a_squad['sum']),
            'home_squad_var': np.log1p(h_squad['var'] + 1),
            'away_squad_var': np.log1p(a_squad['var'] + 1),
            'home_squad_max': np.log1p(h_squad['max']),
            'away_squad_max': np.log1p(a_squad['max']),
            'home_count_50M': h_squad['count_above_50M'],
            'away_count_50M': a_squad['count_above_50M'],
            'h2h_goal_diff': h2h_gd,
            'h2h_wins_home': h2h_wins_home,
            'home_gd_median': h_gd_med,
            'away_gd_median': a_gd_med,
            'home_gd_p95': h_gd_p95,
            'away_gd_p95': a_gd_p95,
            'tourney_mult': tourney_mult,
            'home_fifa': home_fifa,
            'away_fifa': away_fifa,
            'fifa_diff': home_fifa - away_fifa,
            'target_home_goals': row['Home Score'],
            'target_away_goals': row['Away Score'],
            'weight': weight,
            'year': year,
        }
        match_data.append(match_info)
        
        # Update Elo
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
        
        team_history[h].append({'goals_for': h_score, 'goals_against': a_score, 'opponent_elo': a_elo_before, 'was_home': not neutral})
        team_history[a].append({'goals_for': a_score, 'goals_against': h_score, 'opponent_elo': h_elo_before, 'was_home': False})
    
    with open(ELO_CACHE, 'wb') as f:
        pickle.dump({
            'elo_dict': dict(elo),
            'team_history_dict': {k: list(v) for k, v in team_history.items()},
            'match_data': match_data,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"   Elo data cached ({len(match_data):,} matches)")

# ==========================================
# BUILD FEATURE MATRICES (NOW WITH FIFA FEATURES)
# ==========================================
print("[5/6] Building feature matrices...")
static_features = []
for m in match_data:
    feat = [
        m['home_elo'] - m['away_elo'],
        m['home_squad_sum'], m['away_squad_sum'],
        m['home_squad_var'], m['away_squad_var'],
        m['home_squad_max'], m['away_squad_max'],
        m['home_count_50M'], m['away_count_50M'],
        m['h2h_goal_diff'],
        m['h2h_wins_home'],
        m['home_gd_median'], m['away_gd_median'],
        m['home_gd_p95'], m['away_gd_p95'],
        m['tourney_mult'],
        m['home_fifa'], m['away_fifa'], m['fifa_diff']
    ]
    static_features.append(feat)

static_features = np.array(static_features, dtype=np.float32)
static_features = np.nan_to_num(static_features, nan=0.0, posinf=10.0, neginf=-10.0)
scaler = StandardScaler()
static_features_scaled = scaler.fit_transform(static_features)

# Extract arrays
home_idx = np.array([m['home_idx'] for m in match_data])
away_idx = np.array([m['away_idx'] for m in match_data])
home_seq = np.stack([m['home_seq'] for m in match_data])
away_seq = np.stack([m['away_seq'] for m in match_data])
targets = np.array([[m['target_home_goals'], m['target_away_goals']] for m in match_data], dtype=np.float32)
weights = np.array([m['weight'] for m in match_data], dtype=np.float32)
years = np.array([m['year'] for m in match_data])

# Train/val/test split
train_mask = years < VAL_START_YEAR
val_mask = (years >= VAL_START_YEAR) & (years < TEST_START_YEAR)
test_mask = years >= TEST_START_YEAR

print(f"   Train: {train_mask.sum():,} | Val: {val_mask.sum():,} | Test: {test_mask.sum():,}")

# ==========================================
# MODEL DEFINITION (STATIC DIM NOW 19)
# ==========================================
class PoissonLoss(nn.Module):
    def forward(self, pred, target, weights=None):
        pred = torch.clamp(pred, min=0.1, max=6.0)
        loss = pred - target * torch.log(pred + 1e-8)
        loss = loss.sum(dim=1)
        if weights is not None:
            loss = loss * weights
        return loss.mean()

class MatchPredictor(nn.Module):
    def __init__(self, num_teams, embed_dim=16, hist_len=10, hist_input_dim=4, static_dim=19):
        super().__init__()
        self.team_embedding = nn.Embedding(num_teams, embed_dim)
        self.hist_proj = nn.Linear(hist_input_dim, 32)
        encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, batch_first=True, dropout=0.2)
        self.hist_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.static_branch = nn.Sequential(
            nn.Linear(static_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU()
        )
        fusion_dim = embed_dim*2 + 32*2 + 32
        self.final_mlp = nn.Sequential(
            nn.Linear(fusion_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 2), nn.Softplus()
        )
    
    def forward(self, home_id, away_id, home_seq, away_seq, static):
        home_emb = self.team_embedding(home_id)
        away_emb = self.team_embedding(away_id)
        home_proj = self.hist_proj(home_seq)
        away_proj = self.hist_proj(away_seq)
        home_state = self.hist_encoder(home_proj)[:, -1, :]
        away_state = self.hist_encoder(away_proj)[:, -1, :]
        static_out = self.static_branch(static)
        combined = torch.cat([home_emb, away_emb, home_state, away_state, static_out], dim=1)
        return self.final_mlp(combined)

static_dim = static_features_scaled.shape[1]
model = MatchPredictor(num_teams, EMBEDDING_DIM, HISTORY_LEN, 4, static_dim).to(device)
criterion = PoissonLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

total_params = sum(p.numel() for p in model.parameters())
print(f"   Model: {total_params:,} parameters")

# ==========================================
# TRAINING (WITH PIN_MEMORY=FALSE)
# ==========================================
print("[6/6] Training model (80 epochs, ~3-5 min)...")

def create_dataloader(home_idx, away_idx, home_seq, away_seq, static, y, w, batch_size, shuffle=True):
    home_idx_t = torch.tensor(home_idx, dtype=torch.long)
    away_idx_t = torch.tensor(away_idx, dtype=torch.long)
    home_seq_t = torch.tensor(home_seq, dtype=torch.float32)
    away_seq_t = torch.tensor(away_seq, dtype=torch.float32)
    static_t = torch.tensor(static, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    w_t = torch.tensor(w, dtype=torch.float32)
    dataset = TensorDataset(home_idx_t, away_idx_t, home_seq_t, away_seq_t, static_t, y_t, w_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=False, num_workers=0)

train_loader = create_dataloader(
    home_idx[train_mask], away_idx[train_mask], 
    home_seq[train_mask], away_seq[train_mask], 
    static_features_scaled[train_mask], targets[train_mask], weights[train_mask], 
    BATCH_SIZE, shuffle=True
)

val_loader = create_dataloader(
    home_idx[val_mask], away_idx[val_mask], 
    home_seq[val_mask], away_seq[val_mask], 
    static_features_scaled[val_mask], targets[val_mask], weights[val_mask], 
    BATCH_SIZE, shuffle=False
)

best_val_loss = float('inf')
best_state = None

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        hid, aid, hseq, aseq, stat, yb, wb = [x.to(device) for x in batch]
        optimizer.zero_grad()
        pred = model(hid, aid, hseq, aseq, stat)
        loss = criterion(pred, yb, wb)
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            hid, aid, hseq, aseq, stat, yb, wb = [x.to(device) for x in batch]
            pred = model(hid, aid, hseq, aseq, stat)
            loss = criterion(pred, yb, wb)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = model.state_dict().copy()
        torch.save(best_state, f"{OUTPUT_DIR}best_model.pt")
    
    scheduler.step(val_loss)
    if (epoch + 1) % 10 == 0:
        print(f"   Epoch {epoch+1:3d}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")

model.load_state_dict(best_state)

# ==========================================
# EVALUATION (OPTIONAL, NOT REQUIRED FOR FINAL OUTPUT)
# ==========================================
print("\nEvaluating on test set...")
test_loader = create_dataloader(
    home_idx[test_mask], away_idx[test_mask], 
    home_seq[test_mask], away_seq[test_mask], 
    static_features_scaled[test_mask], targets[test_mask], weights[test_mask], 
    BATCH_SIZE, shuffle=False
)

model.eval()
test_preds = []
test_targets = []
with torch.no_grad():
    for batch in test_loader:
        hid, aid, hseq, aseq, stat, yb, wb = [x.to(device) for x in batch]
        pred = model(hid, aid, hseq, aseq, stat)
        test_preds.append(pred.cpu().numpy())
        test_targets.append(yb.cpu().numpy())
test_preds = np.concatenate(test_preds, axis=0)
test_targets = np.concatenate(test_targets, axis=0)

def accuracy(pred, true):
    return np.mean(np.sign(pred[:,0]-pred[:,1]) == np.sign(true[:,0]-true[:,1]))

print(f"\n{'='*50}")
print(f"TEST RESULTS")
print(f"{'='*50}")
print(f"Accuracy: {accuracy(test_preds, test_targets):.2%}")
print(f"Home MAE: {mean_absolute_error(test_targets[:,0], test_preds[:,0]):.3f}")
print(f"Away MAE: {mean_absolute_error(test_targets[:,1], test_preds[:,1]):.3f}")

# ==========================================
# 2026 WORLD CUP PREDICTIONS (CSV ONLY)
# ==========================================
print("\nGenerating 2026 World Cup predictions...")

groups = {
    'A': ["Mexico","South Africa","South Korea","Czech Republic"],
    'B': ["Canada","Bosnia and Herzegovina","Qatar","Switzerland"],
    'C': ["Brazil","Morocco","Haiti","Scotland"],
    'D': ["United States","Paraguay","Australia","Turkey"],
    'E': ["Germany","Curaçao","Ivory Coast","Ecuador"],
    'F': ["Netherlands","Japan","Sweden","Tunisia"],
    'G': ["Belgium","Egypt","Iran","New Zealand"],
    'H': ["Spain","Cape Verde","Saudi Arabia","Uruguay"],
    'I': ["France","Senegal","Iraq","Norway"],
    'J': ["Argentina","Algeria","Austria","Jordan"],
    'K': ["Portugal","DR Congo","Uzbekistan","Colombia"],
    'L': ["England","Croatia","Ghana","Panama"]
}

def build_team_features(team, year=2026):
    hist = team_history.get(team, [])
    seq = []
    for m in hist[-HISTORY_LEN:]:
        seq.append([m['goals_for'], m['goals_against'], m['opponent_elo'], 1.0 if m['was_home'] else 0.0])
    if len(seq) < HISTORY_LEN:
        pad = [[0.0, 0.0, 1500.0, 0.0]] * (HISTORY_LEN - len(seq))
        seq = pad + seq
    seq = np.array(seq[-HISTORY_LEN:], dtype=np.float32)
    # World Cup is neutral → NO home advantage
    elo_val = elo.get(team, 1500.0)
    squad = get_squad_features(team, year)
    if not isinstance(squad, dict):
        squad = {'sum': 50.0, 'var': 0.0, 'max': 50.0, 'count_above_50M': 0}
    # For prediction date, use a date after the last ranking (e.g., 2026-06-01)
    pred_date = pd.Timestamp('2026-06-01')
    fifa_pts = get_fifa_points(team, pred_date)
    return team_to_idx.get(team, 0), seq, elo_val, squad, fifa_pts

team_base = {}
for group_teams in groups.values():
    for team in group_teams:
        if team not in team_base and team in team_to_idx:
            team_base[team] = build_team_features(team, 2026)

predictions = []
model.eval()
with torch.no_grad():
    for group, teams in groups.items():
        matches = [(teams[0],teams[1]), (teams[2],teams[3]), (teams[0],teams[2]),
                   (teams[3],teams[1]), (teams[3],teams[0]), (teams[1],teams[2])]
        for home, away in matches:
            if home not in team_base or away not in team_base:
                continue
            hid, h_seq, h_elo, h_squad, h_fifa = team_base[home]
            aid, a_seq, a_elo, a_squad, a_fifa = team_base[away]
            
            feat = np.array([
                h_elo - a_elo,
                np.log1p(h_squad['sum']), np.log1p(a_squad['sum']),
                np.log1p(h_squad['var']+1), np.log1p(a_squad['var']+1),
                np.log1p(h_squad['max']), np.log1p(a_squad['max']),
                h_squad['count_above_50M'], a_squad['count_above_50M'],
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0,  # h2h, gd_medians, gd_p95, tourney_mult placeholder
                h_fifa, a_fifa, h_fifa - a_fifa
            ], dtype=np.float32)
            feat_scaled = scaler.transform(feat.reshape(1, -1))
            
            hid_t = torch.tensor([hid], device=device)
            aid_t = torch.tensor([aid], device=device)
            h_seq_t = torch.tensor(h_seq.reshape(1, HISTORY_LEN, 4), device=device)
            a_seq_t = torch.tensor(a_seq.reshape(1, HISTORY_LEN, 4), device=device)
            stat_t = torch.tensor(feat_scaled, device=device)
            
            home_xg, away_xg = model(hid_t, aid_t, h_seq_t, a_seq_t, stat_t).cpu().numpy()[0]
            predictions.append({
                'Group': group,
                'Home_Team': home,
                'Away_Team': away,
                'Home_xG': round(float(home_xg), 2),
                'Away_xG': round(float(away_xg), 2)
            })

# Save ONLY the CSV that the C++ simulator expects
out_csv = f"{OUTPUT_DIR}group_stage_predictions.csv"
pd.DataFrame(predictions).to_csv(out_csv, index=False)

print(f"\n{'='*50}")
print(f"✅ COMPLETE!")
print(f"{'='*50}")
print(f"Predictions saved to: {out_csv}")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()