#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIFA World Cup 2026 - Standings Probabilities & Match Results
FIXED VERSION - Handles JSON correctly with Unified Legends
"""

import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import sys

# =========================================================================
# CONFIGURATION
# =========================================================================
PROJECT_DATA_DIR = "/mnt/4tb_games/Projects/2026/data/"
JSON_INPUT_PATH = os.path.join(PROJECT_DATA_DIR, "simulation_matrices.json")

OUTPUT_IMAGE_STANDINGS = os.path.join(PROJECT_DATA_DIR, "world_cup_2026_standings_probabilities.png")
OUTPUT_IMAGE_FIXTURES = os.path.join(PROJECT_DATA_DIR, "world_cup_2026_match_results.png")

# ---- FONT SIZE SETTINGS ----
TEAM_NAME_SIZE = 11
SCORE_SIZE = 14
WIN_PCT_SIZE = 8
GROUP_TITLE_SIZE = 14

# ---- SPACING SETTINGS ----
ROW_SPACING = 1.3
SUBPLOT_TOP_MARGIN = 1.8
SUBPLOT_BOTTOM_MARGIN = 0.3

# ---- COLORS ----
HOME_WIN_COLOR = '#1f77b4'  # Blue
DRAW_COLOR = '#7f8c8d'      # Gray
AWAY_WIN_COLOR = '#9467bd'  # Purple

plt.rcParams['figure.facecolor'] = '#f8f9fa'
plt.rcParams['axes.facecolor'] = '#ffffff'
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']

# =========================================================================
# HELPERS
# =========================================================================
def normalize_name(name: str) -> str:
    """Clean team names for display"""
    return name.replace("_", " ").replace("Bosnia and Herzegovina", "B&H").strip()

COUNTRY_CODES = {
    "Argentina": "ARG", "Australia": "AUS", "Belgium": "BEL",
    "Bosnia and Herzegovina": "BIH", "Brazil": "BRA", "Canada": "CAN",
    "Cape Verde": "CPV", "Colombia": "COL", "Croatia": "CRO", 
    "Czech Republic": "CZE", "DR Congo": "COD", "Ecuador": "ECU", 
    "Egypt": "EGY", "England": "ENG", "France": "FRA", "Germany": "GER", 
    "Ghana": "GHA", "Haiti": "HAI", "Iran": "IRN", "Iraq": "IRQ", 
    "Ivory Coast": "CIV", "Japan": "JPN", "Jordan": "JOR", "Mexico": "MEX", 
    "Morocco": "MAR", "Netherlands": "NED", "New Zealand": "NZL", 
    "Norway": "NOR", "Panama": "PAN", "Paraguay": "PAR", "Portugal": "POR", 
    "Qatar": "QAT", "Saudi Arabia": "KSA", "Scotland": "SCO", "Senegal": "SEN", 
    "South Africa": "RSA", "South Korea": "KOR", "Spain": "ESP", "Sweden": "SWE", 
    "Switzerland": "SUI", "Tunisia": "TUN", "Turkey": "TUR", "United States": "USA", 
    "Uruguay": "URU", "Uzbekistan": "UZB"
}

def get_country_code(name: str) -> str:
    """Get 3-letter country code"""
    clean_name = name.replace("_", " ")
    if "Bosnia" in clean_name: return "BIH"
    if "Czech" in clean_name: return "CZE"
    if "Cape Verde" in clean_name: return "CPV"
    if "Côte" in clean_name or "Ivory" in clean_name: return "CIV"
    if "South Africa" in clean_name: return "RSA"
    if "South Korea" in clean_name: return "KOR"
    if "United States" in clean_name: return "USA"
    if "DR Congo" in clean_name: return "COD"
    return COUNTRY_CODES.get(clean_name, clean_name[:3].upper())

# =========================================================================
# LOAD JSON DATA
# =========================================================================
print("Loading JSON data...")
try:
    with open(JSON_INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"❌ Could not find '{JSON_INPUT_PATH}'.", file=sys.stderr)
    sys.exit(1)
except json.JSONDecodeError as e:
    print(f"❌ JSON PARSE ERROR: {e}", file=sys.stderr)
    sys.exit(1)

if "groups" not in data:
    print(f"❌ JSON STRUCTURE ERROR: 'groups' key not found.", file=sys.stderr)
    sys.exit(1)

groups_data = data["groups"]
groups = sorted(list(groups_data.keys()))
print(f"✅ Loaded {len(groups)} groups successfully")

# =========================================================================
# GRAPHIC 1: STANDINGS PROBABILITY DISTRIBUTION GRID
# =========================================================================
print("\n-> Constructing Graphic 1: Predicted Standings Probabilities...")
fig1 = plt.figure(figsize=(24, 22))
grid_shape1 = (4, 3)
axes1 = []
for r in range(4):
    for c in range(3):
        ax = plt.subplot2grid(grid_shape1, (r, c))
        axes1.append(ax)

colors_standings = ['#1d3557', '#457b9d', '#f4a261', '#e63946']
labels_standings = ['1st Place', '2nd Place', '3rd Place', '4th Place']

bar_handles = []

for idx, g_name in enumerate(groups):
    if idx >= len(axes1):
        break
        
    ax = axes1[idx]
    g_data = groups_data[g_name]
    
    if "table_probabilities" not in g_data:
        print(f"   Warning: No table_probabilities for group {g_name}")
        continue
    
    teams_data = sorted(g_data["table_probabilities"], 
                       key=lambda x: x["expected_points"], reverse=True)
    
    team_names_raw = [t["team"] for t in teams_data]
    
    p1 = [t["probabilities"]["1st"] * 100 for t in teams_data]
    p2 = [t["probabilities"]["2nd"] * 100 for t in teams_data]
    p3 = [t["probabilities"]["3rd"] * 100 for t in teams_data]
    p4 = [t["probabilities"]["4th"] * 100 for t in teams_data]
    
    y_pos = np.arange(len(team_names_raw))
    ax.set_xlim(-12, 100)
    
    b1 = ax.barh(y_pos, p1, color=colors_standings[0], edgecolor='#ffffff', height=0.55, zorder=2)
    b2 = ax.barh(y_pos, p2, left=p1, color=colors_standings[1], edgecolor='#ffffff', height=0.55, zorder=2)
    b3 = ax.barh(y_pos, p3, left=np.array(p1)+np.array(p2), color=colors_standings[2], edgecolor='#ffffff', height=0.55, zorder=2)
    b4 = ax.barh(y_pos, p4, left=np.array(p1)+np.array(p2)+np.array(p3), color=colors_standings[3], edgecolor='#ffffff', height=0.55, zorder=2)
    
    if idx == 0:
        bar_handles = [b1, b2, b3, b4]
    
    ax.axhline(y=1.5, color='#6c757d', linestyle=':', linewidth=1.5, alpha=0.7, zorder=1)
    
    for i, raw_name in enumerate(team_names_raw):
        code = get_country_code(raw_name)
        ax.text(-9.0, i, code, fontsize=13, fontweight='bold',
                va='center', ha='center', color='#2b2d42', zorder=3)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([])
    ax.invert_yaxis()
    
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)
    
    ax.set_title(f"GROUP {g_name}", fontsize=17, fontweight='black',
                 color='#1d3557', pad=12, loc='left')
    ax.xaxis.grid(True, linestyle='--', alpha=0.3, color='#6c757d', zorder=0)
    ax.tick_params(left=False, bottom=False)
    
    for i, t in enumerate(teams_data):
        ax.text(103, i, f"{t['expected_points']:.2f} XP", va='center', ha='left',
                fontsize=10, color='#4a4e69', fontweight='bold', zorder=3)

xp_proxy = plt.plot([], [], color="none", label="XP = Expected Points")[0]

fig1.legend(
    handles=bar_handles + [xp_proxy], 
    labels=labels_standings + ["XP = Expected Points"],
    loc='upper center', 
    bbox_to_anchor=(0.5, 0.94), 
    ncol=5, 
    fontsize=13, 
    frameon=True, 
    facecolor='#ffffff', 
    edgecolor='#cbd5e1'
)

fig1.suptitle("FIFA WORLD CUP 2026 — PREDICTED GROUP STAGE STANDINGS PROBABILITIES",
              fontsize=24, fontweight='black', y=0.98, color='#1d3557')
plt.tight_layout()
plt.subplots_adjust(hspace=0.28, wspace=0.3, bottom=0.03, top=0.89)
plt.savefig(OUTPUT_IMAGE_STANDINGS, bbox_inches='tight', dpi=180)
plt.close()
print(f"   ✅ Standings saved to: {OUTPUT_IMAGE_STANDINGS}")

# =========================================================================
# GRAPHIC 2: FIXTURES - MATCH RESULTS
# =========================================================================
print("\n-> Constructing Graphic 2: Match Results...")

fig2 = plt.figure(figsize=(28, 26))
gs = gridspec.GridSpec(4, 3, figure=fig2, hspace=0.35, wspace=0.35)

for idx, g_name in enumerate(groups):
    if idx >= 12:  
        break
        
    ax = plt.subplot(gs[idx // 3, idx % 3])
    g_data = groups_data[g_name]
    matches = g_data.get("matches", [])
    num_matches = len(matches)
    
    content_height = SUBPLOT_TOP_MARGIN + (num_matches * ROW_SPACING) + SUBPLOT_BOTTOM_MARGIN
    ax.set_ylim(0, content_height)
    ax.set_xlim(0, 12)
    ax.axis('off')
    
    header_y = content_height - SUBPLOT_TOP_MARGIN + 0.3
    ax.fill_between([0, 12], header_y - 0.5, header_y + 0.4, 
                    color='#1d3557', alpha=0.08, zorder=0)
    ax.text(0.3, header_y, f"GROUP {g_name}", fontsize=GROUP_TITLE_SIZE,
            fontweight='black', color='#1d3557', va='center', zorder=3)
    
    y_offset = header_y - 0.85
    
    for m_idx, m in enumerate(matches):
        home_clean = m["home"].replace("_", " ").replace("Bosnia and Herzegovina", "B&H")
        away_clean = m["away"].replace("_", " ").replace("Bosnia and Herzegovina", "B&H")
        
        if "likely_scores" in m and len(m["likely_scores"]) > 0:
            score = m["likely_scores"][0]["score"]
        else:
            score = "--"
        
        home_prob = m.get("home_win_prob", 0.333) * 100
        draw_prob = m.get("draw_prob", 0.333) * 100
        away_prob = m.get("away_win_prob", 0.333) * 100
        
        if m_idx % 2 == 0:
            ax.fill_between([0, 12], y_offset - 0.6, y_offset + 0.5,
                            color='#f1f5f9', alpha=0.3, zorder=0)
        
        ax.text(3.0, y_offset, f"{normalize_name(home_clean)} ({home_prob:.0f}%)", 
                fontsize=TEAM_NAME_SIZE, fontweight='bold', color=HOME_WIN_COLOR,
                ha='right', va='center', zorder=4)
        
        ax.text(6.0, y_offset, score, fontsize=SCORE_SIZE, fontweight='black',
                color='#0f172a', ha='center', va='center', zorder=4,
                bbox=dict(facecolor='#ffffff', edgecolor='#cbd5e1',
                         boxstyle='round,pad=0.2', linewidth=0.8))
        
        ax.text(9.0, y_offset, f"{normalize_name(away_clean)} ({away_prob:.0f}%)", 
                fontsize=TEAM_NAME_SIZE, fontweight='bold', color=AWAY_WIN_COLOR,
                ha='left', va='center', zorder=4)
        
        ax.text(6.0, y_offset - 0.6, f"Draw: {draw_prob:.0f}%", 
                fontsize=WIN_PCT_SIZE, fontweight='bold', color=DRAW_COLOR,
                ha='center', va='center', zorder=4)
        
        y_offset -= ROW_SPACING

for idx in range(len(groups), 12):
    row = idx // 3
    col = idx % 3
    if row < 4 and col < 3:
        ax = plt.subplot(gs[row, col])
        ax.axis('off')
        ax.set_visible(False)

# ---- NEW: ONE LINE TEXT LEGEND FOR MATCH RESULTS ----
fig2.text(0.5, 0.94, "Percentages (%) indicate the probability of a Home Win (Blue), a Draw (Gray), or an Away Win (Purple)",
          fontsize=13, fontweight='medium', color='#475569', ha='center', va='center',
          bbox=dict(facecolor='#ffffff', edgecolor='#e2e8f0', boxstyle='round,pad=0.4'))

fig2.suptitle("FIFA WORLD CUP 2026 — PREDICTED MATCH RESULTS", 
              fontsize=26, fontweight='black', y=0.98, color='#1d3557')
plt.savefig(OUTPUT_IMAGE_FIXTURES, bbox_inches='tight', dpi=180)
plt.close()
print(f"   ✅ Fixtures saved to: {OUTPUT_IMAGE_FIXTURES}")

print("\n" + "=" * 73)
print("🎉 BOTH GRAPHICS COMPILED SUCCESSFULLY!")
print("=" * 73)
print(f"   • {OUTPUT_IMAGE_STANDINGS} (Includes placement / XP legend)")
print(f"   • {OUTPUT_IMAGE_FIXTURES} (Includes win/draw percentage legend)")
print("=" * 73)