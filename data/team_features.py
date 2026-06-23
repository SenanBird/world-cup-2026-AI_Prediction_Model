import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_data_from_file(filename):
    """Reads a file and dynamically extracts Elo ratings and Squad market values."""
    elo_dict = {}
    squad_dict = {}

    # --- FIX: Absolute Path Resolution ---
    # Finds the directory where team_features.py actually lives
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, filename)

    if not os.path.exists(full_path):
        raise FileNotFoundError(
            f"The file '{filename}' was not found at expected path: {full_path}"
        )

    with open(full_path, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.split("\n")
    current_section = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if "ELO RATINGS" in line:
            current_section = "elo"
            continue
        elif "CURRENT SQUAD SUMS" in line:
            current_section = "squad"
            continue
        elif line.startswith("==="):
            continue

        if ":" in line:
            country, rest = line.split(":", 1)
            country = country.strip()
            rest = rest.strip()

            if current_section == "elo":
                try:
                    elo_dict[country] = float(rest)
                except ValueError:
                    pass
            elif current_section == "squad":
                if "Sum =" in rest:
                    try:
                        sum_part = (
                            rest.split("|")[0].replace("Sum =", "").strip()
                        )
                        squad_dict[country] = float(sum_part)
                    except (ValueError, IndexError):
                        pass

    df = pd.DataFrame(
        {"Elo": pd.Series(elo_dict), "Squad_Sum": pd.Series(squad_dict)}
    ).dropna()
    return df, script_dir


# ==============================================================================
# CONFIGURATION
# ==============================================================================
FILENAME = "team_features_debug.txt"

try:
    df, output_dir = parse_data_from_file(FILENAME)

    if df.empty:
        print(
            f"[-] Parsing complete, but no structured data was found inside '{FILENAME}'."
        )
    else:
        print(f"[+] Successfully loaded {len(df)} teams from '{FILENAME}'")

        # Helper to save files to the script's home directory
        def get_save_path(img_name):
            return os.path.join(output_dir, img_name)

        # ----------------------------------------------------------------------
        # GRAPH 1: Just Elo
        # ----------------------------------------------------------------------
        df_elo_sorted = df.sort_values(by="Elo", ascending=True)
        fig1, ax1 = plt.subplots(figsize=(10, 12))
        ax1.barh(
            df_elo_sorted.index,
            df_elo_sorted["Elo"],
            color="skyblue",
            edgecolor="grey",
        )
        ax1.set_title(
            "World Cup 2026 Participants: Elo Ratings",
            fontsize=14,
            weight="bold",
        )
        ax1.set_xlabel("Elo Rating", fontsize=12)
        ax1.set_xlim(1400, 2150)
        ax1.grid(True, axis="x", linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(get_save_path("elo_ratings.png"), dpi=300)
        plt.close()
        print(f"[+] Saved '{get_save_path('elo_ratings.png')}'")

        # ----------------------------------------------------------------------
        # GRAPH 2: Just Squad Sum
        # ----------------------------------------------------------------------
        df_squad_sorted = df.sort_values(by="Squad_Sum", ascending=True)
        fig2, ax2 = plt.subplots(figsize=(10, 12))
        ax2.barh(
            df_squad_sorted.index,
            df_squad_sorted["Squad_Sum"],
            color="lightgreen",
            edgecolor="grey",
        )
        ax2.set_title(
            "World Cup 2026 Participants: Squad Market Value",
            fontsize=14,
            weight="bold",
        )
        ax2.set_xlabel("Total Squad Value (M€)", fontsize=12)
        ax2.grid(True, axis="x", linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(get_save_path("squad_sums.png"), dpi=300)
        plt.close()
        print(f"[+] Saved '{get_save_path('squad_sums.png')}'")

        # ----------------------------------------------------------------------
        # GRAPH 3: Combined View
        # ----------------------------------------------------------------------
        fig3, ax3 = plt.subplots(figsize=(11, 8))
        ax3.scatter(
            df["Squad_Sum"],
            df["Elo"],
            color="purple",
            alpha=0.7,
            edgecolors="k",
            s=60,
        )

        m, b = np.polyfit(df["Squad_Sum"], df["Elo"], 1)
        correlation = df["Squad_Sum"].corr(df["Elo"])
        ax3.plot(
            df["Squad_Sum"],
            m * df["Squad_Sum"] + b,
            color="orange",
            linestyle="--",
            linewidth=1.5,
            label=f"Trendline (r = {correlation:.2f})",
        )

        for country, row in df.iterrows():
            if (
                row["Squad_Sum"] > 500
                or row["Elo"] > 1950
                or country
                in ["Brazil", "Argentina", "France", "England", "Spain"]
            ):
                ax3.annotate(
                    country,
                    (row["Squad_Sum"], row["Elo"]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    ha="left",
                    fontsize=9,
                )

        ax3.set_title(
            "Combined View: Squad Market Value vs. Elo Rating",
            fontsize=14,
            weight="bold",
            pad=12,
        )
        ax3.set_xlabel("Total Squad Value (M€)", fontsize=12)
        ax3.set_ylabel("Elo Rating", fontsize=12)
        ax3.grid(True, linestyle=":", alpha=0.6)
        ax3.legend(loc="lower right", fontsize=11)
        plt.tight_layout()
        plt.savefig(get_save_path("combined_squad_elo.png"), dpi=300)
        plt.close()
        print(f"[+] Saved '{get_save_path('combined_squad_elo.png')}'")

except Exception as e:
    print(f"[-] An error occurred: {e}")