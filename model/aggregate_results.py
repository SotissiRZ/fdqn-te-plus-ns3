#!/usr/bin/env python3
"""
aggregate_results.py — Agrégation multi-seeds et graphes avec intervalles de confiance
=========================================================================================
Lit les fichiers summary.csv / metrics.csv générés par run_multiseed.sh
pour chaque protocole et chaque seed, puis :

  1. Calcule moyenne ± écart-type sur toutes les seeds
  2. Régénère tous les graphes avec intervalles de confiance à 95%
  3. Exporte un tableau synthétique (aggregate_summary.csv)

Usage :
  python3 aggregate_results.py --results_dir results_eval --seeds 42 43 44 45 46
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats as scipy_stats
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────────────────────

PROTOCOLS = {
    "FDQN_TEplus":    ("FDQN-TE+ (proposé)",       "#1a7a4a", "-",   "^", 2.0),
    "LEACH":          ("LEACH",                     "#e05252", "--",  "o", 1.5),
    "HEED":           ("HEED",                      "#9b59b6", "--",  "D", 1.5),
    "QRouting":       ("Q-Routing",                 "#c07820", "-.",  "s", 1.5),
    "DQN_LEACH":      ("DQN-LEACH (baseline DRL)",  "#2980b9", "-.",  "v", 1.5),
}

# Dossiers résultats pour chaque protocole (clé PROTOCOLS → sous-dossier résultats)
# HEED et DQN_LEACH utilisent des noms de dossiers différents des autres
PROTO_DIRS = {
    "FDQN_TEplus": "FDQN_TEplus",
    "LEACH":       "LEACH",
    "HEED":        "HEED",
    "QRouting":    "QRouting",
    "DQN_LEACH":   "DQN_LEACH",   # --resultsDir=results_eval/DQN_LEACH
}

# Format du fichier summary selon le protocole
# "fdqnte" → fdqnte_summary.csv avec colonnes Param,Value et clés FND_t/HND_t
# "eval"   → summary.csv avec colonnes Metric,Value et clés FND_s/HND_s
PROTO_SUMMARY_FORMAT = {
    "FDQN_TEplus": "fdqnte",
    "LEACH":       "eval",
    "HEED":        "eval",
    "QRouting":    "eval",
    "DQN_LEACH":   "fdqnte",  # fdqn_noIFO écrit fdqnte_summary.csv (Param,Value + FND_t)
}

FIGSIZE_SINGLE = (10, 5)
FIGSIZE_DASH   = (14, 11)
ALPHA_BAND     = 0.15   # transparence des bandes ±σ
CI_Z           = 1.96   # z pour IC 95%

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi":     150,
})

# ─── Lecture des données ──────────────────────────────────────────────────────

def read_summary(path: Path) -> dict:
    """
    Lit un summary.csv et retourne un dict {metric: value}.
    Gere les formats :
      - "Param,Value"  (FDQN-TE+)
      - "Metric,Value" (LEACH, Q-Routing)
      - colonnes libres  (1ere colonne=nom, 2e=valeur)
    """
    d = {}
    try:
        df = pd.read_csv(path, comment="#")
        # Detecter la colonne cle (Param ou Metric ou premiere colonne)
        key_col = None
        for candidate in ("Param", "Metric", "metric", "param"):
            if candidate in df.columns:
                key_col = candidate
                break
        val_col = None
        for candidate in ("Value", "value"):
            if candidate in df.columns:
                val_col = candidate
                break

        if key_col and val_col:
            for _, row in df.iterrows():
                try:
                    d[str(row[key_col])] = float(row[val_col])
                except (ValueError, TypeError):
                    d[str(row[key_col])] = row[val_col]
        else:
            # Fallback : 1ere colonne = nom, 2e = valeur
            for _, row in df.iterrows():
                try:
                    d[str(row.iloc[0])] = float(row.iloc[1])
                except Exception:
                    pass
    except Exception as e:
        print(f"  [WARN] Impossible de lire {path}: {e}")
    return d


def read_metrics_timeseries(path: Path) -> pd.DataFrame:
    """Lit metrics.csv (ou energy.csv) et retourne un DataFrame."""
    try:
        df = pd.read_csv(path, comment="#")
        return df
    except Exception as e:
        print(f"  [WARN] Impossible de lire {path}: {e}")
        return pd.DataFrame()



def find_seed_dir(results_dir: Path, proto_dir_name: str, seed: int) -> Path | None:
    """
    Cherche le dossier seed dans deux emplacements :
      1. results_dir/<proto>/seed_<X>/            (structure run_multiseed.sh)
      2. results_dir/scale_N*/<proto>/seed_<X>/   (structure run_scalability.sh)
    Retourne le premier trouvé, ou None.
    """
    # Chemin direct
    direct = results_dir / proto_dir_name / f"seed_{seed}"
    if direct.exists():
        return direct
    # Scan des sous-dossiers scale_N*
    for scale_dir in sorted(results_dir.glob("scale_N*")):
        candidate = scale_dir / proto_dir_name / f"seed_{seed}"
        if candidate.exists():
            return candidate
    return None

def collect_all_data(results_dir: Path, seeds: list, protocols: dict):
    """
    Collecte pour chaque protocole et chaque seed :
      - summary scalaires  → {proto: {metric: [val_seed1, val_seed2, ...]}}
      - séries temporelles → {proto: [df_seed1, df_seed2, ...]}

    Gère deux formats de summary :
      "fdqnte" : fdqnte_summary.csv (Param,Value) — FDQN_TEplus, DQN_LEACH
                 clés : FND_t, HND_t, LND_t, PDR_RL_preFND_pct, PDR_RL_pct
      "eval"   : summary.csv (Metric,Value) — LEACH, HEED, QRouting
                 clés : FND_s, HND_s, LND_s, PDR_Stable_pct, PDR_Global_pct
    """
    scalars = {p: {} for p in protocols}
    series  = {p: [] for p in protocols}

    for proto in protocols:
        # Dossier résultats (peut différer du nom de clé PROTOCOLS)
        proto_dir_name = PROTO_DIRS.get(proto, proto)
        fmt = PROTO_SUMMARY_FORMAT.get(proto, "eval")

        for seed in seeds:
            seed_dir = find_seed_dir(results_dir, proto_dir_name, seed)
            if seed_dir is None:
                continue

            # ── Summary scalaires ─────────────────────────────────────────
            if fmt == "fdqnte":
                summary_files = ["fdqnte_summary.csv", "summary.csv"]
            else:
                summary_files = ["summary.csv", "fdqnte_summary.csv"]

            for fname in summary_files:
                spath = seed_dir / fname
                if spath.exists():
                    s = read_summary(spath)
                    for k, v in s.items():
                        if isinstance(v, (int, float)):
                            scalars[proto].setdefault(k, []).append(v)
                    break

            # ── Séries temporelles ────────────────────────────────────────
            for fname in ["metrics.csv",
                          "energy/fdqnte_energy.csv",
                          "energy.csv"]:
                tpath = seed_dir / fname
                if tpath.exists():
                    df = read_metrics_timeseries(tpath)
                    if not df.empty:
                        series[proto].append(df)
                    break

    return scalars, series


# ─── Alignement des séries sur un axe temporel commun ─────────────────────────

def align_series(dfs: list, time_col: str, value_col: str,
                 t_min: float = 0, t_max: float = 3500, n_pts: int = 70):
    """
    Interpole toutes les séries sur une grille commune [t_min, t_max].
    Retourne (t_grid, mean, ci_low, ci_high, std).
    """
    t_grid = np.linspace(t_min, t_max, n_pts)
    interpolated = []

    for df in dfs:
        if time_col not in df.columns or value_col not in df.columns:
            continue
        t = df[time_col].values.astype(float)
        v = df[value_col].values.astype(float)
        if len(t) < 2:
            continue
        # Trier + dédupliquer
        idx = np.argsort(t)
        t, v = t[idx], v[idx]
        _, uniq = np.unique(t, return_index=True)
        t, v = t[uniq], v[uniq]
        v_interp = np.interp(t_grid, t, v,
                             left=v[0], right=v[-1])
        interpolated.append(v_interp)

    if not interpolated:
        return t_grid, None, None, None, None

    arr  = np.array(interpolated)          # shape: (n_seeds, n_pts)
    mean = arr.mean(axis=0)
    std  = arr.std(axis=0, ddof=1)
    n    = arr.shape[0]

    if n > 1:
        # IC 95% Student (n petit) ou z (n >= 30)
        if n < 30:
            t_crit = scipy_stats.t.ppf(0.975, df=n-1)
        else:
            t_crit = CI_Z
        margin = t_crit * std / np.sqrt(n)
    else:
        margin = np.zeros_like(std)

    ci_low  = mean - margin
    ci_high = mean + margin
    return t_grid, mean, ci_low, ci_high, std


def smooth(arr, window):
    """Lissage par moyenne glissante centree (mode 'same')."""
    if window <= 1 or arr is None:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode='same')


def align_series_clipped(dfs, time_col, value_col,
                         t_min=0, t_max=3500, n_pts=70,
                         clip_min=None, clip_max=None,
                         smooth_window=1):
    """
    Wrapper autour de align_series qui :
      - clippe les bandes IC95 dans [clip_min, clip_max]
      - applique un lissage par fenetre glissante (smooth_window > 1)
    Utile pour le PDR par round, tres bruité en fin de simulation
    (quelques paquets par fenetre de 50 s), alors que le PDR cumule
    reste stable autour de 95-99%.
    """
    t, mean, lo, hi, std = align_series(dfs, time_col, value_col,
                                         t_min, t_max, n_pts)
    if mean is None:
        return t, None, None, None, None
    # Lissage avant clipping
    mean = smooth(mean, smooth_window)
    lo   = smooth(lo,   smooth_window)
    hi   = smooth(hi,   smooth_window)
    if clip_min is not None:
        mean = np.clip(mean, clip_min, None)
        if lo is not None:
            lo = np.clip(lo, clip_min, None)
    if clip_max is not None:
        mean = np.clip(mean, None, clip_max)
        if hi is not None:
            hi = np.clip(hi, None, clip_max)
    return t, mean, lo, hi, std


# ─── Graphes ──────────────────────────────────────────────────────────────────

def detect_columns(dfs_by_proto: dict):
    """
    Detecte les colonnes disponibles POUR CHAQUE PROTOCOLE separement.
    Retourne {proto: {key: col_name}} au lieu d'un mapping global unique.
    Sans cette correction, si FDQN utilise 'AliveNodes' et LEACH 'alive_nodes',
    seul le premier trouve est retenu et les courbes baselines disparaissent.
    """
    candidates = {
        "time":   ["Time_s", "time_s", "time", "Round"],
        "alive":  ["AliveNodes", "alive_nodes", "Alive"],
        "energy": ["TotalDrained_J", "EnergyConsumed_J", "energy_consumed_J",
                   "energy_J", "EnergyTotal_J"],
        # PDR_RL_pct = cumulatif (affiché dans les logs : PDR_RL=99.3%)
        # PDR_RL_round_pct = per-round seulement → bruité, à eviter
        "pdr":    ["PDR_RL_pct", "PDR_pct", "pdr_pct",
                   "PDR_Global_pct", "PDR_NS3_pct",
                   "PDR_RL_round_pct", "pdr"],
        "delay":  ["AvgDelay_ms", "delay_ms", "Delay_ms", "avg_delay_ms"],
    }
    per_proto = {}
    for proto, dfs in dfs_by_proto.items():
        col_map = {}
        for key, cands in candidates.items():
            for df in dfs:
                for c in cands:
                    if c in df.columns:
                        col_map[key] = c
                        break
                if key in col_map:
                    break
        per_proto[proto] = col_map
    return per_proto


def plot_alive_nodes(series, outdir, col_map_per_proto, n_nodes=300):
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    ax.axhline(n_nodes * 0.5, color="gray", lw=1, ls=":", alpha=0.6,
               label="HND (50%)")

    n_seeds_total = 0
    for proto, (label, color, ls, mk, lw) in PROTOCOLS.items():
        dfs = series.get(proto, [])
        if not dfs:
            continue
        col_map = col_map_per_proto.get(proto, {})
        t_col = col_map.get("time", "Time_s")
        v_col = col_map.get("alive", "AliveNodes")
        t, mean, lo, hi, _ = align_series_clipped(dfs, t_col, v_col,
                                                    clip_min=0, clip_max=n_nodes)
        if mean is None:
            continue
        n_seeds_total = max(n_seeds_total, len(dfs))
        ax.plot(t, mean, color=color, ls=ls, lw=lw,
                marker=mk, markevery=8, markersize=5, label=label)
        if lo is not None:
            ax.fill_between(t, lo, hi, color=color, alpha=ALPHA_BAND)

    n_seeds = n_seeds_total or 1
    ax.set_title(f"Nombre de nœuds vivants au cours du temps\n"
                 f"(moyenne ± IC95%, N={n_nodes}, {n_seeds} seeds)")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Nœuds vivants")
    ax.set_xlim(0, 3500)
    ax.set_ylim(0, n_nodes + 10)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "01_alive_nodes_multiseed.png", dpi=150)
    plt.close(fig)
    print("  ✓ 01_alive_nodes_multiseed.png")


def plot_energy(series, outdir, col_map_per_proto):
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    n_seeds_total = 0
    for proto, (label, color, ls, mk, lw) in PROTOCOLS.items():
        dfs = series.get(proto, [])
        if not dfs:
            continue
        col_map = col_map_per_proto.get(proto, {})
        t_col = col_map.get("time", "Time_s")
        v_col = col_map.get("energy", "TotalDrained_J")
        t, mean, lo, hi, _ = align_series_clipped(dfs, t_col, v_col, clip_min=0)
        if mean is None:
            continue
        n_seeds_total = max(n_seeds_total, len(dfs))
        ax.plot(t, mean, color=color, ls=ls, lw=lw,
                marker=mk, markevery=8, markersize=5, label=label)
        if lo is not None:
            ax.fill_between(t, lo, hi, color=color, alpha=ALPHA_BAND)

    n_seeds = n_seeds_total or 1
    ax.set_title(f"Énergie totale consommée au cours du temps\n"
                 f"(moyenne ± IC95%, {n_seeds} seeds)")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Énergie consommée (J)")
    ax.set_xlim(0, 3500)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "02_energy_multiseed.png", dpi=150)
    plt.close(fig)
    print("  ✓ 02_energy_multiseed.png")


def plot_pdr(series, outdir, col_map_per_proto, scalars=None):
    """
    Trace le PDR lisse par round avec IC95%.

    PROBLEME ORIGINAL :
      La colonne PDR_RL_round_pct est calculee PER-ROUND (paquets de la
      fenetre de 50 s). En fin de vie du reseau, quand peu de noeuds
      transmettent, quelques pertes font chuter le per-round a 60-70 %,
      alors que le PDR CUMULATIF reste >95 %. Le graphe brut donnait
      une impression fausse d'effondrement du protocole.

    CORRECTION :
      1. Lissage par fenetre glissante (smooth_window=5 pts ~ 250 s)
         pour attenuer le bruit de fin de reseau.
      2. Annotation du PDR cumulatif reel (depuis summary scalaires)
         pour que le lecteur voie la valeur de reference.
      3. Clip strict [0, 100].
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    ax.axhline(97, color="#2196F3", lw=1.2, ls=":",
               label="97% objectif")

    # Fenetre de lissage : 5 pts x 50 s/pt = ~250 s -> attenuation du bruit
    # sans effacer la tendance reelle
    PDR_SMOOTH = 5

    n_seeds_total = 0
    for proto, (label, color, ls, mk, lw) in PROTOCOLS.items():
        dfs = series.get(proto, [])
        if not dfs:
            continue
        col_map = col_map_per_proto.get(proto, {})
        t_col = col_map.get("time", "Time_s")
        v_col = col_map.get("pdr", "PDR_RL_round_pct")
        # PDR_RL_pct est cumulatif → deja lisse par nature, pas besoin de fenetre
        t, mean, lo, hi, _ = align_series_clipped(
            dfs, t_col, v_col,
            clip_min=0, clip_max=100,
            smooth_window=1)
        if mean is None:
            continue
        n_seeds_total = max(n_seeds_total, len(dfs))
        ax.plot(t, mean, color=color, ls=ls, lw=lw,
                marker=mk, markevery=8, markersize=5, label=label)
        if lo is not None:
            ax.fill_between(t, lo, hi, color=color, alpha=ALPHA_BAND)

        # Annotation : PDR cumulatif global depuis le summary
        if scalars:
            sc = scalars.get(proto, {})
            cum_vals = sc.get("PDR_RL_pct", sc.get("PDR_Global_pct", []))
            if cum_vals:
                cum_mean = float(np.mean(cum_vals))
                ax.axhline(cum_mean, color=color, lw=0.8, ls="--", alpha=0.5)
                ax.text(3520, cum_mean, f"{cum_mean:.1f}%",
                        color=color, fontsize=7, va="center")

    n_seeds = n_seeds_total or 1
    ax.set_title(f"PDR — taux de livraison par round\n"
                 f"(moyenne ± IC95%, {n_seeds} seeds)")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("PDR (%)")
    ax.set_xlim(0, 3500)
    ax.set_ylim(40, 105)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "03_pdr_multiseed.png", dpi=150)
    plt.close(fig)
    print("  ✓ 03_pdr_multiseed.png")


def plot_delay(series, outdir, col_map_per_proto):
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    n_seeds_total = 0
    for proto, (label, color, ls, mk, lw) in PROTOCOLS.items():
        dfs = series.get(proto, [])
        if not dfs:
            continue
        col_map = col_map_per_proto.get(proto, {})
        t_col = col_map.get("time", "Time_s")
        v_col = col_map.get("delay", "AvgDelay_ms")
        t, mean, lo, hi, _ = align_series_clipped(dfs, t_col, v_col, clip_min=0)
        if mean is None:
            continue
        n_seeds_total = max(n_seeds_total, len(dfs))
        ax.plot(t, mean, color=color, ls=ls, lw=lw,
                marker=mk, markevery=8, markersize=5, label=label)
        if lo is not None:
            ax.fill_between(t, lo, hi, color=color, alpha=ALPHA_BAND)

    n_seeds = n_seeds_total or 1
    ax.set_title(f"Délai moyen bout-en-bout au cours du temps\n"
                 f"(FlowMonitor NS-3 réel, moyenne ± IC95%, {n_seeds} seeds)")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Délai moyen (ms)")
    ax.set_xlim(0, 3500)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "04_delay_multiseed.png", dpi=150)
    plt.close(fig)
    print("  ✓ 04_delay_multiseed.png")


def plot_dashboard(series, outdir, col_map_per_proto, n_nodes=300):
    """Dashboard 2x2 avec IC95%."""
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_DASH)
    n_seeds_all = max((len(v) for v in series.values() if v), default=1)
    fig.suptitle(
        f"Evaluation comparative — Moyenne +- IC95% sur {n_seeds_all} seeds\n"
        f"N={n_nodes} noeuds, E_init=1.2J, Zone=1000x1000m",
        fontsize=13, fontweight="bold"
    )

    # (axe, cle_metrique, titre, ylabel, xlim, ylim, clip_min, clip_max)
    plot_specs = [
        (axes[0, 0], "alive",  "Noeuds vivants",
         "Noeuds vivants",        (0, 3500), (0, n_nodes+10),  0,    n_nodes),
        (axes[0, 1], "energy", "Energie consommee",
         "Energie consommee (J)", (0, 3500), None,              0,    None),
        (axes[1, 0], "pdr",    "PDR (par round)",
         "PDR (%)",               (0, 3500), (40, 105),          0,    100),
        (axes[1, 1], "delay",  "Delai moyen (FlowMonitor)",
         "Delai moyen (ms)",      (0, 3500), None,              0,    None),
    ]

    handles = []

    for ax, metric_key, title, ylabel, xlim, ylim, cmin, cmax in plot_specs:
        if metric_key == "pdr":
            ax.axhline(97, color="#2196F3", lw=1, ls=":", alpha=0.8,
                       label="97% objectif")
        if metric_key == "alive":
            ax.axhline(n_nodes * 0.5, color="gray", lw=1, ls=":", alpha=0.6,
                       label="HND (50%)")

        for proto, (label, color, ls, mk, lw) in PROTOCOLS.items():
            dfs = series.get(proto, [])
            if not dfs:
                continue
            col_map = col_map_per_proto.get(proto, {})
            t_col = col_map.get("time", "Time_s")
            v_col = col_map.get(metric_key, metric_key)
            # PDR cumulatif : pas de lissage necessaire
            sw = 1
            t, mean, lo, hi, _ = align_series_clipped(
                dfs, t_col, v_col, clip_min=cmin, clip_max=cmax,
                smooth_window=sw)
            if mean is None:
                continue
            line, = ax.plot(t, mean, color=color, ls=ls, lw=lw,
                            marker=mk, markevery=8, markersize=4, label=label)
            if lo is not None:
                ax.fill_between(t, lo, hi, color=color, alpha=ALPHA_BAND)
            if metric_key == "alive":
                handles.append(line)

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Temps (s)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(*xlim)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)

    # Légende commune sous les graphes
    patches = [
        mpatches.Patch(color=color, label=label)
        for proto, (label, color, ls, mk, lw) in PROTOCOLS.items()
    ]
    fig.legend(handles=patches, loc="lower center",
               ncol=len(PROTOCOLS), fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(outdir / "00_dashboard_multiseed.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 00_dashboard_multiseed.png")


# ─── Tableau agrégat scalaires ────────────────────────────────────────────────

# SCALAR_MAP : {cle_canonique: (nice_key, clip_lo, clip_hi, [aliases...])}
# Les aliases permettent de lire les CSV LEACH/Q-Routing qui utilisent
# des noms differents (FND_s, HND_s, PDR_Stable_pct, etc.)
# SCALAR_MAP : {cle_canonique: (nice_key, clip_lo, clip_hi, [aliases...])}
#
# Gestion des deux formats de clés :
#   Format "fdqnte" (FDQN_TEplus, DQN_LEACH) : FND_t, HND_t, LND_t,
#                    PDR_RL_preFND_pct, PDR_RL_pct, EnergyTotalConsumed_J
#   Format "eval"   (LEACH, HEED, QRouting)  : FND_s, HND_s, LND_s,
#                    PDR_Stable_pct, PDR_Global_pct, TotalEnergyConsumed_J
#
# Les aliases permettent à export_aggregate_summary() de trouver la valeur
# quelle que soit la convention de nommage du protocole.
SCALAR_MAP = {
    "FND_t":             ("FND (s)",               0, None,
                          ["FND_s", "fnd_t", "FND"]),
    "HND_t":             ("HND (s)",               0, None,
                          ["HND_s", "hnd_t", "HND"]),
    "LND_t":             ("LND-90% (s)",           0, None,
                          ["LND_s", "lnd_t", "LND"]),
    "PDR_RL_preFND_pct": ("PDR stable pre-FND (%)", 0, 100,
                          ["PDR_Stable_pct", "PDR_preFND_pct",
                           "PDR_RL_preFND_pct"]),
    "PDR_RL_pct":        ("PDR global RL (%)",      0, 100,
                          ["PDR_Global_pct", "PDR_global_pct",
                           "PDR_RL_pct"]),
    "PDR_NS3_pct":       ("PDR NS-3 (%)",           0, 100,
                          ["PDR_NS3", "pdr_ns3_pct"]),
    "AvgDelay_ms":       ("Delai moyen (ms)",       0, None,
                          ["avg_delay_ms", "Delay_ms"]),
    "EnergyTotalConsumed_J": ("Energie totale (J)", 0, None,
                          ["TotalEnergyConsumed_J",
                           "energy_total_J",
                           "EnergyConsumed_J"]),
}

def export_aggregate_summary(scalars, outdir, seeds):
    rows = []
    for proto, (label, *_) in PROTOCOLS.items():
        sc = scalars.get(proto, {})
        row = {"Protocole": label, "N_seeds": len(seeds)}
        for raw_key, (nice_key, clip_lo, clip_hi, *aliases_list) in SCALAR_MAP.items():
            aliases = aliases_list[0] if aliases_list else []
            # Cherche d'abord la cle canonique, puis les aliases
            vals = sc.get(raw_key, [])
            if not vals:
                for alias in aliases:
                    vals = sc.get(alias, [])
                    if vals:
                        break
            if vals:
                arr = np.array(vals)
                mean = arr.mean()
                std  = arr.std(ddof=1) if len(arr) > 1 else 0.0
                n    = len(arr)
                t_crit = scipy_stats.t.ppf(0.975, df=max(n-1, 1))
                margin = t_crit * std / np.sqrt(n) if n > 1 else 0.0
                ci_low  = mean - margin
                ci_high = mean + margin
                # FIX : clipping physique — evite temps negatif, PDR > 100 %, etc.
                if clip_lo is not None:
                    ci_low  = max(ci_low,  clip_lo)
                    mean    = max(mean,     clip_lo)
                if clip_hi is not None:
                    ci_high = min(ci_high, clip_hi)
                    mean    = min(mean,    clip_hi)
                row[f"{nice_key} — moy"]       = round(mean,    3)
                row[f"{nice_key} — std"]        = round(std,     3)
                row[f"{nice_key} — IC95_low"]   = round(ci_low,  3)
                row[f"{nice_key} — IC95_high"]  = round(ci_high, 3)
            else:
                row[f"{nice_key} — moy"]       = "N/A"
                row[f"{nice_key} — std"]        = "N/A"
                row[f"{nice_key} — IC95_low"]   = "N/A"
                row[f"{nice_key} — IC95_high"]  = "N/A"
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = outdir / "aggregate_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  OK aggregate_summary.csv ({len(df)} protocoles, {len(seeds)} seeds)")

    # Affichage console compact - cles mises a jour apres renommage SCALAR_MAP
    print("\n  Tableau recapitulatif:")
    key_metrics = ["FND (s) --- moy", "HND (s) --- moy",
                   "PDR stable pre-FND (%) --- moy",
                   "Delai moyen (ms) --- moy"]
    # Essaie aussi avec tirets longs si presents
    for _, row in df.iterrows():
        print(f"  {row['Protocole']:<32}", end="")
        for k in list(row.index):
            if "FND (s)" in k and "moy" in k and "IC" not in k and "std" not in k:
                print(f"  FND={row[k]}", end="")
            if "HND (s)" in k and "moy" in k and "IC" not in k and "std" not in k:
                print(f"  HND={row[k]}", end="")
            if "Delai" in k and "moy" in k and "IC" not in k and "std" not in k:
                print(f"  Delay={row[k]}", end="")
        print()

    return df



# ─── Graphe barres multiseed (07_multiseed.png) ───────────────────────────────

def plot_multiseed_bars(scalars, outdir, seeds):
    """
    Reproduit le graphe 07_multiseed.png : barres FND et HND avec IC95%
    pour tous les protocoles (LEACH, HEED, Q-Routing, DQN-LEACH, FDQN-TE+).

    Ordre d'affichage fixe pour cohérence avec l'article :
      LEACH → HEED → Q-Routing → DQN-LEACH → FDQN-TE+
    """
    # Ordre d'affichage et couleurs (indépendant de l'ordre de PROTOCOLS)
    DISPLAY_ORDER = [
        ("LEACH",       "LEACH",                  "#e05252"),
        ("HEED",        "HEED",                   "#9b59b6"),
        ("QRouting",    "Q-Routing",               "#c07820"),
        ("DQN_LEACH",   "DQN-LEACH (baseline DRL)","#2980b9"),
        ("FDQN_TEplus",  "FDQN-TE+ (proposé)",     "#1a7a4a"),
    ]

    # Clés FND/HND avec aliases pour les deux formats
    FND_KEYS = ["FND_t", "FND_s"]
    HND_KEYS = ["HND_t", "HND_s"]

    def get_stat(sc, keys):
        """Retourne (mean, ci_margin, n) depuis les aliases de clés."""
        for k in keys:
            vals = sc.get(k, [])
            if vals:
                arr = np.array([v for v in vals if v and v > 0])
                if len(arr) == 0:
                    return None, 0, 0
                mean = arr.mean()
                std  = arr.std(ddof=1) if len(arr) > 1 else 0.0
                n    = len(arr)
                t_c  = scipy_stats.t.ppf(0.975, df=max(n-1, 1))
                margin = t_c * std / np.sqrt(n) if n > 1 else 0.0
                return mean, margin, n
        return None, 0, 0

    fig, (ax_fnd, ax_hnd) = plt.subplots(1, 2, figsize=(14, 6))
    n_seeds = len(seeds)

    for ax, metric_keys, title in [
        (ax_fnd, FND_KEYS, "First Node Death (FND)"),
        (ax_hnd, HND_KEYS, "Half Node Death (HND)"),
    ]:
        x_labels = []
        means    = []
        margins  = []
        colors   = []
        ns       = []

        for proto_key, label, color in DISPLAY_ORDER:
            sc = scalars.get(proto_key, {})
            mean, margin, n = get_stat(sc, metric_keys)
            if mean is None or mean == 0:
                mean, margin, n = None, 0, 0
            x_labels.append(label)
            means.append(mean)
            margins.append(margin)
            colors.append(color)
            ns.append(n)

        x = np.arange(len(x_labels))
        bars = ax.bar(x, [m if m else 0 for m in means],
                      color=colors, alpha=0.85, width=0.6, zorder=3)

        # Barres d'erreur IC95%
        for i, (m, mg) in enumerate(zip(means, margins)):
            if m and mg > 0:
                ax.errorbar(x[i], m, yerr=mg,
                            fmt="none", color="black",
                            capsize=5, capthick=1.5, linewidth=1.5, zorder=4)

        # Annotations valeur ± CI
        for i, (m, mg, n) in enumerate(zip(means, margins, ns)):
            if m:
                ci_str = f"±{mg:.0f}" if mg > 0 else "±0"
                ax.text(x[i], m + max(m * 0.02, 20),
                        f"{m:.0f}s\n{ci_str}",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")
            else:
                ax.text(x[i], 50, "N/A", ha="center", va="bottom",
                        fontsize=8, color="gray")

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel("Durée (s)")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=15, ha="right", fontsize=9)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.set_axisbelow(True)

    n_available = max(ns) if ns else 0
    fig.suptitle(
        f"Reproductibilité — {n_available} seeds × {len(DISPLAY_ORDER)} modèles\n"
        f"Barres d'erreur : intervalle de confiance 95%",
        fontsize=13, fontweight="bold"
    )
    note = (f"N seeds disponibles : {n_available} (objectif : {n_seeds}) | "
            f"Si un seul seed disponible, CI=0 (pas d'erreur calculable)")
    fig.text(0.5, -0.02, note, ha="center", fontsize=8, color="gray",
             style="italic")

    fig.tight_layout()
    fig.savefig(outdir / "07_multiseed_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 07_multiseed_bars.png")


# ─── Graphe scalabilité (08_scalability.png) ──────────────────────────────────

def plot_scalability(results_dir: Path, outdir: Path, seeds: list,
                     n_nodes_list: list = [50, 100, 200, 300],
                     n_nodes_main: int = 300):
    """
    Trace FND et HND en fonction de N pour tous les protocoles.

    Stratégie de recherche des données (par ordre de priorité) :
      1. results_dir/scale_N{n}/{proto}/seed_{s}/ — données de scalabilité dédiées
      2. results_dir/{proto}/seed_{s}/            — données N=300 existantes (fallback)

    Si une valeur N n'a pas de dossier scale_N{n}/ mais que N correspond à
    --n_nodes (typiquement 300), les données principales sont utilisées.
    Les N sans aucune donnée sont omis de la courbe (pas de point vide).
    """
    DISPLAY_ORDER = [
        ("LEACH",       "LEACH",                   "#e05252", "o", "--"),
        ("HEED",        "HEED",                    "#9b59b6", "D", "--"),
        ("QRouting",    "Q-Routing",                "#c07820", "s", "-."),
        ("DQN_LEACH",   "DQN-LEACH (baseline DRL)", "#2980b9", "v", "-."),
        ("FDQN_TEplus",  "FDQN-TE+ (proposé)",      "#1a7a4a", "^", "-"),
    ]
    FND_KEYS = ["FND_t", "FND_s"]
    HND_KEYS = ["HND_t", "HND_s"]

    def read_proto_seeds(n, proto_key):
        """
        Retourne la liste des valeurs FND pour (proto, N) en cherchant :
          1. results_dir/scale_N{n}/{proto_dir}/seed_{s}/
          2. results_dir/{proto_dir}/seed_{s}/  (seulement si n == n_nodes_list[-1])
        """
        proto_dir = PROTO_DIRS.get(proto_key, proto_key)
        fmt       = PROTO_SUMMARY_FORMAT.get(proto_key, "eval")
        fnames    = (["fdqnte_summary.csv", "summary.csv"] if fmt == "fdqnte"
                     else ["summary.csv", "fdqnte_summary.csv"])
        vals_fnd, vals_hnd = [], []

        for seed in seeds:
            # Candidats : dossier scale dédié d'abord, puis dossier direct {proto}/seed_{s}/
            # Le fallback est toujours ajouté (pas seulement pour n == n_nodes_main)
            # afin de suivre la structure réelle : results_dir/{proto}/seed_{s}/
            candidates = [
                results_dir / f"scale_N{n}" / proto_dir / f"seed_{seed}",
                results_dir / proto_dir / f"seed_{seed}",
            ]

            for spath in candidates:
                for fname in fnames:
                    fp = spath / fname
                    if fp.exists():
                        s = read_summary(fp)
                        for k in FND_KEYS:
                            v = s.get(k)
                            if v is not None:
                                try:
                                    fv = float(v)
                                    if fv > 0:
                                        vals_fnd.append(fv)
                                except (ValueError, TypeError):
                                    pass
                        for k in HND_KEYS:
                            v = s.get(k)
                            if v is not None:
                                try:
                                    hv = float(v)
                                    if hv > 0:
                                        vals_hnd.append(hv)
                                except (ValueError, TypeError):
                                    pass
                        break   # found a summary file for this seed
                else:
                    continue
                break           # found a directory for this seed

        return vals_fnd, vals_hnd

    def agg(vals):
        """Retourne (mean, ci_margin) avec IC95% Student."""
        if not vals:
            return None, 0
        arr = np.array(vals)
        mean = arr.mean()
        n_s  = len(arr)
        std  = arr.std(ddof=1) if n_s > 1 else 0.0
        t_c  = scipy_stats.t.ppf(0.975, df=max(n_s - 1, 1))
        ci   = t_c * std / np.sqrt(n_s) if n_s > 1 else 0.0
        return mean, ci

    # ── Collecter les données disponibles ──────────────────────────────────────
    data = {}   # {proto_key: {n: (fnd_mean, fnd_ci, hnd_mean, hnd_ci)}}
    any_data = False

    for proto_key, *_ in DISPLAY_ORDER:
        data[proto_key] = {}
        for n in n_nodes_list:
            fnd_vals, hnd_vals = read_proto_seeds(n, proto_key)
            if fnd_vals:
                fm, fc = agg(fnd_vals)
                hm, hc = agg(hnd_vals)
                data[proto_key][n] = (fm, fc, hm, hc)
                any_data = True

    # ── Si vraiment aucune donnée nulle part ───────────────────────────────────
    if not any_data:
        missing_ns = [n for n in n_nodes_list if n != n_nodes_list[-1]]
        cmds_lines = ["Lancer les simulations de scalabilité :"]
        for n in missing_ns:
            for sim, d in [("leach_sim","LEACH"),("heed_sim","HEED"),
                           ("qrouting_sim","QRouting"), ("fdqn_leach","DQN_LEACH")]:
                cmds_lines.append(
                    f"./ns3 run scratch/{sim} --nNodes={n} "
                    f"--resultsDir=results_eval/scale_N{n}/{d}/seed_42 --seed=42")
        cmds_lines.append("...")

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_facecolor("#f8f8f8")
        ax.text(0.5, 0.72, "Aucune donnée de scalabilité disponible",
                transform=ax.transAxes, ha="center", fontsize=14,
                fontweight="bold", color="#555")
        ax.text(0.5, 0.38, "\n".join(cmds_lines),
                transform=ax.transAxes, ha="center", fontsize=7.5,
                color="#777", family="monospace",
                bbox=dict(boxstyle="round", fc="white", ec="#ccc"))
        ax.axis("off")
        n_str = "/".join(str(n) for n in n_nodes_list)
        ax.set_title(f"Scalabilité (N={n_str}) — données manquantes", fontsize=12)
        fig.tight_layout()
        fig.savefig(outdir / "08_scalability.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  ✓ 08_scalability.png (données manquantes)")
        return

    # ── Tracer FND et HND côte à côte ─────────────────────────────────────────
    fig, (ax_fnd, ax_hnd) = plt.subplots(1, 2, figsize=(15, 6), sharey=False)

    missing_ns = [n for n in n_nodes_list
                  if not any(n in data[pk] for pk, *_ in DISPLAY_ORDER)]

    for ax, metric_idx, ylabel, title in [
        (ax_fnd, 0, "FND (s)", "Scalabilité — First Node Death"),
        (ax_hnd, 2, "HND (s)", "Scalabilité — Half Node Death"),
    ]:
        for proto_key, label, color, mk, ls in DISPLAY_ORDER:
            xs, ys, es = [], [], []
            for n in n_nodes_list:
                entry = data[proto_key].get(n)
                if entry and entry[metric_idx] is not None:
                    xs.append(n)
                    ys.append(entry[metric_idx])
                    es.append(entry[metric_idx + 1])

            if not xs:
                continue

            eb = ax.errorbar(xs, ys, yerr=es,
                             color=color, marker=mk, linestyle=ls, lw=1.8,
                             capsize=5, capthick=1.5, markersize=7,
                             label=label, zorder=3)

            # Annotations valeur sur chaque point
            for x, y in zip(xs, ys):
                ax.annotate(f"{y:.0f}", xy=(x, y),
                            xytext=(0, 8), textcoords="offset points",
                            ha="center", fontsize=7, color=color)

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Nombre de nœuds N", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(n_nodes_list)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3, zorder=0)
        ax.set_axisbelow(True)

        # Note si certains N manquent
        if missing_ns:
            note = f"Données disponibles seulement pour N∈{{{','.join(str(n) for n in n_nodes_list if n not in missing_ns)}}}"
            ax.text(0.98, 0.03, note, transform=ax.transAxes,
                    ha="right", fontsize=7, color="#888", style="italic")

    n_str = "/".join(str(n) for n in n_nodes_list)
    fig.suptitle(f"Scalabilité (N={n_str}) — FND et HND vs nombre de nœuds\n"
                 f"(moyenne ± IC95%, seeds={seeds})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outdir / "08_scalability.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 08_scalability.png")

def main():
    parser = argparse.ArgumentParser(
        description="Agrégation multi-seeds et graphes IC95%"
    )
    parser.add_argument("--results_dir", default="results_eval",
                        help="Dossier racine contenant les seeds")
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[42, 43, 44, 45, 46],
                        help="Liste des seeds (ex: 42 43 44 45 46)")
    parser.add_argument("--n_nodes", type=int, default=300)
    parser.add_argument("--n_nodes_scale", nargs="+", type=int,
                        default=[50, 100, 200, 300],
                        help="Valeurs de N pour le graphe scalabilité")
    parser.add_argument("--outdir", default="results_eval/figures_multiseed",
                        help="Dossier de sortie des graphes")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    outdir      = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Agrégation multi-seeds FDQN-TE+ vs baselines")
    print(f"  Seeds       : {args.seeds}")
    print(f"  Protocoles  : {list(PROTOCOLS.keys())}")
    print(f"  Dossiers    : { {p: PROTO_DIRS.get(p,p) for p in PROTOCOLS} }")
    print(f"  Résultats   : {results_dir}")
    print(f"  Sortie      : {outdir}")
    print(f"{'='*60}\n")

    # ── Vérification disponibilité des données ────────────────────────────────
    available_seeds = []
    for seed in args.seeds:
        found = False
        for proto in PROTOCOLS:
            proto_dir = PROTO_DIRS.get(proto, proto)
            p = find_seed_dir(results_dir, proto_dir, seed)
            if p is not None:
                found = True
                break
        if found:
            available_seeds.append(seed)
        else:
            print(f"  [WARN] Aucune donnée pour seed={seed} → ignoré")

    if not available_seeds:
        print("[ERROR] Aucune donnée disponible. Lancer d'abord run_multiseed.sh")
        sys.exit(1)

    print(f"  Seeds disponibles : {available_seeds} ({len(available_seeds)}/{len(args.seeds)})")

    # ── Collecte ──────────────────────────────────────────────────────────────
    print("\n  Lecture des données...")
    scalars, series = collect_all_data(results_dir, available_seeds, PROTOCOLS)

    # FIX : detection par protocole (evite qu'un seul nom de colonne soit retenu
    # pour tous les protocoles — les baselines disparaissaient des graphes)
    col_map_per_proto = detect_columns(series)
    for p, cm in col_map_per_proto.items():
        print(f"  Colonnes [{p}] : {cm}")

    # ── Graphes ───────────────────────────────────────────────────────────────
    print("\n  Generation des graphes...")
    plot_alive_nodes(series, outdir, col_map_per_proto, n_nodes=args.n_nodes)
    plot_energy(series, outdir, col_map_per_proto)
    plot_pdr(series, outdir, col_map_per_proto, scalars=scalars)
    plot_delay(series, outdir, col_map_per_proto)
    plot_dashboard(series, outdir, col_map_per_proto, n_nodes=args.n_nodes)
    plot_multiseed_bars(scalars, outdir, available_seeds)
    plot_scalability(results_dir, outdir, available_seeds,
                     n_nodes_list=args.n_nodes_scale,
                     n_nodes_main=args.n_nodes)

    # ── Tableau agrégat ───────────────────────────────────────────────────────
    print("\n  Export du tableau agrégat...")
    export_aggregate_summary(scalars, outdir, available_seeds)

    print(f"\n  Tous les fichiers sont dans : {outdir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
