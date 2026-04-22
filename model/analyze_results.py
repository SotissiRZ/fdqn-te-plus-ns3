#!/usr/bin/env python3
"""
analyze_results.py — Analyse comparative FDQN-TE+
===================================================================

NOMS DE COLONNES CSV EXACTS (vérifiés dans les sources C++) :

  metrics.csv  (LEACH, QRouting — via eval_common.h::WriteMetricsRow) :
    Round, Time_s, AliveNodes, DeadNodes, EnergyMean_J, EnergyConsumed_J,
    PDR_pct, Delay_ms, PktEmitted, PktDelivered, NClusters, PEPM_AtRisk,
    EnergyGini, IsolatedNodes

  summary.csv  (LEACH, QRouting — via eval_common.h::WriteSummaryCSV) :
    Clés Metric/Value : Model, FND_s, HND_s, LND_s, PDR_Stable_pct,
    PDR_Global_pct, AvgDelay_ms, TotalEnergyConsumed_J, AvgEnergyGini,
    TotalPktSent, TotalPktRecv, TotalRounds

  fdqnte_energy.csv  (variantes DQN — via InitEnergyCSV dans .cc) :
    Round, Time_s, AliveNodes, DeadNodes, EnergyMean_J, EnergyStdDev_J,
    EnergyMin_J, EnergyMax_J, TotalDrained_J, PDR_RL_pct, PDR_RL_round_pct,
    PDR_NS3_pct, AvgDelay_ms, AtRiskPEPM, PEPMRiskMean, FND_s, HND_s,
    LND_s, RLSteps, FedRound, IFORound, NClusters, RL_PktEmitted,
    RL_PktDelivered, TotalEnergy_J

  fdqnte_summary.csv  (variantes DQN) :
    Clés Param/Value : N, AliveNodes, DeadNodes, EnergyMean_J,
    EnergyTotalConsumed_J, PDR_RL_pct, PDR_RL_preFND_pct, PDR_NS3_pct,
    AvgDelay_ms, TxPackets, RxPackets, RL_PktEmitted, RL_PktDelivered,
    IFO_Rounds, FND_t, HND_t, LND_t, RL_Steps, FedRounds, Seed,
    SimDuration_s, RadioRange_m, AreaSize_m, InitEnergy_J

PDR utilisé pour les graphiques temporels :
  - LEACH/QRouting : colonne PDR_pct (delta par round via pdrWindow)
  - DQN variants   : colonne PDR_RL_round_pct (delta par round)

PDR stable (summary) :
  - LEACH/QRouting : clé PDR_Stable_pct (gelé au FND dans les .cc)
  - DQN variants   : clé PDR_RL_preFND_pct (gelé au FND dans les .cc)
  → Ces valeurs sont calculées dans les .cc, pas recalculées ici.

Délai :
  - LEACH  : EvalDelay::ComputeDelay_ms (modèle IEEE 802.15.4 CSMA/CA)
  - QRoute : EvalDelay::ComputeDelay_ms (idem)
  - DQN    : ComputeAverageDelay(FlowMonitor) — mesuré NS-3

Aucune interpolation, aucune valeur synthétique.

Usage :
  python3 analyze_results.py --results_dir results_eval/
"""

import os
import sys
import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# ─── Styles ───────────────────────────────────────────────────────────────────

MODEL_STYLES = {
    "LEACH":      {"color": "#E24B4A", "ls": "--",  "marker": "o", "lw": 2.0},
    "HEED":       {"color": "#EE8D82", "ls": ":",   "marker": "v", "lw": 2.0},
    "QRouting":   {"color": "#BA7517", "ls": "-.",  "marker": "s", "lw": 2.0},
    "DQN_LEACH":  {"color": "#69ADDA", "ls": ":",   "marker": "h", "lw": 2.0},
    "DQN_noPEPM": {"color": "#185FA5", "ls": ":",   "marker": "^", "lw": 2.0},
    "DQN_noFed":  {"color": "#8B4DB8", "ls": "--",  "marker": "D", "lw": 2.0},
    "DQN_noIFO":  {"color": "#D4A017", "ls": "-.",  "marker": "P", "lw": 2.2},
    "FDQN_TE+":   {"color": "#0F6E56", "ls": "-",   "marker": "*", "lw": 2.5},
}

MODEL_LABELS = {
    "LEACH":      "LEACH",
    "HEED":       "HEED",
    "QRouting":   "Q-Routing",
    "DQN_LEACH":  "DQN-LEACH (baseline DRL)",
    "DQN_noPEPM": "DQN sans PEPM",
    "DQN_noFed":  "DQN sans Fédération",
    "DQN_noIFO":  "FDQN-noIFO (ablation)",
    "FDQN_TE+":   "FDQN-TE+ (proposé)",
}

# Sous-dossiers attendus sous results_dir
EXPECTED_DIRS = {
    "LEACH":      ["LEACH"],
    "HEED":       ["HEED"],
    "QRouting":   ["QRouting", "QRouting_sim"],
    "DQN_LEACH":  ["DQN_LEACH", "DQN-LEACH"],
    "DQN_noPEPM": ["DQN_noPEPM", "DQN-noPEPM"],
    "DQN_noFed":  ["DQN_noFed",  "DQN-noFed"],
    "DQN_noIFO":  ["DQN_noIFO",  "FDQN_noIFO"],
    "FDQN_TE+":   ["FDQN_TEplus", "FDQN_TE_plus", "FDQN-TE+", "FDQN_full"],
}


# ─── Utilitaires ──────────────────────────────────────────────────────────────

def _safe_float(d: dict, *keys, default=0.0) -> float:
    for k in keys:
        try:
            v = d.get(k)
            if v is not None and str(v).strip() not in ("", "—", "nan"):
                return float(v)
        except (ValueError, TypeError):
            pass
    return default


def find_model_dir(results_dir: str, model_key: str,
                   seed: int = 42) -> Path | None:

    base = Path(results_dir)
    for name in EXPECTED_DIRS.get(model_key, [model_key]):
        p = base / name
        if not p.exists():
            continue
        # Priorite 1 : sous-dossier seed
        seed_dir = p / f"seed_{seed}"
        if seed_dir.exists():
            if (seed_dir / "metrics.csv").exists() \
               or (seed_dir / "energy" / "fdqnte_energy.csv").exists() \
               or (seed_dir / "fdqnte_summary.csv").exists() \
               or (seed_dir / "summary.csv").exists():
                return seed_dir
        # Priorite 2 : donnees a la racine du dossier modele
        if (p / "metrics.csv").exists() \
           or (p / "energy" / "fdqnte_energy.csv").exists():
            return p
    return None


# ─── Chargement ───────────────────────────────────────────────────────────────

def load_metrics_csv(path: Path) -> dict:
    """
    Charge metrics.csv (format LEACH / QRouting — eval_common.h).
    Colonnes : Round, Time_s, AliveNodes, DeadNodes, EnergyMean_J,
               EnergyConsumed_J, PDR_pct, Delay_ms, PktEmitted, PktDelivered,
               NClusters, PEPM_AtRisk, EnergyGini, IsolatedNodes
    """
    data = {
        "time": [], "alive": [], "dead": [],
        "energy_mean": [], "energy_consumed": [],
        "pdr_round": [],   # PDR_pct (delta par round via pdrWindow dans les .cc)
        "delay": [],
        "pkt_emitted": [], "pkt_delivered": [],
        "n_clusters": [], "pepm_at_risk": [],
        "energy_gini": [], "isolated": [],
    }
    with open(path) as f:
        reader = csv.DictReader(row for row in f if not row.startswith('#'))
        for row in reader:
            try:
                data["time"].append(float(row["Time_s"]))
                data["alive"].append(int(row["AliveNodes"]))
                data["dead"].append(int(row["DeadNodes"]))
                data["energy_mean"].append(float(row["EnergyMean_J"]))
                data["energy_consumed"].append(float(row["EnergyConsumed_J"]))
                data["pdr_round"].append(float(row["PDR_pct"]))
                data["delay"].append(float(row["Delay_ms"]))
                data["pkt_emitted"].append(int(row["PktEmitted"]))
                data["pkt_delivered"].append(int(row["PktDelivered"]))
                data["n_clusters"].append(int(float(row.get("NClusters", 0))))
                data["pepm_at_risk"].append(int(float(row.get("PEPM_AtRisk", 0))))
                data["energy_gini"].append(float(row.get("EnergyGini", 0.0)))
                data["isolated"].append(int(float(row.get("IsolatedNodes", 0))))
            except (ValueError, KeyError):
                continue
    return data


def load_summary_csv(path: Path) -> dict:
    """
    Charge summary.csv (format Metric,Value — eval_common.h::WriteSummaryCSV).
    Clés retournées : FND_s, HND_s, LND_s, PDR_Stable_pct, PDR_Global_pct,
                      AvgDelay_ms, TotalEnergyConsumed_J, TotalPktSent,
                      TotalPktRecv, TotalRounds, AvgEnergyGini
    """
    summary = {}
    with open(path) as f:
        reader = csv.reader(row for row in f if not row.startswith('#'))
        for row in reader:
            if len(row) >= 2:
                summary[row[0].strip()] = row[1].strip()
    return summary


def load_energy_csv(path: Path) -> dict:
    """
    Charge fdqnte_energy.csv (format variantes DQN).
    Colonnes clés : Time_s, AliveNodes, DeadNodes, EnergyMean_J,
                    TotalDrained_J, PDR_RL_round_pct, PDR_NS3_pct,
                    AvgDelay_ms, AtRiskPEPM, NClusters,
                    RL_PktEmitted, RL_PktDelivered
    """
    data = {
        "time": [], "alive": [], "dead": [],
        "energy_mean": [], "energy_consumed": [],
        "pdr_cum":   [],   # PDR_RL_pct  — cumulatif (= ce qu'on lit dans les logs)
        "pdr_round": [],   # PDR_RL_round_pct — per-round, volatile (kept for reference)
        "pdr_ns3":   [],
        "delay": [],
        "pkt_emitted": [], "pkt_delivered": [],
        "n_clusters": [], "pepm_at_risk": [],
        "energy_gini": [], "isolated": [],
    }
    with open(path) as f:
        reader = csv.DictReader(row for row in f if not row.startswith('#'))
        for row in reader:
            try:
                data["time"].append(float(row["Time_s"]))
                data["alive"].append(int(row["AliveNodes"]))
                data["dead"].append(int(row["DeadNodes"]))
                data["energy_mean"].append(float(row["EnergyMean_J"]))
                data["energy_consumed"].append(float(row["TotalDrained_J"]))
                # PDR cumulatif : PDR_RL_pct est la vraie valeur affichee dans
                # les logs ("PDR_RL=99.3%"). Si absente, fallback sur le calcul
                # depuis les compteurs bruts (fait dans plot_pdr).
                data["pdr_cum"].append(float(row.get("PDR_RL_pct", -1.0)))
                data["pdr_round"].append(float(row.get("PDR_RL_round_pct", 0.0)))
                data["pdr_ns3"].append(float(row.get("PDR_NS3_pct", 0.0)))
                data["delay"].append(float(row.get("AvgDelay_ms", 0.0)))
                data["pkt_emitted"].append(int(row.get("RL_PktEmitted", 0)))
                data["pkt_delivered"].append(int(row.get("RL_PktDelivered", 0)))
                data["n_clusters"].append(int(float(row.get("NClusters", 0))))
                data["pepm_at_risk"].append(int(float(row.get("AtRiskPEPM", 0))))
                data["energy_gini"].append(0.0)
                data["isolated"].append(0)
            except (ValueError, KeyError):
                continue
    return data


def load_fdqn_summary(path: Path) -> dict:
    """
    Charge fdqnte_summary.csv (format Param,Value — variantes DQN).
    Clés retournées : FND_s (← FND_t), HND_s (← HND_t), LND_s (← LND_t),
                      PDR_Stable_pct (← PDR_RL_preFND_pct),
                      PDR_Global_pct (← PDR_RL_pct),
                      AvgDelay_ms, TotalEnergyConsumed_J (← EnergyTotalConsumed_J),
                      TotalPktSent (← RL_PktEmitted), TotalPktRecv (← RL_PktDelivered)
    """
    raw = {}
    with open(path) as f:
        reader = csv.reader(row for row in f if not row.startswith('#'))
        for row in reader:
            if len(row) >= 2:
                raw[row[0].strip()] = row[1].strip()

    # Normaliser vers les clés standard utilisées partout dans ce script
    return {
        "Model":                 raw.get("Model", ""),
        "FND_s":                 raw.get("FND_t", "0"),
        "HND_s":                 raw.get("HND_t", "0"),
        "LND_s":                 raw.get("LND_t", "0"),
        "PDR_Stable_pct":        raw.get("PDR_RL_preFND_pct",
                                         raw.get("PDR_RL_pct", "0")),
        "PDR_Global_pct":        raw.get("PDR_RL_pct", "0"),
        "AvgDelay_ms":           raw.get("AvgDelay_ms", "0"),
        "TotalEnergyConsumed_J": raw.get("EnergyTotalConsumed_J",
                                         raw.get("TotalEnergyConsumed_J", "0")),
        "TotalPktSent":          raw.get("RL_PktEmitted",
                                         raw.get("TxPackets", "0")),
        "TotalPktRecv":          raw.get("RL_PktDelivered",
                                         raw.get("RxPackets", "0")),
        "TotalRounds":           raw.get("TotalRounds", "0"),
        "AvgEnergyGini":         "0",
    }


def load_model(results_dir: str, model_key: str) -> dict | None:
    """
    Charge toutes les données d'un modèle.
    Retourne None si le dossier ou les fichiers sont absents.

    Priorité des fichiers :
      - Variantes DQN (DQN_noPEPM, DQN_noFed, DQN_noIFO, FDQN_TE+) :
        energy/fdqnte_energy.csv  → fdqnte_summary.csv  (format FDQN)
      - LEACH / QRouting :
        metrics.csv               → summary.csv         (format eval_common.h)

    Pour les variantes DQN, metrics.csv peut exister mais être partiel
    (simulation interrompue, tests préliminaires) → on l'ignore.
    """
    # Modèles qui génèrent fdqnte_energy.csv (format FDQN)
    DQN_MODELS = {"DQN_LEACH", "DQN_noPEPM", "DQN_noFed", "DQN_noIFO", "FDQN_TE+"}

    base = find_model_dir(results_dir, model_key)
    if base is None:
        return None

    metrics_path  = base / "metrics.csv"
    energy_path   = base / "energy" / "fdqnte_energy.csv"
    summary_path  = base / "summary.csv"
    fdqn_sum_path = base / "fdqnte_summary.csv"

    data, summary = None, None

    if model_key in DQN_MODELS:
        # ── Variante DQN : priorité à fdqnte_energy.csv ──────────────────────
        if energy_path.exists():
            try:
                data = load_energy_csv(energy_path)
            except Exception as e:
                print(f"  ✗ Erreur lecture {energy_path}: {e}")
                return None
            if fdqn_sum_path.exists():
                try:
                    summary = load_fdqn_summary(fdqn_sum_path)
                except Exception as e:
                    print(f"  ✗ Erreur lecture {fdqn_sum_path}: {e}")
                    summary = {}
            elif summary_path.exists():
                try:
                    summary = load_summary_csv(summary_path)
                except Exception:
                    summary = {}
        elif metrics_path.exists():
            # Fallback metrics.csv seulement si fdqnte_energy.csv absent
            try:
                data = load_metrics_csv(metrics_path)
            except Exception as e:
                print(f"  ✗ Erreur lecture {metrics_path}: {e}")
                return None
            if summary_path.exists():
                try:
                    summary = load_summary_csv(summary_path)
                except Exception:
                    summary = {}
        else:
            return None

    else:
        # ── Baselines (LEACH, QRouting) : priorité à metrics.csv ─────────────
        if metrics_path.exists():
            try:
                data = load_metrics_csv(metrics_path)
            except Exception as e:
                print(f"  ✗ Erreur lecture {metrics_path}: {e}")
                return None
            if summary_path.exists():
                try:
                    summary = load_summary_csv(summary_path)
                except Exception as e:
                    print(f"  ✗ Erreur lecture {summary_path}: {e}")
                    summary = {}
        elif energy_path.exists():
            try:
                data = load_energy_csv(energy_path)
            except Exception as e:
                print(f"  ✗ Erreur lecture {energy_path}: {e}")
                return None
            if fdqn_sum_path.exists():
                try:
                    summary = load_fdqn_summary(fdqn_sum_path)
                except Exception:
                    summary = {}
        else:
            return None

    if not data or not data["time"]:
        return None

    summary = summary or {}

    # Compléter FND/HND depuis les données alive[] si absents ou nuls dans summary
    fnd_s = _safe_float(summary, "FND_s")
    if fnd_s < 1.0 and data["alive"]:
        n0 = data["alive"][0]
        for i, a in enumerate(data["alive"]):
            if a < n0:
                summary["FND_s"] = str(round(data["time"][i], 1))
                break

    hnd_s = _safe_float(summary, "HND_s")
    if hnd_s < 1.0 and data["alive"]:
        n0 = data["alive"][0]
        for i, a in enumerate(data["alive"]):
            if a <= n0 // 2:
                summary["HND_s"] = str(round(data["time"][i], 1))
                break

    return {
        "data":    data,
        "summary": summary,
        "model":   model_key,
        "dir":     str(base),
    }


def load_all(results_dir: str) -> dict:
    all_data = {}
    print(f"\nChargement depuis : {results_dir}")
    print("-" * 65)

    dirs_seen = {}
    for key in MODEL_STYLES:
        d = load_model(results_dir, key)
        if d is not None:
            # Détecter si deux modèles pointent sur le même dossier
            d_dir = d["dir"]
            if d_dir in dirs_seen:
                print(f"  ⚠ ATTENTION : {MODEL_LABELS[key]} et "
                      f"{MODEL_LABELS[dirs_seen[d_dir]]} → même dossier !")
                print(f"     Les courbes seront IDENTIQUES — vérifiez EXPECTED_DIRS")
            dirs_seen[d_dir] = key

            all_data[key] = d
            n   = len(d["data"]["time"])
            fnd = d["summary"].get("FND_s", "?")
            pdr = d["summary"].get("PDR_Stable_pct", "?")
            print(f"  ✓ {MODEL_LABELS[key]:<30} {n:>4} rounds | "
                  f"FND={fnd}s | PDR_stable={pdr}%")
            print(f"    └─ {d['dir']}")
        else:
            cands = [str(Path(results_dir) / c)
                     for c in EXPECTED_DIRS.get(key, [key])[:2]]
            DQN_MODELS = {"DQN_LEACH", "DQN_noPEPM", "DQN_noFed", "DQN_noIFO", "FDQN_TE+"}
            if key in DQN_MODELS:
                print(f"  ✗ {MODEL_LABELS[key]:<30} DONNÉES MANQUANTES")
                print(f"    └─ cherché : energy/fdqnte_energy.csv dans {cands[0]}")
            else:
                print(f"  ✗ {MODEL_LABELS[key]:<30} DONNÉES MANQUANTES")
                print(f"    └─ cherché : metrics.csv dans {cands[0]}")
    return all_data


# ─── Tableau comparatif ───────────────────────────────────────────────────────

def print_comparison_table(all_data: dict) -> list:
    headers = ["Modèle", "FND (s)", "HND (s)", "LND (s)",
               "PDR stable (%)", "PDR global (%)", "Délai moy (ms)",
               "E_cons (J)", "Paquets émis"]
    rows = []
    for key, d in all_data.items():
        s = d["summary"]
        rows.append([
            MODEL_LABELS[key],
            s.get("FND_s",  "—"),
            s.get("HND_s",  "—"),
            s.get("LND_s",  "—"),
            s.get("PDR_Stable_pct", "—"),
            s.get("PDR_Global_pct", s.get("PDR_pct", "—")),
            s.get("AvgDelay_ms",    "—"),
            s.get("TotalEnergyConsumed_J", "—"),
            s.get("TotalPktSent",   "—"),
        ])

    col_w = [max(len(str(r[i])) for r in [headers] + rows) + 2
             for i in range(len(headers))]
    sep = "+" + "+".join("-" * w for w in col_w) + "+"

    def fmt(r):
        return "|" + "|".join(str(v).center(col_w[i]) for i, v in enumerate(r)) + "|"

    print("\n" + "=" * 80)
    print("  TABLEAU COMPARATIF — ÉVALUATION EXPÉRIMENTALE")
    print("  PDR stable = valeur gelée au FND dans chaque .cc (pas recalculée ici)")
    print("=" * 80)
    print(sep); print(fmt(headers)); print(sep.replace("-", "="))
    for r in rows:
        print(fmt(r)); print(sep)
    return rows


def save_comparison_csv(rows: list, output_path: str):
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Modele", "FND_s", "HND_s", "LND_s",
                         "PDR_Stable_pct", "PDR_Global_pct", "AvgDelay_ms",
                         "TotalEnergyConsumed_J", "TotalPktSent"])
        writer.writerows(rows)
    print(f"  ✓ {output_path}")


# ─── Graphiques ───────────────────────────────────────────────────────────────

def _style_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _plot_field(ax, all_data, field, skip_zero=False):
    for key, d in all_data.items():
        t = d["data"]["time"]
        v = d["data"][field]
        if not t or not v:
            continue
        if skip_zero:
            pairs = [(ti, vi) for ti, vi in zip(t, v) if vi > 0.0]
            if not pairs:
                continue
            t, v = zip(*pairs)
        # Ignorer si la série est vide après filtrage ou longueurs incohérentes
        if len(t) == 0 or len(v) == 0 or len(t) != len(v):
            continue
        st = MODEL_STYLES[key]
        ax.plot(t, v,
                color=st["color"], ls=st["ls"], lw=st["lw"],
                label=MODEL_LABELS[key],
                markevery=max(1, len(t) // 8),
                marker=st["marker"], ms=4.5)


def plot_alive_nodes(all_data, output):
    fig, ax = plt.subplots(figsize=(10, 5))
    _style_ax(ax, "Nombre de nœuds vivants au cours du temps",
              "Temps (s)", "Nœuds vivants")
    _plot_field_smooth(ax, all_data, "alive", window=7)
    n0 = max((d["data"]["alive"][0] if d["data"]["alive"] else 300)
             for d in all_data.values()) if all_data else 300
    ax.axhline(n0 // 2, color='gray', ls=':', lw=1, alpha=0.5)
    ax.text(20, n0 // 2 + 3, "HND (50%)", color='gray', fontsize=8)
    ax.set_ylim(0, n0 + 20)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {output}")


def plot_energy(all_data, output):
    fig, ax = plt.subplots(figsize=(10, 5))
    _style_ax(ax, "Énergie totale consommée au cours du temps",
              "Temps (s)", "Énergie consommée (J)")
    _plot_field(ax, all_data, "energy_consumed")
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {output}")


def _rolling_mean(values, window=5):
    """Moyenne glissante centrée, robuste aux bords."""
    result = []
    half = window // 2
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        result.append(float(np.mean(values[lo:hi])))
    return result


def _trim_at_lnd(t, v, lnd_t, margin=0.05):
    """
    Coupe la série au LND-90% + 5% de marge pour éviter les artefacts de
    fin de simulation (délai/PDR erratiques quand < 10% nœuds vivants).
    """
    if lnd_t <= 0:
        return list(t), list(v)
    cutoff = lnd_t * (1.0 + margin)
    pairs = [(ti, vi) for ti, vi in zip(t, v) if ti <= cutoff]
    if not pairs:
        return list(t), list(v)
    return zip(*pairs)


def _plot_field_smooth(ax, all_data, field, skip_zero=False, window=9,
                       trim_lnd=False):
    """Comme _plot_field mais avec lissage par moyenne glissante."""
    for key, d in all_data.items():
        t = list(d["data"]["time"])
        v = list(d["data"][field])
        if not t or not v:
            continue
        if skip_zero:
            pairs = [(ti, vi) for ti, vi in zip(t, v) if vi > 0.0]
            if not pairs:
                continue
            t, v = zip(*pairs)
            t, v = list(t), list(v)
        if len(t) == 0 or len(v) == 0 or len(t) != len(v):
            continue
        if trim_lnd:
            lnd = _safe_float(d["summary"], "LND_s")
            t, v = _trim_at_lnd(t, v, lnd)
            t, v = list(t), list(v)
        v_smooth = _rolling_mean(v, window=window)
        st = MODEL_STYLES[key]
        ax.plot(t, v_smooth,
                color=st["color"], ls=st["ls"], lw=st["lw"],
                label=MODEL_LABELS[key],
                markevery=max(1, len(t) // 8),
                marker=st["marker"], ms=4.5)


def _cumulative_pdr(pkt_emitted, pkt_delivered):
    """PDR cumulatif recalculé depuis les compteurs bruts."""
    result = []
    for e, d in zip(pkt_emitted, pkt_delivered):
        result.append(100.0 * d / e if e > 0 else 100.0)
    return result


def plot_pdr(all_data, output):
    """
    PDR cumulatif par modele (trait plein).
    Lit PDR_RL_pct depuis le CSV (total livre / total emis depuis t=0).
    Fallback : recalcul depuis RL_PktEmitted / RL_PktDelivered.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    _style_ax(ax, "PDR — taux de livraison cumulatif",
              "Temps (s)", "PDR (%)")

    for key, d in all_data.items():
        t     = list(d["data"]["time"])
        pdr_c = list(d["data"].get("pdr_cum", []))
        pdr_r = list(d["data"]["pdr_round"])
        pkt_e = list(d["data"]["pkt_emitted"])
        pkt_d = list(d["data"]["pkt_delivered"])
        if not t:
            continue

        lnd = _safe_float(d["summary"], "LND_s")
        st  = MODEL_STYLES[key]

        # — Courbe principale : PDR CUMULATIF —
        # Preferer PDR_RL_pct lu du CSV ; si absent (valeur sentinelle -1), recalculer.
        use_csv_cum = pdr_c and any(v >= 0 for v in pdr_c)
        if use_csv_cum:
            pdr_main = list(pdr_c)
        elif pkt_e and pkt_d:
            pdr_main = _cumulative_pdr(pkt_e, pkt_d)
        else:
            pdr_main = []

        if pdr_main:
            t_c, pdr_c_trim = _trim_at_lnd(t, pdr_main, lnd)
            t_c, pdr_c_trim = list(t_c), list(pdr_c_trim)
            ax.plot(t_c, pdr_c_trim,
                    color=st["color"], ls=st["ls"], lw=st["lw"],
                    label=MODEL_LABELS[key],
                    markevery=max(1, len(t_c) // 8),
                    marker=st["marker"], ms=4.5)

    ax.axhline(97, color='#0F6E56', ls=':', lw=1, alpha=0.6)
    ax.text(20, 97.5, "97% objectif", color='#0F6E56', fontsize=8)
    ax.set_ylim(40, 105)
    ax.legend(loc='lower left', fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {output}")


def plot_delay(all_data, output):
    """
    Délai moyen lissé sur 9 rounds et tronqué au LND-90% pour éliminer
    les artefacts de fin de simulation (peu de nœuds → délais erratiques).
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    _style_ax(ax, "Délai moyen bout-en-bout au cours du temps\n"
              "(LEACH/QRouting: IEEE 802.15.4 CSMA/CA | DQN: FlowMonitor NS-3, lissé 9 rounds)",
              "Temps (s)", "Délai moyen (ms)")
    _plot_field_smooth(ax, all_data, "delay", skip_zero=True,
                       window=9, trim_lnd=True)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {output}")


def plot_ablation(all_data, output):
    """
    Analyse d'ablation : 3 composants × 2 métriques.
    Colonne gauche  : FND / HND (durée de vie, en secondes)
    Colonne droite  : PDR stable (%) et Énergie consommée (J) — axes Y séparés

    Toutes les valeurs viennent du summary.csv / fdqnte_summary.csv des .cc.
    Aucun recalcul ici.
    """
    comparisons = [
        ("PEPM",       "FDQN_TE+", "DQN_noPEPM",
         "Impact du PEPM\n(prévention proactive)", "#185FA5"),
        ("Fédération", "FDQN_TE+", "DQN_noFed",
         "Impact de l'apprentissage fédéré\n(FedAvg inter-clusters)", "#8B4DB8"),
        ("IFO",        "FDQN_TE+", "DQN_noIFO",
         "Impact du clustering IFO\n(remplacé par LEACH probabiliste)", "#D4A017"),
    ]

    FULL_COLOR = "#0F6E56"
    w = 0.32

    fig, axes = plt.subplots(3, 2, figsize=(13, 13))
    fig.suptitle("Analyse d'ablation — Contribution de chaque composant FDQN-TE+",
                 fontsize=13, fontweight='bold', y=0.99)

    def gv(key, metric):
        if key not in all_data:
            return 0.0
        return _safe_float(all_data[key]["summary"], metric)

    for row, (label, full_k, ablat_k, title, ablat_col) in enumerate(comparisons):
        if full_k not in all_data or ablat_k not in all_data:
            for c in range(2):
                axes[row, c].set_visible(False)
            continue

        full_lbl  = MODEL_LABELS[full_k]
        ablat_lbl = MODEL_LABELS[ablat_k]
        xs = np.array([0.0, 1.0])

        # ── Colonne gauche : FND / HND ────────────────────────────────────────
        ax = axes[row, 0]
        fnd_f = gv(full_k,  "FND_s");  fnd_a = gv(ablat_k, "FND_s")
        hnd_f = gv(full_k,  "HND_s");  hnd_a = gv(ablat_k, "HND_s")

        ax.bar(xs - w/2, [fnd_f, hnd_f], w, label=full_lbl,
               color=FULL_COLOR, alpha=0.85)
        ax.bar(xs + w/2, [fnd_a, hnd_a], w, label=ablat_lbl,
               color=ablat_col, alpha=0.85)
        ax.set_xticks(xs)
        ax.set_xticklabels(["FND", "HND"], fontsize=10)
        ax.set_ylabel("Durée (s)", fontsize=9)
        ax.set_title(f"{title}\n— Durée de vie réseau —", fontsize=9,
                     fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        for xi, (vf, va) in enumerate([(fnd_f, fnd_a), (hnd_f, hnd_a)]):
            if va > 10.0 and vf > 10.0:
                gain = (vf - va) / va * 100
                ymax = max(vf, va) * 1.04
                ax.text(xs[xi], ymax, f"{gain:+.1f}%",
                        ha='center', fontsize=8, fontweight='bold')
            elif max(vf, va) < 10.0:
                ax.text(xs[xi], 50, "données\nmanquantes",
                        ha='center', fontsize=7, color='red', alpha=0.7)

        # ── Colonne droite : PDR stable + Énergie (axes Y séparés) ───────────
        ax_r = axes[row, 1]
        pdr_f  = gv(full_k,  "PDR_Stable_pct")
        pdr_a  = gv(ablat_k, "PDR_Stable_pct")
        econ_f = gv(full_k,  "TotalEnergyConsumed_J")
        econ_a = gv(ablat_k, "TotalEnergyConsumed_J")

        ax_r.bar(np.array([0.0]) - w/2, [pdr_f],  w,
                 color=FULL_COLOR,  alpha=0.85, label=full_lbl)
        ax_r.bar(np.array([0.0]) + w/2, [pdr_a],  w,
                 color=ablat_col, alpha=0.85, label=ablat_lbl)

        ax2 = ax_r.twinx()
        ax2.bar(np.array([1.0]) - w/2, [econ_f], w,
                color=FULL_COLOR,  alpha=0.60, hatch='///')
        ax2.bar(np.array([1.0]) + w/2, [econ_a], w,
                color=ablat_col, alpha=0.60, hatch='///')
        ax2.set_ylabel("Énergie consommée (J)", fontsize=8, color='#555')
        ax2.tick_params(axis='y', labelcolor='#555')
        ax2.spines['top'].set_visible(False)

        ax_r.set_xticks([0.0, 1.0])
        ax_r.set_xticklabels(["PDR stable\n(%)", "Énergie cons.\n(J)"], fontsize=9)
        ax_r.set_ylabel("PDR stable (%)", fontsize=9)
        ax_r.set_title(f"{title}\n— Qualité de service —", fontsize=9,
                       fontweight='bold')
        ax_r.legend(fontsize=8)
        ax_r.grid(axis='y', alpha=0.3)
        ax_r.spines['top'].set_visible(False)
        ax_r.spines['right'].set_visible(False)

        if pdr_a > 0.5 and pdr_f > 0.5:
            gain_pdr = (pdr_f - pdr_a) / pdr_a * 100
            ax_r.text(0.0, max(pdr_f, pdr_a) * 1.04,
                      f"{gain_pdr:+.1f}%", ha='center', fontsize=8,
                      fontweight='bold')
        if econ_a > 0.1 and econ_f > 0.1:
            gain_e = (econ_f - econ_a) / econ_a * 100
            ax2.text(1.0, max(econ_f, econ_a) * 1.04,
                     f"{gain_e:+.1f}%", ha='center', fontsize=8,
                     fontweight='bold', color='#555')

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {output}")


def plot_radar(all_data, output):
    """
    Radar normalisé.
    FND, HND  : depuis summary FND_s / HND_s (valeurs .cc)
    PDR stable : depuis summary PDR_Stable_pct (valeurs .cc)
    Équilibre énergie : FND/HND (ratio — plus proche de 1 = meilleur équilibre)
    Efficacité PDR/E  : PDR_stable * FND / 100 (qualité × durée)
    """
    criteria = ["FND", "HND", "PDR\nstable", "Équilibre\nénergie", "Efficacité\nPDR/E"]
    N = len(criteria)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), criteria, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8)
    ax.grid(True, alpha=0.3)

    def raw(key, m):
        s = all_data[key]["summary"]
        if m == "FND":  return _safe_float(s, "FND_s")
        if m == "HND":  return _safe_float(s, "HND_s")
        if m == "PDR":  return _safe_float(s, "PDR_Stable_pct")
        if m == "EQ":
            fnd = _safe_float(s, "FND_s")
            hnd = _safe_float(s, "HND_s")
            return fnd / hnd if hnd > 0 else 0.0
        if m == "EFF":
            pdr = _safe_float(s, "PDR_Stable_pct")
            fnd = _safe_float(s, "FND_s")
            return pdr * fnd / 100.0
        return 0.0

    mx = {m: max((raw(k, m) for k in all_data), default=1.0) or 1.0
          for m in ["FND", "HND", "PDR", "EQ", "EFF"]}

    for key, d in all_data.items():
        vals = [
            raw(key, "FND") / mx["FND"],
            raw(key, "HND") / mx["HND"],
            raw(key, "PDR") / 100.0,
            raw(key, "EQ")  / mx["EQ"],
            raw(key, "EFF") / mx["EFF"],
        ]
        vals = [min(1.0, max(0.0, v)) for v in vals]
        vals += vals[:1]
        st = MODEL_STYLES[key]
        ax.plot(angles, vals, color=st["color"], lw=st["lw"],
                ls=st["ls"], label=MODEL_LABELS[key])
        ax.fill(angles, vals, color=st["color"], alpha=0.08)

    ax.set_title("Comparaison multi-critères\n(normalisée)", pad=20,
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {output}")


def plot_dashboard(all_data, output):
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        "Évaluation comparative FDQN-TE+ vs modèles de référence\n"
        "N=300 nœuds, E_init=1.2J, Zone=1000×1000m, Seed=42",
        fontsize=13, fontweight='bold', y=0.98)
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.3)

    subplots = [
        (gs[0, 0], "alive",           "Nœuds vivants",    "Nœuds", False),
        (gs[0, 1], "energy_consumed",  "Énergie consommée","J",     False),
        (gs[1, 0], "pdr_round",        "PDR (par round)",  "%",     False),
        (gs[1, 1], "delay",            "Délai moyen",      "ms",    True),
    ]

    for spec, field, title, ylabel, skip_z in subplots:
        ax = fig.add_subplot(spec)
        _style_ax(ax, title, "Temps (s)", ylabel)
        if field == "pdr_round":
            for key, d in all_data.items():
                t     = list(d["data"]["time"])
                pdr_r = list(d["data"]["pdr_round"])
                if not t or not pdr_r:
                    continue
                lnd = _safe_float(d["summary"], "LND_s")
                t_trim, pdr_trim = _trim_at_lnd(t, pdr_r, lnd)
                t_trim, pdr_trim = list(t_trim), list(pdr_trim)
                st = MODEL_STYLES[key]
                pdr_smooth = _rolling_mean(pdr_trim, window=15)
                ax.plot(t_trim, pdr_smooth,
                        color=st["color"], ls=st["ls"], lw=st["lw"],
                        markevery=max(1, len(t_trim) // 8),
                        marker=st["marker"], ms=4.5)
            ax.set_ylim(40, 105)
            ax.axhline(97, color='gray', ls=':', lw=1, alpha=0.5)
            ax.text(20, 97.5, "97%", color='gray', fontsize=8)
        elif field == "delay":
            _plot_field_smooth(ax, all_data, field, skip_zero=skip_z,
                               window=9, trim_lnd=True)
        elif field == "alive":
            _plot_field_smooth(ax, all_data, field, skip_zero=skip_z, window=7)
        else:
            _plot_field(ax, all_data, field, skip_zero=skip_z)

    handles = [mpatches.Patch(color=MODEL_STYLES[k]["color"],
                               label=MODEL_LABELS[k])
               for k in all_data]
    fig.legend(handles=handles, loc='lower center',
               ncol=min(6, len(handles)), fontsize=9,
               bbox_to_anchor=(0.5, 0.01), framealpha=0.9)
    fig.savefig(output, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {output}")


# ─── Analyse textuelle ────────────────────────────────────────────────────────

def print_analysis(all_data: dict):
    print("\n" + "=" * 70)
    print("  ANALYSE DES RÉSULTATS — valeurs issues des .cc (pas recalculées)")
    print("=" * 70)

    pairs = [
        ("LEACH→FDQN",   "FDQN_TE+", "LEACH",
         "FDQN-TE+ complet vs LEACH baseline."),
        ("HEED→FDQN",    "FDQN_TE+", "HEED",
         "HEED : clustering énergie-aware vs IFO+DRL."),
        ("DQN-LEACH→FDQN","FDQN_TE+", "DQN_LEACH",
         "Apport IFO+PEPM+Fed sur un ADDQN de base avec LEACH."),
        ("Q→DRL",        "FDQN_TE+", "QRouting",
         "ADDQN vs Q-table : apprentissage profond vs tabulaire."),
        ("PEPM",         "FDQN_TE+", "DQN_noPEPM",
         "PEPM : recluster proactif sur nœuds à risque énergétique."),
        ("Fédération",   "FDQN_TE+", "DQN_noFed",
         "FedAvg : partage des poids DQN entre clusters."),
        ("IFO",          "FDQN_TE+", "DQN_noIFO",
         "IFO : clustering équilibré (énergie + densité) vs LEACH probabiliste."),
    ]

    for label, ka, kb, expl in pairs:
        if ka not in all_data or kb not in all_data:
            continue
        sa, sb = all_data[ka]["summary"], all_data[kb]["summary"]
        fnd_a = _safe_float(sa, "FND_s")
        fnd_b = _safe_float(sb, "FND_s")
        hnd_a = _safe_float(sa, "HND_s")
        hnd_b = _safe_float(sb, "HND_s")
        pdr_a = _safe_float(sa, "PDR_Stable_pct")
        pdr_b = _safe_float(sb, "PDR_Stable_pct")
        print(f"\n📌 {label} — {MODEL_LABELS[ka]} vs {MODEL_LABELS[kb]} :")
        if fnd_b > 0:
            print(f"   • FND : {fnd_a:.0f}s vs {fnd_b:.0f}s "
                  f"({(fnd_a - fnd_b) / fnd_b * 100:+.1f}%)")
        if hnd_b > 0:
            print(f"   • HND : {hnd_a:.0f}s vs {hnd_b:.0f}s "
                  f"({(hnd_a - hnd_b) / hnd_b * 100:+.1f}%)")
        if pdr_b > 0:
            print(f"   • PDR stable : {pdr_a:.2f}% vs {pdr_b:.2f}% "
                  f"({pdr_a - pdr_b:+.2f} pts)")
        print(f"   → {expl}")
    print()


# ─── Main ─────────────────────────────────────────────────────────────────────



def main():
    parser = argparse.ArgumentParser(description="Analyse résultats FDQN-TE+")
    parser.add_argument("--results_dir", default="results_eval/scale_N300",
                        help="Répertoire contenant les sous-dossiers de résultats")
    args = parser.parse_args()

    results_dir = args.results_dir
    plots_dir   = Path(results_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    all_data = load_all(results_dir)
    if not all_data:
        print("❌ Aucune donnée trouvée.")
        sys.exit(1)

    print(f"\n{len(all_data)} modèle(s) chargé(s)\n")

    rows = print_comparison_table(all_data)
    save_comparison_csv(rows, str(Path(results_dir) / "comparison_table.csv"))

    print("\nGénération des graphiques...")
    plot_alive_nodes(all_data, str(plots_dir / "01_alive_nodes.png"))
    plot_energy     (all_data, str(plots_dir / "02_energy_consumed.png"))
    plot_pdr        (all_data, str(plots_dir / "03_pdr.png"))
    plot_delay      (all_data, str(plots_dir / "04_delay.png"))
    plot_ablation   (all_data, str(plots_dir / "05_ablation.png"))
    plot_radar      (all_data, str(plots_dir / "06_radar.png"))
    plot_dashboard  (all_data, str(plots_dir / "00_dashboard.png"))


    print_analysis(all_data)
    print(f"\n✅ Résultats dans : {plots_dir}/")


if __name__ == "__main__":
    main()
