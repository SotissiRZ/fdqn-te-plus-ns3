#!/usr/bin/env python3
"""
plot_fdqn.py — Courbes d'évaluation FDQN-TE+
=============================================
Lit les fichiers produits par fdqn_te_plus.cc :
  results/fdqnte_energy.csv          → énergie, E_std, noeuds vivants
  results/rl/fdqnte_rl_history.json  → PDR, récompenses, PEPM, fédération
  results/comparison_metrics.csv     → métriques finales (FND/HND/LND)

Usage :
  python3 plot_fdqn.py                          # cherche dans ./results/
  python3 plot_fdqn.py --results /chemin/vers/results
  python3 plot_fdqn.py --out results/figures/           # dossier de sortie figures

Courbes produites :
  1. Énergie résiduelle moyenne + E_std (incertitude déséquilibre)
  2. Nœuds vivants au fil du temps
  3. PDR_RL vs PDR_NS3 — deux visions de la livraison
  4. Récompense moyenne par round (apprentissage RL)
  5. Énergie totale consommée cumulée
  6. Risque PEPM moyen + noeuds à risque
  7. Nombre de clusters actifs
  8. Rounds fédérés (FedMeta-DRL)
  9. Tableau récapitulatif FND / HND / LND / PDR / Délai
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")   # pas d'écran requis — sortie fichier
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# ─── Palette cohérente ────────────────────────────────────────────────────────
C_ENERGY   = "#2196F3"   # bleu
C_STD      = "#90CAF9"   # bleu clair (fill)
C_ALIVE    = "#4CAF50"   # vert
C_DEAD     = "#F44336"   # rouge
C_PDR_RL   = "#FF9800"   # orange
C_PDR_NS3  = "#9C27B0"   # violet
C_REWARD   = "#00BCD4"   # cyan
C_DRAIN    = "#795548"   # marron
C_PEPM     = "#E91E63"   # rose
C_CLUSTER  = "#607D8B"   # gris-bleu
C_FED      = "#3F51B5"   # indigo


def load_csv(path: str) -> pd.DataFrame | None:
    """Charge un CSV en ignorant les lignes commentaires (#)."""
    if not os.path.exists(path):
        print(f"  [AVERT] Fichier introuvable : {path}", file=sys.stderr)
        return None
    try:
        df = pd.read_csv(path, comment="#")
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        print(f"  [ERREUR] Lecture {path} : {e}", file=sys.stderr)
        return None


def load_json(path: str) -> dict | None:
    """Charge l'historique RL JSON."""
    if not os.path.exists(path):
        print(f"  [AVERT] Fichier introuvable : {path}", file=sys.stderr)
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  [ERREUR] Lecture {path} : {e}", file=sys.stderr)
        return None


def vline_lifetime(ax, fnd, hnd, lnd, ymax=None):
    """Trace les lignes de durée de vie FND/HND/LND si non nulles."""
    for t, label, color in [(fnd, "FND", "#F44336"),
                             (hnd, "HND", "#FF9800"),
                             (lnd, "LND", "#9C27B0")]:
        if t and t > 0:
            ax.axvline(t, color=color, linestyle="--", linewidth=1.2,
                       label=f"{label} = {t:.0f} s", alpha=0.8)


def save_fig(fig, out_dir: str, name: str):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Courbes FDQN-TE+")
    parser.add_argument("--results", default="results",
                        help="Dossier results/ de la simulation (défaut : ./results)")
    parser.add_argument("--out", default="results/figures",
                        help="Dossier de sortie pour les figures (défaut : ./results/figures)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ── Chargement des données ────────────────────────────────────────────────
    energy_csv = load_csv(os.path.join(args.results, "energy", "fdqnte_energy.csv"))
    rl_json    = load_json(os.path.join(args.results, "rl", "fdqnte_rl_history.json"))
    comp_csv   = load_csv(os.path.join(args.results, "comparison_metrics.csv"))

    # Fallback : rl_history peut aussi venir directement du JSON
    history = []
    fnd_s = hnd_s = lnd_s = 0.0
    sim_info = {}

    if rl_json:
        history  = rl_json.get("history", [])
        metrics  = rl_json.get("metrics", {})
        sim_info = rl_json.get("simulation_info", {})
        fnd_s    = metrics.get("fnd_time_s", 0.0)
        hnd_s    = metrics.get("hnd_time_s", 0.0)
        lnd_s    = metrics.get("lnd_time_s", 0.0)

    if not history and energy_csv is not None:
        print("  [INFO] JSON vide — reconstruction depuis le CSV énergie.")

    # ── Préparer les séries temporelles ──────────────────────────────────────
    def col(name, df, fallback=[]):
        return df[name].values if df is not None and name in df.columns else np.array(fallback)

    if energy_csv is not None:
        time_s    = col("Time_s",       energy_csv)
        rounds    = col("Round",         energy_csv)
        e_mean    = col("EnergyMean_J",  energy_csv)
        e_std     = col("EnergyStdDev_J",energy_csv)
        e_min     = col("EnergyMin_J",   energy_csv)
        e_total   = col("TotalDrained_J",energy_csv)
        alive     = col("AliveNodes",    energy_csv)
        dead      = col("DeadNodes",     energy_csv)
        pdr_rl    = col("PDR_RL_pct",    energy_csv)
        pdr_ns3   = col("PDR_NS3_pct",   energy_csv)
        delay_ms  = col("AvgDelay_ms",   energy_csv)
        pepm_risk = col("PEPMRiskMean",  energy_csv)
        pepm_at   = col("AtRiskPEPM",    energy_csv)
        nclusters = col("NClusters",     energy_csv)
        fed_round = col("FedRound",      energy_csv)
        fnd_s     = fnd_s or col("FND_s", energy_csv).max()
        hnd_s     = hnd_s or col("HND_s", energy_csv).max()
        lnd_s     = lnd_s or col("LND_s", energy_csv).max()
    else:
        # Reconstruction minimale depuis l'historique JSON
        time_s    = np.array([h["timestamp_s"]           for h in history])
        rounds    = np.array([h["round"]                  for h in history])
        e_mean    = np.array([h["avg_energy_J"]           for h in history])
        e_std     = np.zeros(len(history))
        e_total   = np.array([h["total_energy_consumed_J"]for h in history])
        alive     = np.array([h["alive_nodes"]            for h in history])
        dead      = np.array([h["dead_nodes"]             for h in history])
        pdr_rl    = np.array([h["pdr_RL_pct"]             for h in history])
        pdr_ns3   = np.array([h["pdr_NS3_pct"]            for h in history])
        delay_ms  = np.array([h["avg_delay_ms"]           for h in history])
        pepm_risk = np.zeros(len(history))
        pepm_at   = np.array([h["at_risk_pepm"]           for h in history])
        nclusters = np.array([h["n_clusters"]             for h in history])
        fed_round = np.array([h["fed_round"]              for h in history])

    reward_mean = np.array([h["rewards"]["mean"]          for h in history]) if history else np.array([])
    reward_min  = np.array([h["rewards"]["min"]           for h in history]) if history else np.array([])
    reward_max  = np.array([h["rewards"]["max"]           for h in history]) if history else np.array([])
    hist_rounds = np.array([h["round"]                    for h in history]) if history else rounds

    x = time_s  # axe X principal : temps (s)
    n_nodes = int(sim_info.get("nNodes", alive[0] if len(alive) else 300))

    plt.rcParams.update({
        "font.size":      11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.dpi":     150,
    })

    # =========================================================================
    # Figure 1 — Énergie résiduelle moyenne ± E_std
    # =========================================================================
    print("\n[1/8] Énergie résiduelle...")
    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(x, e_mean, color=C_ENERGY, linewidth=2, label="Énergie moyenne (J)")
    if len(e_std) == len(x) and e_std.max() > 0:
        ax.fill_between(x, e_mean - e_std, e_mean + e_std,
                        color=C_STD, alpha=0.35,
                        label="±E_std — déséquilibre énergétique")
    if len(e_min) == len(x):
        ax.plot(x, e_min, color=C_DEAD, linewidth=1, linestyle=":", label="Énergie min (nœud le + épuisé)")

    vline_lifetime(ax, fnd_s, hnd_s, lnd_s)
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Énergie résiduelle (J)")
    ax.set_title("FDQN-TE+ — Énergie résiduelle par nœud vivant\n"
                 "E_std = écart-type inter-nœuds (faible = consommation équilibrée)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    save_fig(fig, args.out, "01_energy_residual.png")

    # =========================================================================
    # Figure 2 — Nœuds vivants / morts
    # =========================================================================
    print("[2/8] Survie des nœuds...")
    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(x, alive, color=C_ALIVE, linewidth=2, label="Nœuds vivants")
    ax.fill_between(x, 0, alive, color=C_ALIVE, alpha=0.15)
    if len(dead) == len(x):
        ax2 = ax.twinx()
        ax2.plot(x, dead, color=C_DEAD, linewidth=1.5, linestyle="--", label="Nœuds morts")
        ax2.set_ylabel("Nœuds morts", color=C_DEAD)
        ax2.tick_params(axis="y", colors=C_DEAD)
        ax2.set_ylim(bottom=0)
        ax2.legend(loc="lower right")

    vline_lifetime(ax, fnd_s, hnd_s, lnd_s)
    ax.axhline(n_nodes * 0.5, color="gray", linestyle=":", linewidth=1, label="50% nœuds (HND)")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Nœuds vivants")
    ax.set_title("FDQN-TE+ — Durée de vie du réseau\nFND / HND (50%) / LND (90%)")
    ax.legend(loc="upper right")
    ax.set_ylim(bottom=0, top=n_nodes * 1.05)
    ax.grid(True, alpha=0.3)
    save_fig(fig, args.out, "02_network_lifetime.png")

    # =========================================================================
    # Figure 3 — PDR_RL vs PDR_NS3
    # =========================================================================
    print("[3/8] PDR RL vs NS-3...")
    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(x, pdr_rl,  color=C_PDR_RL,  linewidth=2,
            label="PDR_RL (%) — décisions ADDQN")
    ax.plot(x, pdr_ns3, color=C_PDR_NS3, linewidth=2, linestyle="--",
            label="PDR_NS3 (%) — FlowMonitor physique")

    vline_lifetime(ax, fnd_s, hnd_s, lnd_s)
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("PDR (%)")
    ax.set_title("FDQN-TE+ — Taux de livraison des paquets (PDR)\n"
                 "PDR_RL : couche routage  |  PDR_NS3 : FlowMonitor physique")
    ax.legend()
    ax.set_ylim(top=101, bottom=max(0, pdr_rl.min() - 2) if len(pdr_rl) else 95)
    ax.grid(True, alpha=0.3)
    save_fig(fig, args.out, "03_pdr.png")

    # =========================================================================
    # Figure 4 — Récompense RL (courbe d'apprentissage)
    # =========================================================================
    print("[4/8] Récompense / apprentissage RL...")
    if len(reward_mean):
        fig, ax = plt.subplots(figsize=(9, 4))

        if len(reward_min) == len(reward_mean) and len(reward_max) == len(reward_mean):
            ax.fill_between(hist_rounds, reward_min, reward_max,
                            color=C_REWARD, alpha=0.2, label="min–max récompense")
        ax.plot(hist_rounds, reward_mean, color=C_REWARD, linewidth=2,
                label="Récompense moyenne")

        # Moyenne mobile 10 rounds
        if len(reward_mean) >= 10:
            rm = pd.Series(reward_mean).rolling(10, min_periods=1).mean().values
            ax.plot(hist_rounds, rm, color="navy", linewidth=1.5, linestyle="--",
                    label="Moy. mobile (10 rounds)")

        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
        ax.set_xlabel("Round")
        ax.set_ylabel("Récompense")
        ax.set_title("FDQN-TE+ — Courbe d'apprentissage de l'agent DQN\n"
                     "λ PDR=0.45  |  λ E=0.20  |  λ délai=0.10  |  λ risque=0.10  |  λ hier=0.15")
        ax.legend()
        ax.grid(True, alpha=0.3)
        save_fig(fig, args.out, "04_rl_reward.png")
    else:
        print("  (données de récompense absentes — courbe ignorée)")

    # =========================================================================
    # Figure 5 — Énergie cumulée consommée
    # =========================================================================
    print("[5/8] Énergie consommée cumulée...")
    if len(e_total):
        fig, ax = plt.subplots(figsize=(9, 4))

        ax.plot(x, e_total, color=C_DRAIN, linewidth=2, label="Énergie drainée cumulée (J)")
        ax.fill_between(x, 0, e_total, color=C_DRAIN, alpha=0.1)
        vline_lifetime(ax, fnd_s, hnd_s, lnd_s)
        ax.set_xlabel("Temps (s)")
        ax.set_ylabel("Énergie consommée cumulée (J)")
        ax.set_title("FDQN-TE+ — Énergie totale consommée (cumul réseau)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        save_fig(fig, args.out, "05_energy_cumulative.png")

    # =========================================================================
    # Figure 6 — Risque PEPM
    # =========================================================================
    print("[6/8] Risque PEPM...")
    fig, ax = plt.subplots(figsize=(9, 4))

    if len(pepm_risk) and pepm_risk.max() > 0:
        ax.plot(x, pepm_risk, color=C_PEPM, linewidth=2, label="Risque PEPM moyen [0,1]")
        ax.axhline(0.7, color=C_PEPM, linestyle="--", linewidth=1, alpha=0.6,
                   label="Seuil alerte (0.7)")
        ax.fill_between(x, 0.7, pepm_risk,
                        where=(pepm_risk > 0.7), color=C_PEPM, alpha=0.15,
                        label="Zone à risque")

    if len(pepm_at):
        ax2 = ax.twinx()
        ax2.bar(x, pepm_at, width=(x[1]-x[0]) if len(x)>1 else 50,
                color=C_PEPM, alpha=0.25, label="Nœuds à risque (count)")
        ax2.set_ylabel("Nœuds à risque (n)", color=C_PEPM)
        ax2.tick_params(axis="y", colors=C_PEPM)
        ax2.legend(loc="upper left")

    vline_lifetime(ax, fnd_s, hnd_s, lnd_s)
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Risque PEPM moyen")
    ax.set_title("FDQN-TE+ — Mécanisme PEPM (Predictive Energy Proactive Maintenance)\n"
                 "Score = exp(−TTD / 300)  |  Seuil d'alerte = 0.7  |  LSTM window = 10 steps")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    save_fig(fig, args.out, "06_pepm_risk.png")

    # =========================================================================
    # Figure 7 — Clusters actifs
    # =========================================================================
    print("[7/8] Clusters actifs...")
    fig, ax = plt.subplots(figsize=(9, 4))

    if len(nclusters):
        ax.step(x, nclusters, color=C_CLUSTER, linewidth=2, where="post",
                label="Clusters actifs")
        ax.axhline(30, color="gray", linestyle="--", linewidth=1, alpha=0.5,
                   label="Cible N_CLUSTERS=30")
        ax.fill_between(x, 0, nclusters, step="post", color=C_CLUSTER, alpha=0.1)

    vline_lifetime(ax, fnd_s, hnd_s, lnd_s)
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Nombre de clusters")
    ax.set_title("FDQN-TE+ — Clusters IFO actifs au cours du temps\n"
                 "Cible initiale : N_CLUSTERS = 30  |  Adaptatif : suit la densité résiduelle")
    ax.legend()
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    save_fig(fig, args.out, "07_clusters.png")

    # =========================================================================
    # Figure 8 — FedMeta-DRL rounds fédérés
    # =========================================================================
    print("[8/8] Fédération FedMeta-DRL...")
    fig, ax = plt.subplots(figsize=(9, 4))

    if len(fed_round) and fed_round.max() > 0:
        ax.plot(x, fed_round, color=C_FED, linewidth=2, label="Rounds fédérés cumulés")
        # Pente théorique (1 round tous les FED_PERIOD=50 steps RL)
        ax.set_xlabel("Temps (s)")
        ax.set_ylabel("Rounds fédérés (cumulés)")
        ax.set_title("FDQN-TE+ — Fédération FedAvg inter-clusters\n"
                     "Période : FED_PERIOD = 50 steps RL  |  Agrégation 2 niveaux (intra-cluster → global)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        save_fig(fig, args.out, "08_federation.png")

    # =========================================================================
    # Tableau récapitulatif
    # =========================================================================
    print("\n[Tableau] Métriques finales...")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    labels = [
        "Nœuds simulés",
        "FND (1er nœud mort)",
        "HND (50% morts)",
        "LND (90% morts)",
        "PDR_RL moyen",
        "PDR_NS3 moyen",
        "Délai moyen",
        "Énergie consommée totale",
        "E_std max (déséquilibre)",
        "Rounds RL total",
        "Rounds fédérés total",
    ]
    def safe(arr, fn=np.mean, fmt="{:.2f}"):
        try:
            v = fn(arr[arr > 0]) if len(arr) and (arr > 0).any() else fn(arr)
            return fmt.format(v)
        except Exception:
            return "N/A"

    values = [
        str(n_nodes),
        f"{fnd_s:.1f} s" if fnd_s > 0 else "non atteint",
        f"{hnd_s:.1f} s" if hnd_s > 0 else "non atteint",
        f"{lnd_s:.1f} s" if lnd_s > 0 else "non atteint",
        safe(pdr_rl,  np.mean, "{:.1f} %"),
        safe(pdr_ns3, np.mean, "{:.1f} %"),
        safe(delay_ms, np.mean, "{:.2f} ms"),
        f"{e_total[-1]:.3f} J" if len(e_total) else "N/A",
        f"{e_std.max():.4f} J" if len(e_std) and e_std.max() > 0 else "N/A",
        str(int(fed_round[-1]) * 50) if len(fed_round) and fed_round[-1] > 0 else "N/A",
        str(int(fed_round[-1]))      if len(fed_round) else "N/A",
    ]

    table = ax.table(
        cellText=[[l, v] for l, v in zip(labels, values)],
        colLabels=["Métrique", "Valeur"],
        cellLoc="left",
        loc="center",
        colWidths=[0.55, 0.35],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1565C0")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#E3F2FD")
        cell.set_edgecolor("#BDBDBD")

    ax.set_title("FDQN-TE+ — Récapitulatif métriques de simulation", pad=12,
                 fontsize=13, fontweight="bold")
    save_fig(fig, args.out, "09_summary_table.png")

    print(f"\n✓ {len(os.listdir(args.out))} figures sauvegardées dans : {args.out}/")
    print("  Conseil : ouvrir figures/ avec un navigateur d'images ou :")
    print("  eog figures/*.png   (Linux)   |   open figures/ (macOS)")


if __name__ == "__main__":
    main()
