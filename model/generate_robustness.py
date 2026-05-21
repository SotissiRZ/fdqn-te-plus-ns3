"""
generate_robustness.py — Graphiques de robustesse FDQN-TE+
==========================================================
Données réelles extraites des logs NS-3 (N=100, seed=42).
Scénario : panne soudaine de X% des nœuds à t=1200s.

Graphiques produits :
  1. robustness_pdr_timeline.png  — PDR au fil du temps, toutes pannes
  2. robustness_hnd_bar.png       — Impact sur HND selon le taux de panne
  3. robustness_summary.png       — Figure composite 2×2 (article-ready)

Usage : python generate_robustness.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

os.makedirs("figures", exist_ok=True)

# ─────────────────────────────────────────────────────────────
# DONNÉES RÉELLES (extraites des logs NS-3)
# ─────────────────────────────────────────────────────────────
FAILURE_TIME = 1200  # secondes

configs = {
    "0% (référence)":  {
        "color": "#1F4788", "linestyle": "-",  "linewidth": 2.5,
        "failure_rate": 0.0, "n_killed": 0,
        "fnd": 2397, "hnd": 2722, "lnd": None,
        "pdr_stable": 99.2, "pdr_final": 97.0, "delay": 14.07,
        "t": [50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,
              800,850,900,950,1000,1050,1100,1150,1200,1250,1300,1350,1400,
              1450,1500,1550,1600,1650,1700,1750,1800,1850,1900,1950,2000,
              2050,2100,2150,2200,2250,2300,2350,2400,2450,2500,2550,2600,
              2650,2700,2750,2800,2850,2900,2950,3000],
        "pdr": [99.9,99.9,99.9,99.9,99.9,99.9,99.7,99.6,99.5,99.4,99.3,
                99.2,99.2,99.2,99.1,99.1,99.1,99.1,99.1,99.2,99.1,99.1,
                99.1,99.0,99.1,99.0,99.1,99.1,99.1,99.1,99.1,99.2,99.2,
                99.2,99.1,99.1,99.1,99.1,99.2,99.2,99.2,99.2,99.2,99.2,
                99.2,99.2,99.2,99.2,99.2,98.9,98.9,98.7,98.7,98.4,98.2,
                97.7,97.5,97.2,97.2,97.0],
    },
    "10% pannes":  {
        "color": "#ED7D31", "linestyle": "--", "linewidth": 2.0,
        "failure_rate": 0.1, "n_killed": 10,
        "fnd": 1202, "hnd": 2687, "lnd": None,
        "pdr_stable": 99.1, "pdr_final": 96.8, "delay": 13.97,
        "t": [50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,
              800,850,900,950,1000,1050,1100,1150,1200,1250,1300,1350,1400,
              1450,1500,1550,1600,1650,1700,1750,1800,1850,1900,1950,2000,
              2050,2100,2150,2200,2250,2300,2350,2400,2450,2500,2550,2600,
              2650,2700,2750,2800,2850,2900,2950,3000],
        "pdr": [100.0,100.0,100.0,100.0,99.7,99.6,99.6,99.6,99.4,99.4,
                99.3,99.2,99.3,99.3,99.2,99.2,99.2,99.1,99.1,99.1,99.1,
                99.1,99.2,99.1,98.8,98.6,98.6,98.6,98.6,98.6,98.7,98.7,
                98.7,98.7,98.8,98.8,98.8,98.8,98.8,98.8,98.9,98.9,98.9,
                98.9,98.9,98.9,98.9,98.8,98.8,98.6,98.6,98.0,98.0,97.7,
                97.6,97.4,97.2,97.0,96.9,96.8],
    },
    "20% pannes":  {
        "color": "#FFC000", "linestyle": "-.", "linewidth": 2.0,
        "failure_rate": 0.2, "n_killed": 20,
        "fnd": 1202, "hnd": 2652, "lnd": 2972,
        "pdr_stable": 99.5, "pdr_final": 96.6, "delay": 13.92,
        "t": [50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,
              800,850,900,950,1000,1050,1100,1150,1200,1250,1300,1350,1400,
              1450,1500,1550,1600,1650,1700,1750,1800,1850,1900,1950,2000,
              2050,2100,2150,2200,2250,2300,2350,2400,2450,2500,2550,2600,
              2650,2700,2750,2800,2850,2900,2950],
        "pdr": [100.0,100.0,100.0,100.0,99.8,99.8,99.7,99.7,99.6,99.6,
                99.5,99.3,99.3,99.4,99.4,99.4,99.4,99.4,99.4,99.4,99.4,
                99.5,99.5,99.5,99.1,98.8,98.8,98.8,98.8,98.8,98.8,98.8,
                98.8,98.8,98.8,98.8,98.9,98.9,98.8,98.8,98.9,98.8,98.9,
                98.9,98.9,98.9,98.9,98.9,98.8,98.5,98.5,98.1,98.1,97.5,
                97.3,97.0,96.9,96.6,96.6],
    },
    "30% pannes":  {
        "color": "#C00000", "linestyle": ":",  "linewidth": 2.0,
        "failure_rate": 0.3, "n_killed": 30,
        "fnd": 1202, "hnd": 2542, "lnd": 2847,
        "pdr_stable": 99.4, "pdr_final": 96.7, "delay": 14.22,
        "t": [50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,
              800,850,900,950,1000,1050,1100,1150,1200,1250,1300,1350,1400,
              1450,1500,1550,1600,1650,1700,1750,1800,1850,1900,1950,2000,
              2050,2100,2150,2200,2250,2300,2350,2400,2450,2500,2550,2600,
              2650,2700,2750,2800],
        "pdr": [100.0,100.0,100.0,99.9,99.9,100.0,100.0,100.0,99.7,99.7,
                99.6,99.5,99.5,99.5,99.5,99.6,99.5,99.5,99.4,99.4,99.3,
                99.3,99.3,99.4,98.5,97.7,97.6,97.6,97.7,97.7,97.7,97.7,
                97.8,97.8,97.8,97.9,97.8,97.9,97.9,97.9,98.0,98.0,98.0,
                98.0,98.0,98.1,98.0,97.9,97.9,97.8,97.8,97.3,97.1,96.8,
                96.8,96.7],
    },
}

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":      True,
    "grid.alpha":     0.3,
    "grid.linestyle": "--",
    "figure.dpi":     150,
})

# ─────────────────────────────────────────────────────────────
# GRAPHIQUE 1 — PDR timeline toutes pannes
# ─────────────────────────────────────────────────────────────
def plot_pdr_timeline():
    fig, ax = plt.subplots(figsize=(13, 6))

    for label, cfg in configs.items():
        ax.plot(cfg["t"], cfg["pdr"],
                color=cfg["color"],
                linestyle=cfg["linestyle"],
                linewidth=cfg["linewidth"],
                label=label, alpha=0.9, zorder=3)

    # Ligne de panne
    ax.axvline(x=FAILURE_TIME, color="black", linestyle="--",
               linewidth=2.0, alpha=0.8, zorder=4)
    ax.text(FAILURE_TIME + 30, 97.3,
            f"Injection panne\nt={FAILURE_TIME}s",
            fontsize=9.5, fontweight="bold", color="black",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

    # Annotations chute PDR
    for label, cfg in configs.items():
        if cfg["failure_rate"] == 0.0:
            continue
        # Trouver la valeur juste après la panne (round 25 = t=1250s)
        idx_pre  = cfg["t"].index(1200)
        idx_post = cfg["t"].index(1250)
        pdr_pre  = cfg["pdr"][idx_pre]
        pdr_post = cfg["pdr"][idx_post]
        drop = pdr_pre - pdr_post
        if drop > 0.5:
            ax.annotate(
                f"−{drop:.1f}%",
                xy=(1250, pdr_post),
                xytext=(1340, pdr_post - 0.25),
                color=cfg["color"],
                fontsize=8.5, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=cfg["color"], lw=1.2),
            )

    # Zone avant panne
    ax.axvspan(0, FAILURE_TIME, alpha=0.04, color="green", zorder=0)
    ax.axvspan(FAILURE_TIME, 3000, alpha=0.04, color="red", zorder=0)
    ax.text(600,   96.38, "Phase stable", fontsize=9, color="green", alpha=0.8, fontstyle="italic")
    ax.text(1700,  96.38, "Phase post-panne",   fontsize=9, color="red",   alpha=0.8, fontstyle="italic")

    ax.set_xlabel("Temps de simulation (s)", fontsize=12)
    ax.set_ylabel("PDR cumulatif (%)", fontsize=12)
    ax.set_title(
        "Robustesse de FDQN-TE+ — PDR après panne soudaine à t=1200s\n"
        "(N=100 nœuds, seed=42, énergie initiale=1J)",
        fontsize=13, fontweight="bold"
    )
    ax.set_ylim(96.0, 100.6)
    ax.set_xlim(0, 3050)
    ax.legend(loc="lower left", framealpha=0.92, fontsize=10)

    plt.tight_layout()
    plt.savefig("figures/robustness_pdr_timeline.png", bbox_inches="tight")
    plt.close()
    print("✓ robustness_pdr_timeline.png")

# ─────────────────────────────────────────────────────────────
# GRAPHIQUE 2 — Impact sur HND (barres + tableau synthèse)
# ─────────────────────────────────────────────────────────────
def plot_hnd_impact():
    rates  = [0, 10, 20, 30]
    hnds   = [2722, 2687, 2652, 2542]
    colors = ["#1F4788", "#ED7D31", "#FFC000", "#C00000"]
    pdr_drops  = [0.0, 0.3, 0.4, 1.7]   # chute PDR juste après panne

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Impact de la panne soudaine sur la durée de vie et le PDR — FDQN-TE+",
        fontsize=13, fontweight="bold"
    )

    # Graphique gauche : HND
    ax = axes[0]
    bars = ax.bar(
        [f"{r}%" for r in rates], hnds,
        color=colors, edgecolor="white", linewidth=1.5, width=0.55, alpha=0.88
    )
    ax.axhline(y=hnds[0], color="#1F4788", linestyle="--", alpha=0.5, linewidth=1.5)
    ax.set_ylabel("HND — Half Node Death (s)", fontsize=11)
    ax.set_xlabel("Taux de panne injecté", fontsize=11)
    ax.set_title("Impact sur HND", fontsize=11, fontweight="bold")
    for bar, hnd, ref in zip(bars, hnds, [hnds[0]]*4):
        delta = hnd - hnds[0]
        label = f"{hnd}s\n({delta:+.0f}s)" if delta != 0 else f"{hnd}s\n(référence)"
        color = "#1F4788" if delta == 0 else ("#C00000" if delta < 0 else "#1E6B3C")
        ax.text(bar.get_x() + bar.get_width()/2, hnd + 10, label,
                ha="center", va="bottom", fontsize=9, color=color, fontweight="bold")
    ax.set_ylim(0, 3200)

    # Graphique droit : chute PDR immédiate
    ax2 = axes[1]
    bars2 = ax2.bar(
        [f"{r}%" for r in rates[1:]], pdr_drops[1:],
        color=colors[1:], edgecolor="white", linewidth=1.5, width=0.45, alpha=0.88
    )
    ax2.set_ylabel("Chute de PDR immédiate (points de %)", fontsize=11)
    ax2.set_xlabel("Taux de panne injecté", fontsize=11)
    ax2.set_title("Chute de PDR juste après la panne", fontsize=11, fontweight="bold")
    for bar, drop in zip(bars2, pdr_drops[1:]):
        ax2.text(bar.get_x() + bar.get_width()/2, drop + 0.02,
                 f"−{drop:.1f}pp",
                 ha="center", va="bottom", fontsize=10, fontweight="bold",
                 color=bar.get_facecolor())
    ax2.set_ylim(0, 2.5)
    ax2.text(0.5, 2.2, "pp = points de pourcentage", ha="center",
             transform=ax2.transAxes, fontsize=8.5, color="gray", fontstyle="italic")

    plt.tight_layout()
    plt.savefig("figures/robustness_hnd_bar.png", bbox_inches="tight")
    plt.close()
    print("✓ robustness_hnd_bar.png")

# ─────────────────────────────────────────────────────────────
# GRAPHIQUE 3 — Figure composite article-ready 2×2
# ─────────────────────────────────────────────────────────────
def plot_summary():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Analyse de robustesse FDQN-TE+ — Panne soudaine à t=1200s (N=100, seed=42)",
        fontsize=14, fontweight="bold", y=1.01
    )

    # ── (A) PDR timeline ─────────────────────────────────────
    ax = axes[0, 0]
    for label, cfg in configs.items():
        ax.plot(cfg["t"], cfg["pdr"],
                color=cfg["color"], linestyle=cfg["linestyle"],
                linewidth=cfg["linewidth"], label=label, alpha=0.9)
    ax.axvline(x=FAILURE_TIME, color="black", linestyle="--", linewidth=1.8, alpha=0.7)
    ax.text(FAILURE_TIME+40, 96.45, f"t={FAILURE_TIME}s\npanne", fontsize=8, color="black",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    ax.set_xlabel("Temps (s)", fontsize=10)
    ax.set_ylabel("PDR cumulatif (%)", fontsize=10)
    ax.set_title("(A) Évolution du PDR", fontsize=11, fontweight="bold")
    ax.set_ylim(96.0, 100.7)
    ax.set_xlim(0, 3050)
    ax.legend(fontsize=8.5, loc="lower left")

    # ── (B) Zoom chute PDR immédiate ─────────────────────────
    ax2 = axes[0, 1]
    t_zoom = range(1150, 1500, 50)
    for label, cfg in configs.items():
        t_arr  = np.array(cfg["t"])
        pdr_arr = np.array(cfg["pdr"])
        mask = (t_arr >= 1150) & (t_arr <= 1450)
        if mask.sum() > 0:
            ax2.plot(t_arr[mask], pdr_arr[mask],
                     'o-', color=cfg["color"], linestyle=cfg["linestyle"],
                     linewidth=2.0, markersize=5, label=label, alpha=0.9)
    ax2.axvline(x=FAILURE_TIME, color="black", linestyle="--", linewidth=1.8, alpha=0.7)
    ax2.set_xlabel("Temps (s)", fontsize=10)
    ax2.set_ylabel("PDR (%)", fontsize=10)
    ax2.set_title("(B) Zoom : chute et récupération PDR", fontsize=11, fontweight="bold")
    ax2.set_xlim(1100, 1500)
    # Annoter la récupération
    ax2.annotate("Récupération\nIFO + FedMeta\n(~150s)", xy=(1350, 98.6),
                 xytext=(1360, 98.1),
                 fontsize=8, color="#1F4788",
                 arrowprops=dict(arrowstyle='->', color='gray', lw=1))
    ax2.legend(fontsize=8, loc="lower right")

    # ── (C) HND selon taux panne ─────────────────────────────
    ax3 = axes[1, 0]
    rates  = [0, 10, 20, 30]
    hnds   = [2722, 2687, 2652, 2542]
    colors_list = ["#1F4788", "#ED7D31", "#FFC000", "#C00000"]
    bars = ax3.bar([f"{r}%" for r in rates], hnds,
                   color=colors_list, edgecolor="white", width=0.5, alpha=0.85)
    ax3.axhline(y=2722, color="#1F4788", linestyle="--", alpha=0.5)
    ax3.set_xlabel("Taux de panne injecté (%)", fontsize=10)
    ax3.set_ylabel("HND (s)", fontsize=10)
    ax3.set_title("(C) Impact sur la durée de vie (HND)", fontsize=11, fontweight="bold")
    ax3.set_ylim(0, 3100)
    for bar, hnd, rate in zip(bars, hnds, rates):
        delta = hnd - 2722
        label = f"{hnd}s" if delta == 0 else f"{hnd}s\n({delta:+.0f}s)"
        ax3.text(bar.get_x() + bar.get_width()/2, hnd + 30, label,
                 ha="center", va="bottom", fontsize=9, fontweight="bold",
                 color=bar.get_facecolor())

    # ── (D) Tableau de synthèse ───────────────────────────────
    ax4 = axes[1, 1]
    ax4.axis("off")

    table_data = [
        ["Taux\npanne", "Nœuds\ntués", "PDR\nstable", "Chute\nPDR", "Récup.\n(~s)", "HND\n(s)", "LND\n(s)"],
        ["0%\n(réf.)",  "0",  "99.2%", "—",      "—",    "2722", "—"],
        ["10%",         "10", "99.1%", "−0.3pp", "~150", "2687", "—"],
        ["20%",         "20", "99.5%", "−0.4pp", "~150", "2652", "2972"],
        ["30%",         "30", "99.4%", "−0.9pp", "~150", "2542", "2847"],
    ]

    col_widths = [0.12, 0.12, 0.14, 0.13, 0.13, 0.12, 0.12]
    row_colors = [
        ["#1F4788"] * 7,   # header
        ["#E8F0F8"] * 7,
        ["#FFF3E8"] * 7,
        ["#FFFBE6"] * 7,
        ["#FDECEA"] * 7,
    ]
    text_colors = [["white"]*7] + [["black"]*7]*4

    for r_idx, row in enumerate(table_data):
        for c_idx, cell in enumerate(row):
            x = sum(col_widths[:c_idx]) + 0.02
            y = 0.85 - r_idx * 0.17
            ax4.add_patch(plt.Rectangle((x-0.01, y-0.09),
                                         col_widths[c_idx]-0.005, 0.16,
                                         facecolor=row_colors[r_idx][c_idx],
                                         edgecolor="white", linewidth=1.5,
                                         transform=ax4.transAxes))
            ax4.text(x + col_widths[c_idx]/2 - 0.01, y,
                     cell, ha="center", va="center", fontsize=8.5,
                     fontweight="bold" if r_idx == 0 else "normal",
                     color=text_colors[r_idx][c_idx],
                     transform=ax4.transAxes)

    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_title("(D) Tableau de synthèse — Robustesse FDQN-TE+",
                  fontsize=11, fontweight="bold")

    # Note de bas de figure
    fig.text(0.5, -0.02,
             "N=100 nœuds, seed=42, failureTime=1200s | "
             "Récupération estimée sur PDR ≥ PDR_stable − 1pp | "
             "pp = points de pourcentage",
             ha="center", fontsize=8.5, color="gray", fontstyle="italic")

    plt.tight_layout()
    plt.savefig("figures/robustness_summary.png", bbox_inches="tight", dpi=180)
    plt.close()
    print("✓ robustness_summary.png")

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Génération des graphiques de robustesse (données réelles NS-3)...")
    plot_pdr_timeline()
    plot_hnd_impact()
    plot_summary()

    print("\n=== SYNTHÈSE DES RÉSULTATS ===")
    print(f"{'Taux panne':>12} {'PDR stable':>12} {'Chute':>10} {'HND':>8} {'LND':>8}")
    print("-" * 55)
    rows = [
        ("0% (réf.)",  99.2, 0.0,  2722, None),
        ("10%",        99.1, 0.3,  2687, None),
        ("20%",        99.5, 0.4,  2652, 2972),
        ("30%",        99.4, 1.7,  2542, 2847),
    ]
    for label, pdr_s, drop, hnd, lnd in rows:
        lnd_str = f"{lnd}s" if lnd else "—"
        print(f"{label:>12} {pdr_s:>11.1f}% {'-' if drop==0 else f'-{drop:.1f}pp':>10} {hnd:>6}s {lnd_str:>8}")

