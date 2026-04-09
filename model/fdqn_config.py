"""
fdqn_config.py — Configuration centralisée FDQN-TE+ (Python)
=============================================================
Miroir exact de modules/fdqn_config.h.
SOURCE UNIQUE DE VÉRITÉ pour tous les paramètres Python.

Importé par : rl_server.py, addqn_agent.py, pepm_lstm.py, fedmeta_drl.py
"""


class FdqnConfig:

    # ── Réseau & déploiement ──────────────────────────────────────────────────
    AREA_SIZE       = 1000.0    # Côté zone carrée (m)
    SINK_X          = 500.0
    SINK_Y          = 500.0
    RADIO_RANGE     = 150.0     # Portée radio physique (m) — valeur simulation actuelle
    N_NODES         = 300       # Scénario référence rapport (ch5: 50/100/200/300)
    N_CLUSTERS      = 30        # Cible IFO : rapport fixe explicitement 30 [CORR écart 2]
    CLUSTER_MEM_MIN = 8
    CLUSTER_MEM_MAX = 12

    # ── Modèle énergétique LEACH ──────────────────────────────────────────────
    E_INIT          = 1.2      # Énergie initiale (J) — synchronisé avec fdqn_config.h
    E_ELEC          = 50e-9     # 50 nJ/bit — circuits TX/RX
    E_AMP           = 10e-12    # 10 pJ/bit/m² — amplificateur
    E_DA            = 5e-9      #  5 nJ/bit — agrégation CH
    PKT_BITS        = 4000      # Taille paquet = 500 octets
    DRAIN_BITS      = 8000     # 1.5 paquets/step (calibré: FND_CH ~round 14 ≈ 696s)

    # ── ADDQN ─────────────────────────────────────────────────────────────────
    # CRITIQUE : STATE_DIM doit correspondre à BuildState() dans fdqn_te_plus.cc
    # BuildState() retourne 10 dimensions :
    #   0  énergie normalisée
    #   1  distance sink normalisée
    #   2  risque PEPM
    #   3  nb voisins vivants normalisé
    #   4  est CH (0/1)
    #   5  nb transmissions normalisé
    #   6  énergie moyenne voisins
    #   7  fraction nœuds vivants
    #   8  distance au CH normalisée
    #   9  recluster count normalisé
    STATE_DIM       = 10        # ← doit rester synchronisé avec BuildState()

    MAX_NEIGHBORS   = 12        # Actions max = CLUSTER_MEM_MAX (voisins max par nœud)
    GAMMA           = 0.99      # Facteur d'actualisation
    LR              = 3e-4      # Taux d'apprentissage (réduit pour stabilité)
    # Écart 3 — ε-greedy : valeurs rapport (ε_max=0.9, ε_min=0.1, δ linéaire=0.002)
    EPSILON_MAX     = 0.9       # ε initial — rapport ch4 [CORR écart 3]
    EPSILON_MIN     = 0.1       # ε minimum — rapport ch4 [CORR écart 3]
    EPSILON_DELTA   = 0.002     # Pas de décroissance linéaire ε(t+1) = ε(t) − δ [CORR écart 3]
    EPSILON_DECAY   = 0.998     # Conservé pour compat C++ fallback (≈ δ=0.002 linéaire)
    # Écart 4 — Mémoire replay M=10 000 transitions (rapport ch4)
    REPLAY_SIZE     = 10000     # Capacité du replay buffer [CORR écart 4]
    BATCH_SIZE      = 64        # Taille du batch d'apprentissage
    # Écart 5 — Target update toutes les 100 itérations (rapport ch4)
    TARGET_UPDATE   = 100       # Fréquence MAJ target network [CORR écart 5]
    TAU             = 0.005     # Soft update (réservé — non utilisé en hard copy)

    # Coefficients récompense multi-objectif  [CORR anomalie 1+3]
    # R = λ1·PDR_signal + λ2·E_norm - λ3·delay - λ4·risk + λ5·hier
    # PDR_signal = +1 si livré, -1 si non livré (pénalité explicite)
    LAMBDA_PDR      = 0.45      # Livraison paquet (dominant — cible 99% PDR)
    LAMBDA_ENERGY   = 0.20      # Énergie résiduelle (maximiser)
    LAMBDA_DELAY    = 0.10      # Délai (dist nextHop→sink)
    LAMBDA_SAFE     = 0.10      # Sécurité PEPM
    LAMBDA_HIER     = 0.15      # Bonus hiérarchie LEACH

    # ── PEPM ──────────────────────────────────────────────────────────────────
    PEPM_HIDDEN     = 64        # Cellules LSTM cachées
    PEPM_WINDOW     = 10        # Fenêtre historique énergie
    PEPM_RISK_THRESHOLD = 0.5   # Seuil alerte : synchronisé avec fdqn_config.h (0.7)
    PEPM_TE_MAX     = 0.5       # Risque progressif dès E_norm < 50% (sync fdqn_config.h)
                                # TE_pred(t) = σ(lstm) × TE_MAX
                                # → si E_résiduelle < TE_pred → nœud à risque
    PEPM_ALPHA = 0.1
    # ── FedMeta-DRL ───────────────────────────────────────────────────────────
    FED_PERIOD      = 50        # Steps RL entre rondes fédérées
    META_ALPHA      = 0.01      # Pas de méta-gradient
    FED_MOMENTUM    = 0.9       # Momentum FedAvg

    # ── Simulation ────────────────────────────────────────────────────────────
    RL_STEP_INTERVAL = 5.0      # Fréquence callback RL (s)
    RL_PORT         = 5555      # Port TCP serveur Python
