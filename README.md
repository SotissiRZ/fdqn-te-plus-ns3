# FDQN-TE+ : Guide d'installation et d'utilisation
> Chapitre 4 — Implémentation et simulation NS-3

## 1. Architecture du projet

```
fdqn-te-plus/
│
├── scratch/                          ← Simulations NS-3 (tous les modèles)
│   ├── eval_config.h                 ← Paramètres communs (N, E_init, RADIO_RANGE…)
│   ├── eval_common.h                 ← Fonctions partagées (drain, summary, métriques)
│   ├── leach_energy.h                ← Modèle radio LEACH (Heinzelman 2002)
│   │
│   ├── leach_sim.cc                  ← Baseline 1 : LEACH
│   ├── heed_sim.cc                   ← Baseline 2 : HEED
│   ├── qrouting_sim.cc               ← Baseline 3 : Q-Routing tabulaire
│   ├── fdqn_te_plus_noPEPM.cc       ← Ablation : sans PEPM
│   ├── fdqn_te_plus_noFed.cc        ← Ablation : sans Fédération
│   ├── fdqn_noIFO.cc                 ← Ablation : sans IFO (DQN-LEACH baseline DRL)
│   ├── fdqn_te_plus_eval.cc         ← FDQN-TE+ complet (évaluation)
│   │
│   ├── rl_server_eval.py             ← Serveur FDQN-TE+   (port 5555)
│   ├── rl_server_nopepm.py           ← Serveur sans PEPM  (port 5556)
│   ├── rl_server_nofed.py            ← Serveur sans Fed   (port 5557)
│   ├── rl_server_dqnleach.py         ← Serveur DQN-LEACH  (port 5559)
│   │
│   ├── run_multiseed.sh              ← Lance les 7 modèles × 5 seeds
│   └── aggregate_results.py          ← Agrégation multi-seeds, IC95%, graphes
│
├── model/
│   ├── ifo/
│   │   ├── ifo-clustering.h          ← Interface C++ du clustering IFO
│   │   └── ifo-clustering.cc         ← Implémentation des 5 phases IFO
│   │
│   ├── addqn/
│   │   ├── addqn-routing.h           ← Interface C++ du bridge agent ADDQN
│   │   └── addqn_agent.py            ← Agent Double DQN (NumPy pur)
│   │
│   ├── pepm/
│   │   └── pepm_lstm.py              ← Module LSTM de prédiction énergétique
│   │
│   └── federated/
│       └── fedmeta_drl.py            ← Agrégation FedAvg + méta-adaptation
│
├── results/                          ← Sorties de simulation (auto-créé)
│   ├── fdqnte_stats.csv
│   ├── flow_monitor.xml
│   └── fdqnte_animation.xml
│
└── CMakeLists.txt                    ← Configuration de build NS-3
```

---

## 2. Prérequis

| Outil     | Version recommandée | Rôle                          |
|-----------|---------------------|-------------------------------|
| NS-3      | 3.39+               | Simulateur réseau             |
| Python    | 3.10+               | Agents RL (ADDQN, PEPM, Fed)  |
| NumPy     | 1.24+               | Calcul numérique              |
| CMake     | 3.13+               | Build système                 |
| NetAnim   | optionnel           | Visualisation                 |

```bash
# Installation NS-3 (Ubuntu/Debian)
sudo apt install g++ python3 python3-dev cmake ninja-build git
git clone https://gitlab.com/nsnam/ns-3-dev.git ns-3
cd ns-3
./ns3 configure --enable-examples

# Dépendances Python
pip install numpy scipy pandas matplotlib
```

---

## 3. Installation du module FDQN-TE+

```bash
# 1. Copier les fichiers scratch dans NS-3
cp scratch/*.cc scratch/*.h scratch/*.py \
   ~/ns-allinone-3.39/ns-3.39/scratch/

# 2. Compiler
cd ~/ns-allinone-3.39/ns-3.39
./ns3 build
```

---

## 4. Paramètres de simulation

| Paramètre     | Défaut  | Description                                  |
|---------------|---------|----------------------------------------------|
| `nNodes`      | 300     | Nombre de nœuds capteurs                     |
| `areaSize`    | 1000    | Côté de la zone de déploiement (m)           |
| `radioRange`  | **150** | Portée radio maximale (m)                    |
| `initEnergy`  | **1.2** | Énergie initiale par nœud (J)               |
| `simDuration` | **3500**| Durée de simulation (s)                      |
| `seed`        | 42      | Graine aléatoire                             |
| `fedRound`    | 50      | Période d'agrégation fédérée (steps)         |
| `lstmHidden`  | 64      | Neurones cachés du LSTM (PEPM)               |
| `histWindow`  | 10      | Fenêtre temporelle du LSTM                   |
| `gamma`       | 0.9     | Facteur d'actualisation (ADDQN)              |
| `epsilonMax`  | 0.9     | Exploration initiale                         |
| `epsilonMin`  | 0.1     | Exploration minimale                         |

> **Paramètres modifiés** par rapport aux versions préliminaires :
> `radioRange` 100 m → **150 m**, `initEnergy` 2.0 J → **1.2 J**,
> `simDuration` 3000 s → **3500 s**.

---

## 5. Lancement rapide

### Simulation FDQN-TE+ complète

```bash
cd ~/ns-allinone-3.39/ns-3.39

# Démarrer le serveur RL
python3 scratch/rl_server_eval.py &

# Lancer la simulation
./ns3 run "scratch/fdqn_te_plus_eval \
  --nNodes=300 --seed=42 \
  --resultsDir=results_eval/FDQN_TEplus/seed_42"

kill %1
```

### Évaluation comparative complète (7 modèles × 5 seeds)

```bash
chmod +x scratch/run_multiseed.sh
scratch/run_multiseed.sh --seeds "42 43 44 45 46"

# Agréger et générer les graphes
python3 scratch/aggregate_results.py \
  --results_dir results_eval \
  --seeds 42 43 44 45 46 \
  --outdir results_eval/figures_multiseed
```

### Tests unitaires Python (sans NS-3)

```bash
python3 model/addqn/addqn_agent.py
python3 model/pepm/pepm_lstm.py
python3 model/federated/fedmeta_drl.py
```

---

## 6. Métriques de sortie

| Métrique               | Description                                        |
|------------------------|----------------------------------------------------|
| FND (First Node Death) | Temps où le 1er nœud meurt (s)                    |
| HND (Half Node Death)  | Temps où 50 % des nœuds sont morts (s)            |
| LND-90%                | Temps où 90 % des nœuds sont morts (s)            |
| PDR stable (pré-FND)   | Taux de livraison avant FND (%)                   |
| PDR global             | Taux de livraison sur toute la simulation (%)      |
| Délai moyen            | Délai bout-en-bout moyen FlowMonitor (ms)          |
| Énergie consommée      | Énergie totale drainée depuis t=0 (J)              |
| Équilibre énergie      | Gini des énergies résiduelles                      |

Les résultats sont sauvegardés dans :
- `fdqnte_summary.csv` — scalaires finaux (FDQN-TE+, ablations, DQN-LEACH)
- `summary.csv` — scalaires finaux (LEACH, HEED, Q-Routing)
- `energy/fdqnte_energy.csv` — séries temporelles
- `figures_multiseed/` — graphes PNG avec IC95%

---

## 7. Flux de données résumé

```
Nœud capteur
  │
  ├─ Observe état s_i(t) = [E_i, d_i, ETX_ij, Q_i, σ_PEPM]
  │
  ├─ PEPM (LSTM) → prédit risque σ → enrichit état
  │
  ├─ ADDQN → choisit next hop a_i (ε-greedy Double DQN)
  │
  ├─ Transmet le paquet → reçoit récompense r_i(t)
  │
  ├─ Stocke (s, a, r, s') dans Replay Memory
  │
  └─ Apprend (Double DQN, batch=32)
       │
       └── Toutes les 50 steps (FedRound) :
             CH agrège modèles des membres (FedAvg intra-cluster)
             Sink agrège modèles des CH (FedAvg global + FedMeta)
             Modèle global → diffusé à tous les nœuds
```

