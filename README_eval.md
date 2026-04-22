# Guide d'évaluation expérimentale — FDQN-TE+

## Structure des fichiers

```
scratch/
├── fdqn_config.h               ← Configuration du modèle fdqn
├── eval_config.h               ← Configuration partagée (tous les modèles eval)
├── eval_common.h               ← Structures communes (EvalNodeState, EvalDrainCH/Member,
│                                  WriteSummaryCSV, WriteMetricsRow…)
├── leach_energy.h              ← Modèle radio LEACH (LeachEtx/Erx/Eda, EnergyState)
│                                  Partagé par qrouting_sim.cc et fdqn_*.cc
├── eval_common.h               ← fonctions partagées des modèles d'évaluation
├── node_state.h                ← collecteur de l'état d'un noeud
│
├── fdqn_te_plus.cc             ← Modèle 0 : FDQN-TE+ complet
├── leach_sim.cc                ← Modèle 1 : LEACH baseline
├── heed_sim.cc                 ← Modèle 2 : HEED baseline
├── qrouting_sim.cc             ← Modèle 3 : Q-Routing tabulaire
├── fdqn_te_plus_noPEPM.cc      ← Modèle 4 : DQN sans PEPM (ablation)
├── fdqn_te_plus_noFed.cc       ← Modèle 5 : DQN sans Fédération (ablation)
├── fdqn_noIFO.cc               ← Modèle 6 : DQN-LEACH — sans IFO (ablation + baseline DRL)
├── fdqn_LEACH.cc               ← Modèle 7 : DQN-LEACH — sans IFO (ablation + baseline DRL)
│
model/
├── rl_server_nopepm.py         ← Serveur RL — Modèle 4 (port 5556)
├── rl_server_nofed.py          ← Serveur RL — Modèle 5 (port 5557)
├── rl_server_dqnleach.py       ← Serveur RL — Modèle 6 DQN-LEACH (port 5559)
├── rl_server.py                ← Serveur RL — Modèle 7 FDQN-TE+ (port 5555)
│
├── run_multiseed.sh            ← Script maître multi-seeds (lance tout en séquence)
└── aggregate_results.py        ← Agrégation multi-seeds, IC95%, graphes comparatifs
```

---

## Contraintes expérimentales (identiques pour tous les modèles)

| Paramètre           | Valeur             | Notes                                      |
|---------------------|--------------------|--------------------------------------------|
| Nœuds (N)           | 300                | Déployés aléatoirement dans la zone        |
| Zone                | 1000 × 1000 m      |                                            |
| Sink                | (500, 500)         | Centre de la zone                          |
| Énergie initiale    | 1.2 J              | Par nœud                                   |
| Portée radio        | 150 m              | RADIO_RANGE dans eval_config.h             |
| Durée simulation    | 3500 s             | SIM_DURATION dans eval_config.h            |
| Seeds               | 42, 43, 44, 45, 46 | 5 seeds pour IC95%                         |
| DRAIN_BITS          | 8000 bits/step     | 1 paquet complet par step RL               |
| Step RL             | 5 s                | RL_STEP_INTERVAL                           |
| Period reclustering | 100 s              | RECLUSTER_PERIOD — CH election LEACH/HEED  |
| PHY                 | 802.11b DSSS 1 Mbps| delayPerHop = 4.832 ms                     |

> **Note DRAIN_BITS** : chaque step RL envoie 1 paquet complet (8000 bits).
> Ne pas confondre avec la période de reclustering LEACH (100 s) :
> `EvalDrainCH/Member` utilisent `DRAIN_BITS=8000`, pas `DRAIN_BITS_PER_STEP`.

---

## Modèles implémentés

### 1. LEACH (baseline)
- **Principe** : sélection CH probabiliste (p=5%) par round de 100 s
- **Clustering** : membres → CH le plus proche dans RADIO_RANGE
- **Routing** : 2 sauts (membre→CH→sink), sink BS longue portée (1 saut CH→sink)
- **PDR** : ~65 % (nœuds isolés = paquets perdus)
- **Fichier** : `leach_sim.cc` (autonome, pas de serveur Python)
- **Corrections appliquées** :
  - Nœuds isolés → `EvalDrainIsolated()` (idle 8.25 µJ) au lieu de TX fictif
  - Nœuds isolés exclus du calcul de délai moyen
  - `EvalDrainCH/Member` → `DRAIN_BITS=8000` (FIX-3 annulé)

### 2. HEED (baseline)
- **Principe** : sélection CH basée sur l'énergie résiduelle (Younis & Fahmy 2004)
  - `ch_prob = C_PROB × E_résiduelle / E_max`, doublement itératif jusqu'à 1.0
  - Un nœud devient CH_FINAL si `ch_prob ≥ 1.0` et aucun voisin CH plus énergétique
- **Clustering** : convergence en MAX_IT=10 itérations, reclustering toutes les 100 s
- **PDR** : ~94 % (quelques isolés selon densité locale)
- **Paramètre** : `C_prob=0.05`
- **Fichier** : `heed_sim.cc` (autonome)
- **Corrections appliquées** :
  - `has_cluster` flag : stoppe le doubling dès qu'un CH voisin est trouvé
    (sans ce flag : tous les nœuds atteignent ch_prob=1.0 → CH=300)
  - Reclustering gardé par RECLUSTER_PERIOD (100 s), pas à chaque step RL de 5 s
  - Nœuds isolés → `EvalDrainIsolated()` + exclus du délai moyen

### 3. Q-Routing (tabulaire)
- **Principe** : Q-learning classique, table Q(nœud, nextHop), sans réseau de neurones
- **Clustering** : IFO identique à FDQN-TE+
- **Récompense** : même formule que FDQN-TE+
- **Fichier** : `qrouting_sim.cc` (autonome, pas de serveur Python)

### 4. DQN sans PEPM (ablation PEPM)
- **Principe** : FDQN-TE+ complet SAUF PEPM désactivé (risk=0.0 fixe)
- **Fédération** : activée — isole l'impact PEPM seul
- **IFO** : activé
- **Serveur** : `rl_server_nopepm.py` (port **5556**)
- **Fichier C++** : `fdqn_te_plus_noPEPM.cc`

### 5. DQN sans Fédération (ablation Fed)
- **Principe** : FDQN-TE+ avec PEPM MAIS sans agrégation FedAvg inter-clusters
- **Apprentissage** : purement local — chaque nœud conserve son modèle
- **PEPM** : activé
- **IFO** : activé
- **Serveur** : `rl_server_nofed.py` (port **5557**)
- **Fichier C++** : `fdqn_te_plus_noFed.cc`

### 6. DQN-LEACH — ablation IFO (baseline DRL externe)
- **Principe** : ADDQN seul, sans IFO, sans PEPM, sans Fédération
  - Clustering : LEACH probabiliste géré côté C++
  - Mesure l'apport combiné des 3 composants IFO+PEPM+Fed
- **Serveur** : `rl_server_dqnleach.py` (port **5559**)
- **Fichier C++** : `fdqn_noIFO.cc`
- **Dossier résultats** : `results_eval/DQN_LEACH/`

### 7. FDQN-TE+ complet (référence proposée)
- Modèle complet : ADDQN + PEPM (LSTM) + IFO + FedMeta-DRL
- **Serveur** : `rl_server_eval.py` (port **5555**)
- **Fichier C++** : `fdqn_te_plus_eval.cc`
- **Dossier résultats** : `results_eval/FDQN_TEplus/`

---

## Attribution des ports RL

| Modèle              | Serveur Python           | Port  |
|---------------------|--------------------------|-------|
| FDQN-TE+ complet    | `rl_server.py`           | 5555  |
| DQN sans PEPM       | `rl_server_nopepm.py`    | 5556  |
| DQN sans Fédération | `rl_server_nofed.py`     | 5557  |
| DQN sans IFO        | `rl_server_noIFO.py`     | 5558  |
| DQN-LEACH (noIFO)   | `rl_server_dqnleach.py`  | 5559  |

---

## Lancement

### Option A — Multi-seeds automatique (recommandé)

```bash
cd ~/ns-allinone-3.39/ns-3.39/scratch
chmod +x run_multiseed.sh
./run_multiseed.sh --seeds "42 43 44 45 46"
```

### Option B — Seed unique, modèle par modèle

```bash
NS3=~/ns-allinone-3.39/ns-3.39
SEED=42

# 1. LEACH
$NS3 ./ns3 run "scratch/leach_sim \
  --resultsDir=results_eval/LEACH/seed_${SEED} --seed=${SEED}"

# 2. HEED
$NS3 ./ns3 run "scratch/heed_sim \
  --resultsDir=results_eval/HEED/seed_${SEED} --seed=${SEED}"

# 3. Q-Routing
$NS3 ./ns3 run "scratch/qrouting_sim \
  --resultsDir=results_eval/QRouting/seed_${SEED} --seed=${SEED}"

# 4. DQN sans PEPM
python3 /usr/bin/python3 /home/zongo/ns-allinone-3.39/ns-3.39/model/rl_server_nopepm.py &
$NS3 ./ns3 run "scratch/fdqn_te_plus_noPEPM \
  --rlPort=5556                              \
  --resultsDir=results_eval/DQN_noPEPM/seed_${SEED} --seed=${SEED}"
kill %1

# 5. DQN sans Fédération
python3 /usr/bin/python3 /home/zongo/ns-allinone-3.39/ns-3.39/model/rl_server_nofed.py &
$NS ./ns3 run "scratch/fdqn_te_plus_noFed \
  --rlPort=5557                            \
  --resultsDir=results_eval/DQN_noFed/seed_${SEED} --seed=${SEED}"
kill %1

# 6. DQN sans Fédération
python3 /usr/bin/python3 /home/zongo/ns-allinone-3.39/ns-3.39/model/rl_server_noIFO.py &
$NS ./ns3 run "scratch/fdqn_noIFO \
  --rlPort=5558                    \
  --resultsDir=results_eval/DQN_noIFO/seed_${SEED} --seed=${SEED}"
kill %1

# 7. DQN-LEACH (ablation IFO + baseline DRL)
python3 /usr/bin/python3 /home/zongo/ns-allinone-3.39/ns-3.39/model/rl_server_dqnleach.py &
$NS3 ./ns3 run "scratch/fdqn_noIFO   \
  --rlPort=5559                       \
  --resultsDir=results_eval/DQN_LEACH/seed_${SEED} --seed=${SEED}"
kill %1

# 8. FDQN-TE+ complet
python3 /usr/bin/python3 /home/zongo/ns-allinone-3.39/ns-3.39/model/rl_server.py &
$NS3 ./ns3 run "scratch/fdqn_te_plus \
  --rlPort=5555                            \
  --resultsDir=results_eval/FDQN_TEplus/seed_${SEED} --seed=${SEED}"
kill %1
```

### Agrégation et graphes

```bash
python3 /usr/bin/python3 /home/zongo/ns-allinone-3.39/ns-3.39/model/aggregate_results.py \
  --results_dir results_eval \
  --seeds 42 43 44 45 46 \
  --n_nodes 300 \
  --outdir results_eval/figures_multiseed
```

---

## Structure des sorties

```
results_eval/scale_N300/
  ├── LEACH/seed_42/
  │   ├── metrics.csv                 ← séries temporelles par round
  │   └── summary.csv                 ← scalaires finaux (FND_s, HND_s, PDR_Stable_pct…)
  ├── HEED/seed_42/
  │   ├── metrics.csv
  │   └── summary.csv
  ├── QRouting/seed_42/
  │   ├── metrics.csv
  │   └── summary.csv
  ├── DQN_noPEPM/seed_42/
  │   ├── energy/fdqnte_energy.csv
  │   └── fdqnte_summary.csv           ← clés FND_t, HND_t, PDR_RL_preFND_pct…
  ├── DQN_noFed/seed_42/
  │   ├── energy/fdqnte_energy.csv
  │   └── fdqnte_summary.csv
  ├── DQN_LEACH/seed_42/
  │   ├── energy/fdqnte_energy.csv
  │   └── fdqnte_summary.csv
  ├── FDQN_TEplus/seed_42/
  │   ├── energy/fdqnte_energy.csv
  │   ├── routing/fdqnte_routing.csv
  │   └── fdqnte_summary.csv
  │
  └── plots/
      ├── 00_dashboard.png       ← vue d'ensemble
      ├── 01_alive_nodes.png
      ├── 02_energy_consumed.png
      ├── 03_pdr.png
      ├── 04_delay.png
      ├── 05_ablation.png        ← barres FND/HND ± IC95% (5 modèles)
      ├── 0-_radar.png           ← FND vs N ∈ {50,100,200,300}
      └── aggregate_summary.csv  ← tableau agrégé toutes métriques
  |
  figures_multiseed/
  ├── 00_dashboard_multiseed.png    ← vue d'ensemble 2×2 avec IC95%
  ├── 01_alive_nodes_multiseed.png
  ├── 02_energy_multiseed.png
  ├── 03_pdr_multiseed.png
  ├── 04_delay_multiseed.png
  ├── 07_multiseed_bars.png        ← barres FND/HND ± IC95% (5 modèles)
  ├── 08_scalability.png           ← FND vs N ∈ {50,100,200,300}
  └── aggregate_summary.csv        ← tableau agrégé toutes métriques
```

> **Deux formats de summary** :
> - `summary.csv` (Metric,Value) : LEACH, HEED, QRouting — clés `FND_s`/`HND_s`
> - `fdqnte_summary.csv` (Param,Value) : DQN_LEACH, FDQN_TEplus, ablations — clés `FND_t`/`HND_t`
>
> `aggregate_results.py` gère les deux via `PROTO_SUMMARY_FORMAT` automatiquement.

---

## Modèle de délai unifié (tous simulateurs)

| Couche   | Valeur                                        |
|----------|-----------------------------------------------|
| PHY      | IEEE 802.11b DSSS @ 1 Mbps                    |
| Paquet   | 500 octets → txTime = 4.0 ms                  |
| Overhead | SIFS + ACK + DIFS + backoff = 0.832 ms        |
| **Total**| **4.832 ms / saut**                           |


| Protocole          | Topologie CH→sink       | Sauts typiques |
|--------------------|-------------------------|----------------|
| LEACH / HEED       | 1 saut direct (BS LP)   | 2 sauts total  |
| QRouting / DQN / FDQN-TE+ | ceil(dist/range) sauts | 2–4 sauts  |

Les nœuds isolés sont **exclus** du délai moyen dans tous les simulateurs.

---

## Scalabilité (graphe 08)

```bash
for N in 50 100 200 300; do
  $NS3/ns3 run "scratch/leach_sim   --nNodes=${N} \
    --resultsDir=results_eval/scale_N${N}/LEACH/seed_42     --seed=42"
  $NS3/ns3 run "scratch/heed_sim    --nNodes=${N} \
    --resultsDir=results_eval/scale_N${N}/HEED/seed_42      --seed=42"
  $NS3/ns3 run "scratch/qrouting_sim --nNodes=${N} \
    --resultsDir=results_eval/scale_N${N}/QRouting/seed_42  --seed=42"
  # DQN_LEACH et FDQN_TEplus : idem avec leurs serveurs RL
done

python3 scratch/aggregate_results.py \
  --results_dir results_eval \
  --seeds 42 \
  --n_nodes_scale 50 100 200 300
```

---

## Interprétation des résultats

`aggregate_results.py` génère `aggregate_summary.csv` et les graphes couvrant :

1. **Impact PEPM** : FDQN-TE+ vs DQN-noPEPM — FND, HND, PDR stable
2. **Impact Fédération** : FDQN-TE+ vs DQN-noFed
3. **Impact IFO** : FDQN-TE+ vs DQN-LEACH (sans IFO, clustering LEACH)
4. **DRL vs Q-learning** : FDQN-TE+ vs Q-Routing
5. **Gains vs baselines classiques** : FDQN-TE+ vs LEACH et HEED
