# FDQN-TE+ : Guide d'installation et d'utilisation
> Chapitre 4 — Implémentation et simulation

## 1. Architecture du projet

```
fdqn-te-plus/
│
├── scratch/
│   └── fdqn_te_plus.cc          ← Point d'entrée NS-3 (simulation principale)
│
├── model/
│   ├── ifo/
│   │   ├── ifo-clustering.h     ← Interface C++ du clustering IFO
│   │   └── ifo-clustering.cc    ← Implémentation des 5 phases IFO
│   │
│   ├── addqn/
│   │   ├── addqn-routing.h      ← Interface C++ du bridge agent ADDQN
│   │   └── addqn_agent.py       ← Agent Double DQN (NumPy pur)
│   │
│   ├── pepm/
│   │   └── pepm_lstm.py         ← Module LSTM de prédiction énergétique
│   │
│   └── federated/
│       └── fedmeta_drl.py       ← Agrégation FedAvg + méta-adaptation
│
├── utils/
│   └── fdqnte-stats.h           ← Collecte de métriques (PDR, énergie, délai)
│
├── results/                     ← Sorties de simulation (auto-créé)
│   ├── fdqnte_stats.csv
│   ├── flow_monitor.xml
│   └── fdqnte_animation.xml
│
└── CMakeLists.txt               ← Configuration de build NS-3
```

---

## 2. Prérequis

| Outil     | Version recommandée | Rôle                        |
|-----------|--------------------|-----------------------------|
| NS-3      | 3.42+              | Simulateur réseau            |
| Python    | 3.10+              | Agents RL (ADDQN, PEPM)     |
| NumPy     | 1.24+              | Calcul numérique             |
| CMake     | 3.13+              | Build système                |
| NetAnim   | optionnel          | Visualisation                |

```bash
# Installation NS-3 (Ubuntu/Debian)
sudo apt install g++ python3 python3-dev cmake ninja-build git
git clone https://gitlab.com/nsnam/ns-3-dev.git ns-3
cd ns-3
./ns3 configure --enable-examples --enable-python-bindings

# Dépendances Python
pip install numpy
```

---

## 3. Installation du module FDQN-TE+

```bash
# 1. Copier le module dans NS-3
cp -r fdqn-te-plus/ <ns3-root>/contrib/

# 2. Reconfigurer NS-3 pour inclure le module
cd <ns3-root>
./ns3 configure --enable-examples

# 3. Compiler
./ns3 build fdqn-te-plus
```

---

## 4. Lancement de la simulation

### Simulation de base (paramètres du Chapitre 3)
```bash
./ns3 run "fdqn_te_plus"
```

### Simulation personnalisée
```bash
./ns3 run "fdqn_te_plus \
  --nNodes=300 \
  --seed=42 \
  --nClusters=30 \
  --fedRound=50 \
  --stopTime=3600"
```

### Tests des composants Python (sans NS-3)
```bash
# Test de l'agent ADDQN
python3 model/addqn/addqn_agent.py

# Test du module PEPM
python3 model/pepm/pepm_lstm.py

# Test de l'agrégation fédérée
python3 model/federated/fedmeta_drl.py
```

---

## 5. Paramètres configurables

| Paramètre    | Défaut | Description                          |
|-------------|--------|--------------------------------------|
| `nNodes`    | 300    | Nombre de nœuds capteurs             |
| `areaSize`  | 1000   | Côté de la zone de déploiement (m)   |
| `radioRange`| 100    | Portée radio maximale (m)            |
| `initEnergy`| 2.0    | Énergie initiale par nœud (J)        |
| `nClusters` | 30     | Nombre cible de clusters (IFO)       |
| `fedRound`  | 50     | Période d'agrégation fédérée (steps) |
| `lstmHidden`| 64     | Neurones cachés du LSTM (PEPM)       |
| `histWindow`| 10     | Fenêtre temporelle du LSTM           |
| `gamma`     | 0.9    | Facteur d'actualisation (ADDQN)      |
| `epsilonMax`| 0.9    | Exploration initiale                 |
| `epsilonMin`| 0.1    | Exploration minimale                 |
| `seed`      | 42     | Graine aléatoire                     |

---

## 6. Métriques de sortie

Les résultats sont sauvegardés dans `results/fdqnte_stats.csv` :

| Métrique              | Description                          |
|-----------------------|--------------------------------------|
| FND (First Node Death)| Round où le 1er nœud meurt           |
| HND (Half Node Death) | Round où 50% des nœuds sont morts    |
| LND (Last Node Death) | Round où le dernier nœud meurt       |
| PDR                   | Taux de livraison de paquets (%)     |
| E2E Delay             | Délai moyen de bout en bout (ms)     |
| Energy Variance       | Écart-type des énergies résiduelles  |
| Throughput            | Débit utile (kbps)                   |

---

## 7. Baselines pour comparaison (Chapitre 5)

```bash
# LEACH (baseline clustering)
./ns3 run "fdqn_te_plus --protocol=leach"

# Q-Routing tabulaire
./ns3 run "fdqn_te_plus --protocol=q-routing"

# DQN standard (sans PEPM, sans fédéré)
./ns3 run "fdqn_te_plus --protocol=dqn"

# FDQN sans PEPM (ablation)
./ns3 run "fdqn_te_plus --protocol=fdqn --disable-pepm"

# FDQN sans fédéré (ablation)
./ns3 run "fdqn_te_plus --protocol=fdqn --disable-federated"

# FDQN-TE+ complet
./ns3 run "fdqn_te_plus --protocol=fdqnte-plus"
```

---

## 8. Flux de données résumé

```
Nœud capteur
  │
  ├─ Observe état s_i(t) = [E_i, d_i, ETX_ij, Q_i]
  │
  ├─ PEPM (LSTM) → prédit risque σ → enrichit état : s_i + [σ]
  │
  ├─ ADDQN → choisit next hop a_i (ε-greedy)
  │
  ├─ Transmet le paquet → reçoit récompense r_i(t)
  │
  ├─ Stocke (s, a, r, s') dans Replay Memory
  │
  └─ Apprend (Double DQN, batch=32)
       │
       └── Toutes les 50 steps :
             CH agrège modèles des membres (FedAvg intra-cluster)
             Sink agrège modèles des CH (FedAvg global + FedMeta)
             Modèle global → diffusé à tous les nœuds
```
