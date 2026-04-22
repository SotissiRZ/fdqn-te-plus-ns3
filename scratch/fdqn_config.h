/* =============================================================================
 * fdqn_config.h — Configuration centralisée FDQN-TE+
 *
 * SOURCE UNIQUE DE VÉRITÉ pour tous les paramètres.
 * Inclus par : fdqn_te_plus.cc, ifo-clustering.h, leach_energy.h
 *
 * ORGANISATION :
 *   1. Réseau & déploiement
 *   2. Modèle énergétique LEACH (Heinzelman 2002)
 *   3. Calibration courants NS-3 BasicEnergySource
 *   4. Clustering IFO
 *   5. ADDQN & récompense
 *   6. PEPM
 *   7. FedMeta-DRL
 *   8. Scheduling simulation
 * ============================================================================= */

#ifndef FDQN_CONFIG_H
#define FDQN_CONFIG_H

#include <cstdint>

namespace FdqnCfg {

constexpr double   SIM_DURATION   = 3500.0;  // en secondes

// ─────────────────────────────────────────────────────────────────────────────
// 1. RÉSEAU & DÉPLOIEMENT
// ─────────────────────────────────────────────────────────────────────────────

constexpr double   AREA_SIZE      = 1000.0;  // Côté de la zone carrée (m)
constexpr double   SINK_X         = 500.0;   // Sink au centre
constexpr double   SINK_Y         = 500.0;
constexpr double   RADIO_RANGE    = 150.0;   // Portée radio physique (m) — valeur simulation actuelle
constexpr uint32_t N_NODES        = 300;     // Scénario référence rapport (ch5: 50/100/200/300)

// ─────────────────────────────────────────────────────────────────────────────
// 2. MODÈLE ÉNERGÉTIQUE LEACH (Heinzelman et al., 2002)
//
//   E_tx(k, d) = k * E_ELEC + k * E_AMP * d²     (TX avec amplification libre)
//   E_rx(k)    = k * E_ELEC                        (RX — circuits seulement)
//   E_da(k)    = k * E_DA                          (agrégation données au CH)
//
//   Avec k = PKT_BITS = 4000 bits, d = distance TX→RX en mètres
// ─────────────────────────────────────────────────────────────────────────────

constexpr double   E_INIT         = 1.2;     // Énergie initiale (J) — synchronisé avec fdqn_config.py
constexpr double   E_ELEC         = 50e-9;   // 50 nJ/bit — circuits TX & RX
constexpr double   E_AMP          = 10e-12;  // 10 pJ/bit/m² — amplificateur

// Bits drainés par appel rlStep (toutes les RL_STEP_INTERVAL=5s)
// CORRECTION: 466.00 bits = 10 paquets de 4000 bits par step
// Cela donne une consommation réaliste:
// Membre: 40000 * (50e-9 + 10e-12 * 75²) ≈ 40000 * 106.25e-9 ≈ 4.25 mJ/step
// FND ≈ 2J / 4.25mJ ≈ 470 steps × 5s ≈ 2350s
constexpr uint32_t DRAIN_BITS     = 8000;   // Calibré: FND_CH ~round 14 (t≈696s)
constexpr double   CH_MIN_ENERGY_NORM = 0.30;  // Fallback statique (utilisé hors TriggerProactiveRecluster)
                                               // TriggerProactiveRecluster() utilise un seuil DYNAMIQUE :
                                               // dynMinEnergy = max(0.20, E_moy_norm - 0.05)
                                               // → garantit des candidats quelle que soit la phase de la sim
                                               // Ancienne valeur 0.70 bloquait toutes les rotations dès t≈800s
constexpr double   E_DA           = 5e-9;    //  5 nJ/bit — data aggregation (CH)
constexpr uint32_t PKT_BITS       = 4000;    // Taille paquet = 500 octets
// Énergie d'un cycle complet LEACH (transmission membre → CH → sink) :
//   Membre  : E_tx(d_membre_CH)
//   CH      : E_rx(k)*N_membres + E_da(k)*N_membres + E_tx(d_CH_sink)

// ─────────────────────────────────────────────────────────────────────────────
// 3. CALIBRATION COURANTS NS-3 BasicEnergySource
//
//   But : aligner la consommation NS-3 (courant × tension × durée)
//         sur le modèle LEACH analytique.
//
//   Vérification TX :
//     E_tx(4000 bits, d=100m) = 4000×50nJ + 4000×10pJ×10000 = 600 µJ
//     DataRate = 2000 bps → TxDuration = 4000/2000 = 2 s
//     TX_CURRENT × 3.3V × 2s = 600 µJ  →  TX_CURRENT ≈ 91 µA  ✓
//
//   Note : ces courants servent à NS-3 pour la détection de mort via
//   EnergyCallback. Le bilan LEACH logique est calculé séparément
//   dans leach_energy.h pour les métriques de simulation.
// ─────────────────────────────────────────────────────────────────────────────

constexpr double   SUPPLY_VOLTAGE = 3.3;     // V — tension batterie
constexpr double   TX_CURRENT     = 91e-6;   // A — état transmission
constexpr double   RX_CURRENT     = 61e-6;   // A — état réception
constexpr double   IDLE_CURRENT   = 0.5e-6;  // A — veille active
constexpr double   SLEEP_CURRENT  = 0.1e-6;  // A — sommeil profond

// ─────────────────────────────────────────────────────────────────────────────
// 4. CLUSTERING IFO
// ─────────────────────────────────────────────────────────────────────────────

constexpr uint32_t N_CLUSTERS     = 30;      // Cible IFO — rapport fixe explicitement 30 (ch4)
constexpr uint32_t IFO_ITER       = 20;      // Itérations algorithme IFO
constexpr double   CLUSTER_OPT    = 10.0;    // Taille optimale cluster (N/nC)

// Poids fitness IFO — Formule :
//   F(i) = W1*(E_res/E_max) + W2*(1 - d_sink/d_max) + W3*min(deg/N_opt, 1)
//   Contrainte : W1+W2+W3 = 1.0  [CORR anomalie 7 — ancienne somme = 1.1]
constexpr double   IFO_W1         = 0.50;    // Poids énergie résiduelle
constexpr double   IFO_W2         = 0.35;    // Poids proximité sink
constexpr double   IFO_W3         = 0.15;    // Poids densité locale

// Contrainte membres par cluster : 8 ≤ membres ≤ 12
constexpr double   CLUSTER_MEM_MIN = 8.0;    // minimum membres par cluster (garanti par IFO)
constexpr double   CLUSTER_MEM_MAX = 12.0;   // maximum membres par cluster

constexpr double   RECLUSTER_PERIOD = 100.0; // Période re-clustering (s)

// ─────────────────────────────────────────────────────────────────────────────
// 5. ADDQN & RÉCOMPENSE MULTI-OBJECTIF
// ─────────────────────────────────────────────────────────────────────────────

constexpr uint32_t STATE_DIM      = 10;      // Dimension vecteur d'état (10 composantes — voir BuildState())
constexpr double   GAMMA          = 0.99;    // Facteur d'actualisation
// Écart 3 — ε-greedy : valeurs rapport (ε_max=0.9, ε_min=0.1, décroissance linéaire δ=0.002)
constexpr double   EPSILON_MAX    = 0.9;     // ε initial — rapport ch4
constexpr double   EPSILON_MIN    = 0.1;     // ε minimum — rapport ch4
constexpr double   EPSILON_DELTA  = 0.002;   // Pas décroissance linéaire ε : ε(t+1) = ε(t) − δ
// Note : EPSILON_DECAY conservé pour compatibilité Q-table fallback C++
constexpr double   EPSILON_DECAY  = 0.998;   // Équivalent approx. de δ=0.002 linéaire (inutilisé côté Python)
// Écart 4 — Mémoire replay M=10 000 transitions (rapport ch4 analyse mémoire)
constexpr uint32_t REPLAY_SIZE    = 10000;   // Capacité replay buffer — rapport ch4
constexpr uint32_t BATCH_SIZE     = 64;      // Taille batch d'apprentissage
// Écart 5 — Target update toutes les 100 itérations (rapport ch4)
constexpr uint32_t TARGET_UPDATE  = 100;     // Fréquence MAJ target network — rapport ch4
constexpr uint32_t DEFAULT_SEED = 42;

// Récompense : R = λ1·PDR_signal + λ2·E_norm - λ3·delay - λ4·risk + λ5·hier
// PDR_signal = +1 si livré, -1 si non livré (pénalité explicite)
// Objectif 99% PDR → LAMBDA_PDR dominant  [CORR anomalie 1+3]
constexpr double   LAMBDA_PDR     = 0.45;  // Livraison paquet (dominant — cible 99%)
constexpr double   LAMBDA_ENERGY  = 0.20;  // Énergie résiduelle (maximiser)
constexpr double   LAMBDA_DELAY   = 0.10;  // Délai (dist nextHop→sink)
constexpr double   LAMBDA_SAFE    = 0.10;  // Sécurité PEPM
constexpr double   LAMBDA_HIER    = 0.15;  // Bonus hiérarchie LEACH
// Somme λi = 1.0 ✓

// Référence : Heinzelman 2002 eq. (4)-(5) : énergie définie par ROUND, pas par step.
constexpr uint32_t DRAIN_BITS_PER_STEP = static_cast<uint32_t>(
    DRAIN_BITS * (5.0 / 100.0));  // = 400 bits/step  (RL_STEP=5s, RECLUSTER=100s)

// ─────────────────────────────────────────────────────────────────────────────
// 6. PEPM (Predictive Energy & Path Management)
// ─────────────────────────────────────────────────────────────────────────────

constexpr double   PEPM_RISK_THRESHOLD = 0.7;  // Seuil alerte : détection dès mi-vie (sync pepm_lstm.py)
constexpr double   PEPM_TE_MAX         = 0.5;  // Risque progressif dès E_norm < 50% (sync pepm_lstm.py)
constexpr double   PEPM_ALPHA          = 0.1;  // Taux d'apprentissage PEPM
constexpr int      PEPM_HIDDEN     = 64;
constexpr int      PEPM_WINDOW     = 10;

// ─────────────────────────────────────────────────────────────────────────────
// 7. FEDMETA-DRL
// ─────────────────────────────────────────────────────────────────────────────

constexpr uint32_t FED_PERIOD     = 50;      // Steps RL entre rondes fédérées
constexpr double   META_ALPHA     = 0.01;    // Pas de méta-gradient
constexpr double   FED_MOMENTUM   = 0.9;     // Momentum FedAvg

// ─────────────────────────────────────────────────────────────────────────────
// 8. SCHEDULING SIMULATION
// ─────────────────────────────────────────────────────────────────────────────

constexpr double   RL_STEP_INTERVAL = 5.0;   // Fréquence callback RL (s)
constexpr double   METRICS_INTERVAL = 50.0;  // Fréquence log métriques (s)
constexpr int      RL_PORT          = 5555;  // Port TCP serveur Python
constexpr double   LEARNING_RATE= 0.1;      // Taux d'apprentissage Q-learning

} // namespace FdqnCfg

#endif // FDQN_CONFIG_H
