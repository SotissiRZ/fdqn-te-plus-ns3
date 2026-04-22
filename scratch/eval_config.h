/* =============================================================================
 * eval_config.h — Configuration partagée pour l'évaluation comparative
 *
 * SOURCE UNIQUE DE VÉRITÉ pour tous les paramètres expérimentaux.
 * Tous les modèles de comparaison incluent ce fichier.
 *
 * Modèles implémentés :
 *   1. LEACH           — baseline classique (clustering aléatoire)
 *   2. Q-Routing       — Q-learning tabulaire standard
 *   3. DQN-noPEPM      — DQN sans prédiction énergétique (ablation PEPM)
 *   4. DQN-noFed       — DQN sans apprentissage fédéré (ablation fédération)
 *   5. FDQN-TE+        — modèle complet (référence)

 * ============================================================================= */

#ifndef EVAL_CONFIG_H
#define EVAL_CONFIG_H

#include <cstdint>
#include <string>

namespace EvalCfg {

// ─────────────────────────────────────────────────────────────────────────────
// CONTRAINTES EXPÉRIMENTALES — identiques pour TOUS les modèles
// ─────────────────────────────────────────────────────────────────────────────

constexpr uint32_t N_NODES        = 300;
constexpr double   AREA_SIZE      = 1000.0;   // m
constexpr double   SINK_X         = 500.0;
constexpr double   SINK_Y         = 500.0;
constexpr double   RADIO_RANGE    = 150.0;    // m
constexpr double   E_INIT         = 1.2;      // J
constexpr double   SIM_DURATION   = 3500.0;   // s
constexpr uint32_t SEED           = 42;

// ─────────────────────────────────────────────────────────────────────────────
// MODÈLE ÉNERGÉTIQUE LEACH (Heinzelman 2002) — commun à tous
// ─────────────────────────────────────────────────────────────────────────────

constexpr double   E_ELEC         = 50e-9;    // nJ/bit — circuits TX/RX
constexpr double   E_AMP          = 10e-12;   // pJ/bit/m²
constexpr double   E_DA           = 5e-9;     // nJ/bit — agrégation CH
constexpr uint32_t PKT_BITS       = 4000;     // 500 octets
constexpr uint32_t DRAIN_BITS     = 8000;     // 2 paquets par round LEACH complet

// ─── Drain effectif par step logique (RL_STEP_INTERVAL) ─────────────────────
// LEACH applique un drain à chaque leachStep (toutes les RL_STEP_INTERVAL=5s),
// mais le modèle Heinzelman 2002 définit 1 transmission par ROUND (100s).
// Sans correction : drain ×20 par round → sur-consommation massive (~328J/sim).
// Correction : DRAIN_BITS_PER_STEP = DRAIN_BITS × (RL_STEP_INTERVAL / RECLUSTER_PERIOD)
// = 8000 × (5/100) = 400 bits/step → drain fidèle à Heinzelman 2002.
//
// Référence : Heinzelman 2002 eq. (4)-(5) : énergie définie par ROUND, pas par step.
constexpr uint32_t DRAIN_BITS_PER_STEP = static_cast<uint32_t>(
    DRAIN_BITS * (5.0 / 100.0));  // = 400 bits/step  (RL_STEP=5s, RECLUSTER=100s)

// Drain idle listening nœud isolé (Heinzelman 2002, eq. 4) :
// E_idle = IDLE_CURRENT × SUPPLY_VOLTAGE × RL_STEP_INTERVAL
// = 0.5µA × 3.3V × 5s = 8.25 µJ/step — bien inférieur à EvalErx (20 µJ/step)
// Ref : TelosB datasheet ; Heinzelman 2000 Section IV-B
constexpr double   IDLE_DRAIN_J   = 0.5e-6 * 3.3 * 5.0;  // = 8.25 µJ/step

constexpr double   SUPPLY_VOLTAGE = 3.3;
constexpr double   TX_CURRENT     = 91e-6;
constexpr double   RX_CURRENT     = 61e-6;
constexpr double   IDLE_CURRENT   = 0.5e-6;
constexpr double   SLEEP_CURRENT  = 0.1e-6;

// ─────────────────────────────────────────────────────────────────────────────
// CLUSTERING — paramètres communs
// ─────────────────────────────────────────────────────────────────────────────

constexpr double   CLUSTER_MEM_MIN = 8.0;
constexpr double   CLUSTER_MEM_MAX = 12.0;
constexpr double   CLUSTER_OPT     = 10.0;
constexpr uint32_t N_CLUSTERS      = 30;
constexpr double   RECLUSTER_PERIOD= 100.0;   // s

// LEACH : fraction de CH par round (formule originale Heinzelman)
constexpr double   LEACH_P         = 0.05;    // 5% de CH par round

// ─────────────────────────────────────────────────────────────────────────────
// Q-ROUTING TABULAIRE — paramètres apprentissage
// ─────────────────────────────────────────────────────────────────────────────

constexpr double   QROUTE_LR      = 0.1;
constexpr double   QROUTE_GAMMA   = 0.9;
constexpr double   QROUTE_EPS_MAX = 0.9;
constexpr double   QROUTE_EPS_MIN = 0.05;
constexpr double   QROUTE_EPS_DECAY = 0.995;

// ─────────────────────────────────────────────────────────────────────────────
// DQN (commun aux variantes DQN-noPEPM, DQN-noFed, FDQN-TE+)
// ─────────────────────────────────────────────────────────────────────────────

constexpr uint32_t STATE_DIM      = 10;
constexpr uint32_t MAX_NEIGHBORS  = 12;
constexpr double   DQN_GAMMA      = 0.99;
constexpr double   DQN_LR         = 3e-4;
constexpr double   DQN_EPS_MAX    = 0.9;
constexpr double   DQN_EPS_MIN    = 0.1;
constexpr double   DQN_EPS_DELTA  = 0.002;
constexpr uint32_t DQN_REPLAY     = 10000;
constexpr uint32_t DQN_BATCH      = 64;
constexpr uint32_t DQN_TARGET_UPD = 100;

// Récompense (identique pour DQN-noPEPM, DQN-noFed, FDQN-TE+)

constexpr double   LAMBDA_PDR     = 0.45;
constexpr double   LAMBDA_ENERGY  = 0.20;
constexpr double   LAMBDA_DELAY   = 0.10;
constexpr double   LAMBDA_SAFE    = 0.10;
constexpr double   LAMBDA_HIER    = 0.15;

// ─────────────────────────────────────────────────────────────────────────────
// PEPM — uniquement pour FDQN-TE+
// ─────────────────────────────────────────────────────────────────────────────

constexpr double   PEPM_RISK_THRESHOLD = 0.7;  // sync fdqn_config.h
constexpr double   PEPM_TE_MAX         = 0.5;
constexpr int      PEPM_HIDDEN         = 64;
constexpr int      PEPM_WINDOW         = 10;
constexpr double   PEPM_ALPHA          = 0.1;

// ─────────────────────────────────────────────────────────────────────────────
// FÉDÉRATION — uniquement pour FDQN-TE+
// ─────────────────────────────────────────────────────────────────────────────

constexpr uint32_t FED_PERIOD     = 50;
constexpr double   META_ALPHA     = 0.01;
constexpr double   FED_MOMENTUM   = 0.9;

// ─────────────────────────────────────────────────────────────────────────────
// SCHEDULING
// ─────────────────────────────────────────────────────────────────────────────

constexpr double   RL_STEP_INTERVAL = 5.0;    // s
constexpr double   METRICS_INTERVAL = 50.0;   // s
constexpr int      RL_PORT_FDQN        = 5555;    // FDQN-TE+
constexpr int      RL_PORT_DQN_NOPEPM  = 5556;    // DQN-noPEPM
constexpr int      RL_PORT_DQN_NOFED   = 5557;    // DQN-noFed
constexpr int      RL_PORT_DQN_NOIFO   = 5558;    // DQN-noIFO (ablation clustering)

// ─────────────────────────────────────────────────────────────────────────────
// NOMS DES MODÈLES (pour les fichiers de sortie)
// ─────────────────────────────────────────────────────────────────────────────

inline const char* MODEL_LEACH         = "LEACH";
inline const char* MODEL_QROUTING      = "QRouting";
inline const char* MODEL_DQN_NOPEPM    = "DQN_noPEPM";
inline const char* MODEL_DQN_NOFED     = "DQN_noFed";
inline const char* MODEL_DQN_NOIFO     = "DQN_noIFO";   // ablation clustering IFO
inline const char* MODEL_FDQNTEPLUS    = "FDQN_TE+";

} // namespace EvalCfg

#endif // EVAL_CONFIG_H
