/* =============================================================================
 * addqn-routing.h — Module ADDQN (Adaptive Double Deep Q-Network)
 *
 * Bridge C++ ↔ Python pour l'agent de routage.
 * Le C++ collecte l'état réseau (5 dimensions) et délègue au serveur Python
 * via socket TCP (rl_server.py) ou en mode fallback Q-table.
 *
 * État : s_i(t) = [E_i/E_max, 1-d_i/d_max, ETX_ij, Q_i/Q_max, PEPM_risk]
 * Action : prochain saut parmi les voisins vivants
 * Récompense : λ1*PDR - λ2*E_cost - λ3*delay + λ4*(1-pepm_risk)
 *
 * CORRECTIONS vs version précédente :
 *   ✓ Utilise fdqn_config.h pour les hyperparamètres
 *   ✓ Référence ifo-clustering.h de ce dépôt (pas un dépôt externe)
 *   ✓ ADDQNEnvState inclut ETX calculé (pas hardcodé 0.8)
 *   ✓ Fallback Q-table utilisé si socket TCP indisponible
 *   ✓ FederatedUpdate() : reçoit modèle global et le redistribue
 * ============================================================================= */

#ifndef ADDQN_ROUTING_H
#define ADDQN_ROUTING_H

#include "ns3/object.h"
#include "ns3/node-container.h"
#include "ns3/node.h"
#include "ifo-clustering.h"
#include "fdqn_config.h"
#include <vector>
#include <map>
#include <cstdint>
#include <string>
#include <functional>

namespace ns3 {

// ─── Structures échangées entre C++ et Python ──────────────────────────────

/**
 * État de l'environnement envoyé à Python.
 * Toutes les valeurs sont normalisées dans [0, 1].
 */
struct ADDQNEnvState {
    float residualEnergy;      // E_i(t) / E_max
    float distToSink;          // 1 - d_i / d_max
    float etxToBestNeighbor;   // ETX calculé via ComputeETX() — pas hardcodé
    float queueSize;           // Q_i(t) / Q_max (file d'attente normalisée)
    float predictedRisk;       // Risque PEPM [0, 1] — depuis pepm_lstm.py

    uint32_t              nodeId;
    std::vector<uint32_t> availableNeighbors;  // IDs voisins vivants
};

/**
 * Action retournée par Python (ou la Q-table locale).
 */
struct ADDQNAction {
    uint32_t nextHopNodeId;   // ID du prochain saut choisi
    float    qValue;          // Q-value associée
    bool     isExploration;   // true si ε-greedy a choisi aléatoirement
    int      actionIdx;       // Index dans availableNeighbors
};

/**
 * Récompense calculée après exécution de l'action.
 * r = λ1*pdr - λ2*energy_cost - λ3*delay + λ4*(1 - pepm_risk)
 */
struct ADDQNReward {
    float pdr;          // 1.0 si paquet livré, 0.0 sinon
    float energyCost;   // E_consommée / E_init ∈ [0,1]
    float delay;        // delay_norm ∈ [0,1]
    float pepmSafety;   // (1 - pepm_risk) ∈ [0,1]
    float totalReward;  // λ1*pdr - λ2*energy - λ3*delay + λ4*safety
};

// ─── Classe principale ADDQNRouting ──────────────────────────────────────────

class ADDQNRouting : public Object {
public:

    static TypeId GetTypeId();
    ADDQNRouting();
    virtual ~ADDQNRouting();

    // ── Configuration ────────────────────────────────────────────────────────

    /**
     * Configure tous les hyperparamètres (depuis fdqn_config.h par défaut).
     */
    void SetHyperParams(
        double gamma        = FdqnCfg::GAMMA,
        double epsilonMax   = FdqnCfg::EPSILON_MAX,
        double epsilonMin   = FdqnCfg::EPSILON_MIN,
        double epsilonDecay = FdqnCfg::EPSILON_DECAY,
        uint32_t replaySize = FdqnCfg::REPLAY_SIZE,
        uint32_t batchSize  = FdqnCfg::BATCH_SIZE
    );

    /**
     * Configure la période d'agrégation fédérée.
     */
    void SetFederatedPeriod(uint32_t steps = FdqnCfg::FED_PERIOD);

    /**
     * Installe l'agent sur les nœuds.
     * @param ifo Référence au clustering IFO (pour GetAliveNeighbors, ETX)
     */
    void Install(NodeContainer& sensors, Ptr<IFOClustering> ifo);

    // ── Interface principale ─────────────────────────────────────────────────

    /**
     * Construit l'état complet d'un nœud et demande l'action au serveur Python.
     * Si Python indisponible : utilise la Q-table locale (fallback).
     *
     * @param nodeId ID du nœud
     * @return ADDQNAction avec nextHopNodeId valide
     */
    ADDQNAction DecideNextHop(uint32_t nodeId);

    /**
     * Calcule et soumet la récompense après une transmission.
     * Déclenche un pas d'apprentissage côté Python.
     *
     * @param nodeId         ID du nœud
     * @param txSuccess      true si le paquet a été transmis avec succès
     * @param energyConsumed Énergie dépensée pour la transmission (J)
     * @param delayMs        Délai mesuré (ms)
     */
    ADDQNReward SubmitOutcome(uint32_t nodeId, bool txSuccess,
                               double energyConsumed, double delayMs);

    // ── Fédération ─────────────────────────────────────────────────────────

    /**
     * Récupère les paramètres du modèle ADDQN d'un nœud.
     * Utilisé par FedMeta-DRL pour l'agrégation.
     */
    std::vector<float> GetModelParams(uint32_t nodeId) const;

    /**
     * Applique les paramètres globaux après agrégation fédérée.
     * Appelé après chaque ronde FedMeta-DRL.
     */
    void ApplyFederatedUpdate(const std::vector<float>& globalParams);

    /**
     * True si une ronde fédérée doit être déclenchée.
     */
    bool ShouldFederate() const;

    /**
     * Vérifie si un CH a levé son flag pepmTriggeredRecluster et, si oui,
     * appelle IFOClustering::TriggerProactiveRecluster() immédiatement.
     * À appeler après chaque SubmitOutcome() — coût O(nCH) seulement si flag levé.
     *
     * @param nodes   États des nœuds (modifiés in-place par IFO si rotation)
     * @return        Nombre de rotations CH effectuées
     */
    uint32_t CheckAndTriggerPepmRecluster(std::vector<ns3::NodeState>& nodes);

    // ── Stats ────────────────────────────────────────────────────────────────

    double   GetCurrentEpsilon() const { return m_epsilon; }
    uint32_t GetStepCount()      const { return m_stepCount; }
    uint32_t GetFedRound()       const { return m_fedRound; }

    /**
     * Retourne les stats globales sous forme JSON (pour le dashboard).
     */
    std::string GetStatsJSON() const;

private:

    // ── Collecte d'état ───────────────────────────────────────────────────────
    ADDQNEnvState CollectState(uint32_t nodeId) const;

    // ── Calcul récompense ─────────────────────────────────────────────────────
    ADDQNReward ComputeReward(uint32_t nodeId, bool txSuccess,
                               double energyConsumed, double delayMs) const;

    // ── Fallback Q-table (si Python indisponible) ─────────────────────────────
    ADDQNAction FallbackQTable(uint32_t nodeId,
                                const ADDQNEnvState& state);

    // ── Membres ───────────────────────────────────────────────────────────────
    double    m_gamma;
    double    m_epsilon;
    double    m_epsilonMax;
    double    m_epsilonMin;
    double    m_epsilonDecay;
    uint32_t  m_replaySize;
    uint32_t  m_batchSize;
    uint32_t  m_fedPeriod;
    uint32_t  m_stepCount;
    uint32_t  m_fedRound;

    // Coefficients récompense (depuis fdqn_config.h)
    const float m_lambda1 = static_cast<float>(FdqnCfg::LAMBDA_PDR);
    const float m_lambda2 = static_cast<float>(FdqnCfg::LAMBDA_ENERGY);
    const float m_lambda3 = static_cast<float>(FdqnCfg::LAMBDA_DELAY);
    const float m_lambda4 = static_cast<float>(FdqnCfg::LAMBDA_SAFE);

    NodeContainer        m_sensors;
    Ptr<IFOClustering>   m_ifo;

    // Q-table locale de fallback : nodeId → {neighborId → qValue}
    std::map<uint32_t, std::map<uint32_t, double>> m_qTable;

    // Dernier état par nœud (pour calculer la récompense)
    std::map<uint32_t, ADDQNEnvState> m_lastState;

    // Stats par nœud
    std::map<uint32_t, uint32_t> m_txCount;
    std::map<uint32_t, uint32_t> m_rxCount;
    std::map<uint32_t, double>   m_totalReward;
};

} // namespace ns3

#endif // ADDQN_ROUTING_H
