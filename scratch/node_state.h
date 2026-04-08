/* =============================================================================
 * node_state.h — Structures partagées NodeState / ClusterInfo
 *
 * Inclus par : fdqn_te_plus.cc, ifo_clustering.h, addqn-routing.h
 *
 * NodeState  — état complet d'un nœud capteur (énergie, position, rôle…)
 * ClusterInfo — descripteur d'un cluster (CH + liste membres)
 * NodeDist   — utilitaire distance euclidienne inline
 *
 * Placement NS-3 : scratch/ (même dossier que fdqn_te_plus.cc)
 * ============================================================================= */

#ifndef NODE_STATE_H
#define NODE_STATE_H

#include "fdqn_config.h"

#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>

// =============================================================================
// Fonction utilitaire — distance euclidienne
// =============================================================================

/**
 * Distance euclidienne entre deux points (x1,y1) et (x2,y2).
 */
inline double NodeDist(double x1, double y1, double x2, double y2) {
    const double dx = x1 - x2;
    const double dy = y1 - y2;
    return std::sqrt(dx * dx + dy * dy);
}

// Surcharges pratiques acceptant des NodeState directement (déclarées après la struct)

// =============================================================================
// Structure NodeState
// =============================================================================

/**
 * État complet d'un nœud capteur pendant la simulation.
 *
 * Deux niveaux d'énergie cohabitent :
 *   • energy      — bilan LEACH logique (utilisé pour FND/HND et les métriques)
 *   • NS-3        — BasicEnergySource (géré par NS-3 pour EnergyCallback)
 *
 * Les méthodes Consume() / NormEnergy() / UpdatePEPM() opèrent sur
 * le bilan logique uniquement.
 */
struct NodeState {

    // ── Identification ─────────────────────────────────────────────────────────
    uint32_t id        = 0;
    double   x         = 0.0;
    double   y         = 0.0;

    // ── Énergie (bilan LEACH logique) ─────────────────────────────────────────
    double   energy    = FdqnCfg::E_INIT;  ///< Énergie résiduelle (J)
    bool     isAlive   = true;             ///< false dès energy ≤ 0

    // ── Topologie / cluster ───────────────────────────────────────────────────
    uint32_t clusterId      = 0;           ///< ID du CH de ce nœud (= id si CH)
    bool     isClusterHead  = false;
    double   distToSink     = 0.0;         ///< Distance au sink (m) — mis à jour par IFO

    // ── PEPM ──────────────────────────────────────────────────────────────────
    double   pepmRisk       = 0.0;         ///< Risque PEPM ∈ [0, 1]

    // ── Statistiques ──────────────────────────────────────────────────────────
    uint32_t txCount        = 0;           ///< Nb de transmissions effectuées
    uint32_t reclusterCount = 0;           ///< Nb de reclusters subis
    double   totalReward    = 0.0;         ///< Cumul de récompense RL
    double   fitness        = 0.0;         ///< Fitness IFO (calculée Phase 1)

    // ── Constructeurs ─────────────────────────────────────────────────────────

    NodeState() = default;

    /**
     * Constructeur principal.
     * @param nid       ID NS-3 du nœud
     * @param nx, ny    Position initiale (m)
     * @param initE     Énergie initiale (J)
     * @param dSink     Distance initiale au sink (m)
     */
    NodeState(uint32_t nid, double nx, double ny,
              double initE = FdqnCfg::E_INIT,
              double dSink = 0.0)
        : id(nid), x(nx), y(ny),
          energy(initE), isAlive(true),
          clusterId(0), isClusterHead(false),
          distToSink(dSink),
          pepmRisk(0.0),
          txCount(0), reclusterCount(0),
          totalReward(0.0), fitness(0.0)
    {}

    // ── Méthodes d'état ───────────────────────────────────────────────────────

    /**
     * Consomme @p drain joules. Met isAlive à false si énergie ≤ 0.
     * @return true si le nœud est encore vivant après la déduction.
     */
    bool Consume(double drain) {
        if (!isAlive) return false;
        drain = std::max(0.0, drain);
        energy -= drain;
        if (energy <= 0.0) {
            energy  = 0.0;
            isAlive = false;
        }
        return isAlive;
    }

    /**
     * Fraction d'énergie résiduelle ∈ [0, 1].
     * @param eInit Énergie initiale de référence (défaut = FdqnCfg::E_INIT)
     */
    double NormEnergy(double eInit = FdqnCfg::E_INIT) const {
        return (eInit > 0.0) ? std::max(0.0, energy / eInit) : 0.0;
    }

    /**
     * [CORRECTION Problème 2] UpdatePEPM() SUPPRIMÉE côté C++.
     *
     * Le risque PEPM est exclusivement calculé par pepm_lstm.py via le handler
     * pepm_batch de rl_server.py, puis injecté dans pepmRisk par RequestPEPMBatch().
     *
     * Si Python est indisponible, la dernière valeur de pepmRisk est conservée
     * sans aucun recalcul local — ce qui évite la divergence entre le modèle C++
     * simplifié et le modèle Python (LSTM + EWMA + seuil absolu).
     *
     * Cette méthode est conservée comme stub vide pour ne pas casser les
     * éventuelles références restantes dans le code avant qu'elles soient nettoyées.
     */
    void UpdatePEPM() {
        // NO-OP intentionnel — voir commentaire ci-dessus
    }
};

// =============================================================================
// Surcharges NodeDist acceptant des NodeState
// =============================================================================

inline double NodeDist(const NodeState& a, const NodeState& b) {
    return NodeDist(a.x, a.y, b.x, b.y);
}

inline double NodeDist(const NodeState& a, double bx, double by) {
    return NodeDist(a.x, a.y, bx, by);
}

// =============================================================================
// Structure ClusterInfo
// =============================================================================

/**
 * Descripteur d'un cluster IFO.
 * Créé et maintenu par IFOClustering, lu par fdqn_te_plus.cc et rl_server.py.
 */
struct ClusterInfo {
    uint32_t             chId        = 0;    ///< ID du Cluster Head
    std::vector<uint32_t> members;            ///< IDs des membres (hors CH)
    double               totalEnergy = 0.0;  ///< Somme énergies membres
    double               avgDist     = 0.0;  ///< Distance moyenne membre→CH (m)

    ClusterInfo() = default;

    explicit ClusterInfo(uint32_t ch) : chId(ch) {}

    /** Nombre de membres (hors CH). */
    uint32_t Size() const {
        return static_cast<uint32_t>(members.size());
    }

    /** True si @p nodeId est membre de ce cluster (CH inclus). */
    bool Contains(uint32_t nodeId) const {
        if (nodeId == chId) return true;
        return std::find(members.begin(), members.end(), nodeId) != members.end();
    }
};

#endif // NODE_STATE_H
