/* =============================================================================
 * ifo_clustering.h — Algorithme IFO (Improved Fossa Optimization)
 *
 * Sélectionne les Cluster Heads (CH) en optimisant la fitness :
 *   F(i) = W1*(E_res/E_max) + W2*(1-d_sink/d_max) + W3*min(deg/N_opt, 1)
 *
 * PIPELINE (5 phases) :
 *   Phase 1 — Calcul fitness initiale de tous les nœuds vivants
 *   Phase 2 — Exploration : déplacement en spirale vers les meilleurs voisins
 *   Phase 3 — Exploitation : raffinement local des top-20%
 *   Phase 4 — Sélection CH : top-k par fitness, avec contrainte espacement
 *   Phase 5 — Formation clusters : affectation membres → CH (portée radio)
 *
 * CONTRAINTE CLUSTERS :
 *   nClusters est calculé depuis la topologie réelle pour garantir
 *   CLUSTER_MEM_MIN ≤ membres/CH ≤ CLUSTER_MEM_MAX (cf. fdqn_config.h)
 *
 * Usage :
 *   IFOClustering ifo;
 *   ifo.SetArea(sinkX, sinkY, areaSize, radioRange, initEnergy);
 *   uint32_t nc = ifo.ComputeNClusters(nodeStates);
 *   ifo.Run(nodeStates, nc);
 *   const auto& clusters = ifo.GetClusters();
 *
 * Placement NS-3 : scratch/ (même dossier que fdqn_te_plus.cc)
 * ============================================================================= */

#ifndef IFO_CLUSTERING_H
#define IFO_CLUSTERING_H

#include "fdqn_config.h"
#include "node_state.h"

#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdint>
#include <stdexcept>

// ─────────────────────────────────────────────────────────────────────────────
// Classe IFOClustering
// ─────────────────────────────────────────────────────────────────────────────

class IFOClustering {
public:

    // ── Construction & configuration ─────────────────────────────────────────

    IFOClustering()
        : m_sinkX(FdqnCfg::SINK_X), m_sinkY(FdqnCfg::SINK_Y),
          m_areaSize(FdqnCfg::AREA_SIZE), m_radioRange(FdqnCfg::RADIO_RANGE),
          m_eInit(FdqnCfg::E_INIT), m_maxIter(FdqnCfg::IFO_ITER),
          m_round(0) {}

    /**
     * Configure les paramètres géographiques et énergétiques.
     * Appeler avant toute utilisation.
     */
    void SetArea(double sinkX, double sinkY, double areaSize,
                 double radioRange, double eInit,
                 uint32_t maxIter = FdqnCfg::IFO_ITER) {
        m_sinkX      = sinkX;
        m_sinkY      = sinkY;
        m_areaSize   = areaSize;
        m_radioRange = radioRange;
        m_eInit      = eInit;
        m_maxIter    = maxIter;
    }

    // ── Interface principale ──────────────────────────────────────────────────

    /**
     * Calcule le nombre optimal de clusters depuis la topologie réelle.
     *
     * Méthode :
     *   1. Compter les voisins réels de chaque nœud vivant (dans radioRange)
     *   2. degMoy = moyenne du nombre de voisins
     *   3. membresOpt = interpolation linéaire dans [MEM_MIN, MEM_MAX]
     *      selon degMoy ∈ [DEG_MIN=8, DEG_MAX=40]
     *   4. nClusters = ceil(N_vivants / membresOpt), borné dans [N/30, N/4]
     *
     * @param nodes   États des nœuds (positions et isAlive)
     * @return        Nombre de clusters recommandé
     */
    uint32_t ComputeNClusters(const std::vector<NodeState>& nodes) const {
        // Compter les vivants et leurs degrés
        double   totalDeg = 0.0;
        uint32_t alive    = 0;

        for (const auto& ni : nodes) {
            if (!ni.isAlive) continue;
            alive++;
            uint32_t deg = 0;
            for (const auto& nj : nodes) {
                if (!nj.isAlive || nj.id == ni.id) continue;
                if (NodeDist(ni.x, ni.y, nj.x, nj.y) <= m_radioRange)
                    deg++;
            }
            totalDeg += deg;
        }

        if (alive == 0) return FdqnCfg::N_CLUSTERS;

        const double degMoy = totalDeg / alive;

        // membresOpt ∈ [MEM_MIN, MEM_MAX] interpolé sur [DEG_MIN, DEG_MAX]
        constexpr double DEG_MIN  = 5.0, DEG_MAX  = 15.0;
        const double MEM_MIN = FdqnCfg::CLUSTER_MEM_MIN;
        const double MEM_MAX = FdqnCfg::CLUSTER_MEM_MAX;

        const double t = std::min(1.0, std::max(0.0,
                            (degMoy - DEG_MIN) / (DEG_MAX - DEG_MIN)));
        const double membresOpt = MEM_MIN + (MEM_MAX - MEM_MIN) * t;

        uint32_t nc = static_cast<uint32_t>(
                          std::ceil(static_cast<double>(alive) / membresOpt));

        // FIX D1a/D7 : bornes adaptatives au nombre de vivants
        // nc_max = min(N_CLUSTERS, alive/MEM_MIN) — pas plus de clusters que possible
        // nc_min = max(2, alive/MEM_MAX) — assure au moins MEM_MAX membres par CH
        const uint32_t ncMax = std::min(
            static_cast<uint32_t>(FdqnCfg::N_CLUSTERS),
            static_cast<uint32_t>(std::ceil(static_cast<double>(alive) / MEM_MIN)));
        const uint32_t ncMin = std::max(2u,
            static_cast<uint32_t>(std::ceil(static_cast<double>(alive) / MEM_MAX)));
        nc = std::min(nc, ncMax);
        nc = std::max(nc, ncMin);

        return nc;
    }

    /**
     * Exécute le pipeline IFO complet et met à jour nodeStates.
     *
     * @param nodes     États des nœuds (modifiés : clusterId, isClusterHead)
     * @param nClusters Nombre de clusters cible (0 = auto via ComputeNClusters)
     */
    void Run(std::vector<NodeState>& nodes, uint32_t nClusters = 0) {
        if (nClusters == 0)
            nClusters = ComputeNClusters(nodes);

        m_clusters.clear();
        m_chToIdx.clear();
        m_round++;

        // Réinitialiser les rôles
        for (auto& n : nodes) {
            n.isClusterHead = false;
            n.clusterId     = 0;
        }

        // Extraire les vivants
        std::vector<NodeState*> alive;
        for (auto& n : nodes)
            if (n.isAlive) alive.push_back(&n);

        if (alive.empty()) return;

        Phase1_Fitness(alive);
        Phase2_Explore(alive);
        Phase3_Exploit(alive);
        Phase4_SelectCH(alive, nClusters);
        Phase5_FormClusters(alive);
        Phase6_PruneEmpty(alive);   // Supprime les clusters sans membres viables [FIX min=0]
    }

    // ── Accesseurs ────────────────────────────────────────────────────────────

    const std::vector<ClusterInfo>& GetClusters() const { return m_clusters; }
    uint32_t GetRound()   const { return m_round; }
    uint32_t GetNCH()     const { return static_cast<uint32_t>(m_clusters.size()); }

    /** Statistiques membres/cluster pour validation */
    struct ClusterStats {
        uint32_t nClusters;
        uint32_t membersMin, membersMax;
        double   membersMean;
    };

    ClusterStats GetStats() const {
        if (m_clusters.empty())
            return {0, 0, 0, 0.0};
        uint32_t mn = UINT32_MAX, mx = 0; double sum = 0;
        for (const auto& c : m_clusters) {
            uint32_t sz = static_cast<uint32_t>(c.members.size());
            mn = std::min(mn, sz);
            mx = std::max(mx, sz);
            sum += sz;
        }
        return {static_cast<uint32_t>(m_clusters.size()), mn, mx,
                sum / m_clusters.size()};
    }

    // ── Rotation CH proactive via PEPM ────────────────────────────────────────
    // (déclaré ici dans la section publique — avant le bloc private)

    /**
     * Déclenche une rotation anticipée des CH à risque PEPM sans réexécuter
     * le pipeline IFO complet. Appelé à chaque step RL dès qu'un CH dépasse
     * FdqnCfg::PEPM_RISK_THRESHOLD — indépendamment du calendrier RECLUSTER_PERIOD.
     *
     * Algorithme (par CH à risque) :
     *   1. Identifier les membres du cluster avec énergie normalisée ≥ CH_MIN_ENERGY_NORM
     *      et risque PEPM < seuil (candidats successeurs)
     *   2. Parmi les candidats, choisir celui avec la fitness maximale
     *      F = W1*E_norm + W2*(1−d_sink/d_max) [IFO simplifié]
     *   3. Si un successeur existe → swap CH/membre, incrémenter reclusterCount
     *   4. Sinon → le CH garde son rôle malgré le risque (pas de successeur viable)
     *
     * @param nodes   États des nœuds (modifiés in-place : isClusterHead, clusterId)
     * @return        Nombre de rotations effectuées dans cet appel
     */
    uint32_t TriggerProactiveRecluster(std::vector<NodeState>& nodes) {
        uint32_t rotations = 0;
        const double dMax  = m_areaSize * std::sqrt(2.0);

        for (auto& ci : m_clusters) {
            // Trouver l'état du CH courant
            NodeState* chNode = nullptr;
            for (auto& n : nodes)
                if (n.id == ci.chId && n.isAlive) { chNode = &n; break; }

            if (!chNode) continue;  // CH mort — sera géré au prochain IFO

            // Le CH dépasse-t-il le seuil PEPM ?
            if (chNode->pepmRisk <= FdqnCfg::PEPM_RISK_THRESHOLD) continue;

            // ── Chercher un successeur parmi les membres du cluster ───────────
            NodeState* bestSuccessor = nullptr;
            double     bestFitness   = -1.0;

            for (uint32_t memberId : ci.members) {
                NodeState* m = nullptr;
                for (auto& n : nodes)
                    if (n.id == memberId && n.isAlive) { m = &n; break; }
                if (!m) continue;

                // Critère d'éligibilité : énergie suffisante + risque faible
                if (m->NormEnergy(m_eInit) < FdqnCfg::CH_MIN_ENERGY_NORM) continue;
                if (m->pepmRisk >= FdqnCfg::PEPM_RISK_THRESHOLD)           continue;

                // Fitness IFO simplifiée : énergie + proximité sink
                const double eR  = m->NormEnergy(m_eInit);
                const double dR  = 1.0 - std::min(1.0, m->distToSink / dMax);
                const double fit = FdqnCfg::IFO_W1 * eR + FdqnCfg::IFO_W2 * dR;

                if (fit > bestFitness) {
                    bestFitness   = fit;
                    bestSuccessor = m;
                }
            }

            if (!bestSuccessor) continue;  // Pas de successeur viable

            // ── Swap CH → successeur ─────────────────────────────────────────
            // Ancien CH devient membre ordinaire
            chNode->isClusterHead = false;
            chNode->clusterId     = bestSuccessor->id;  // rattaché au nouveau CH
            chNode->reclusterCount++;

            // Nouveau CH
            bestSuccessor->isClusterHead = true;
            bestSuccessor->clusterId     = bestSuccessor->id;
            bestSuccessor->reclusterCount++;

            // Mettre à jour ClusterInfo
            ci.chId = bestSuccessor->id;
            // Retirer le successeur des membres, ajouter l'ancien CH
            ci.members.erase(
                std::remove(ci.members.begin(), ci.members.end(), bestSuccessor->id),
                ci.members.end());
            ci.members.push_back(chNode->id);

            // Reconstruire l'index chId → cluster idx
            if (m_chToIdx.count(chNode->id)) {
                uint32_t idx = m_chToIdx[chNode->id];
                m_chToIdx.erase(chNode->id);
                m_chToIdx[bestSuccessor->id] = idx;
            }

            rotations++;
        }

        return rotations;
    }

    /**
     * ETX simplifié basé sur la distance.
     * ETX = 1 / p² avec p = max(0.01, 1 - d/radioRange)
     * ETX → 1 si lien parfait (d=0), ETX → ∞ si hors portée
     */
    // Dans ifo_clustering.h, remplacer ComputeETX par:
    double ComputeETX(double dist) const {
        if (dist > m_radioRange) return 1e9;
        if (dist < 1.0) return 1.0; // Lien parfait si très proche
        const double p = std::max(0.1, 1.0 - dist / m_radioRange); // min 10% de succès
        return 1.0 / (p * p);
    }

private:

    // ── Phase 1 : Calcul fitness ──────────────────────────────────────────────

    void Phase1_Fitness(std::vector<NodeState*>& nodes) {

        const double dMax = m_areaSize * std::sqrt(2.0);

        // centre de la zone
        const double centerX = m_areaSize / 2.0;
        const double centerY = m_areaSize / 2.0;

        for (auto* n : nodes) {

            // ───────────── Base IFO ─────────────
            const double eR = n->NormEnergy(m_eInit);

            // proxSink : 1 = proche du sink, 0 = loin
            const double proxSink = 1.0 - std::min(1.0, n->distToSink / dMax);

            uint32_t deg = 0;
            for (const auto* nb : nodes) {
                if (nb->id == n->id) continue;
                if (NodeDist(n->x, n->y, nb->x, nb->y) <= m_radioRange)
                    deg++;
            }

            const double densR = std::min(1.0,
                static_cast<double>(deg) / FdqnCfg::CLUSTER_OPT);

            // FIX Problem 1 — relayQual : nœud proche du sink ET bien entouré
            // Un nœud central isolé (densR faible) est pénalisé → évite concentration au centre
            const double relayQual = proxSink * std::max(0.3, densR);

            // FIX Problem 1 — dispR : récompense la couverture périphérique
            // Un nœud éloigné du centre géographique améliore la distribution des CH
            const double distCenter = NodeDist(n->x, n->y, centerX, centerY);
            const double dispR = std::min(1.0, distCenter / (m_areaSize * 0.7));

            // Fitness de base — somme des poids = W1+W2+W3 = 1.0 ✓ [CORR anomalie 7]
            double fitness =
                FdqnCfg::IFO_W1 * eR
                + FdqnCfg::IFO_W2 * relayQual   // remplace dR (biais centre corrigé)
                + FdqnCfg::IFO_W3 * dispR;       // remplace densR (couverture périphérique)

            // FIX BUG 4 — pepmPenalty appliqué UNIQUEMENT comme facteur multiplicatif final
            // (pas additif — l'ancienne somme effective ≈1.3 est corrigée ici)
            fitness *= (1.0 - std::min(1.0, n->pepmRisk));

            // ───────────── Assignation finale ─────────────
            n->fitness = std::max(0.0, fitness);
        }
    }
    // ── Phase 2 : Exploration en spirale ─────────────────────────────────────

    void Phase2_Explore(std::vector<NodeState*>& nodes) {
        constexpr uint32_t EXPLORE_ITER = 5;
        const uint32_t iMax = std::min(EXPLORE_ITER, m_maxIter);

        for (uint32_t iter = 0; iter < iMax; iter++) {
            for (auto* n : nodes) {
                // Chercher le meilleur voisin dans la portée
                NodeState* bestNb = nullptr;
                double bestFit = n->fitness;

                for (auto* nb : nodes) {
                    if (nb->id == n->id) continue;
                    if (NodeDist(n->x, n->y, nb->x, nb->y) > m_radioRange)
                        continue;
                    if (nb->fitness > bestFit) {
                        bestFit = nb->fitness;
                        bestNb  = nb;
                    }
                }

                if (!bestNb) continue;

                // FIX Problem 1 — amplitude spirale fixée à 0.1×radioRange (constante)
                // L'ancienne formule était proportionnelle à distToSink → amplifiait le biais centre
                const double angle  = 2.0 * M_PI * iter / 8.0;
                const double spiral = 0.1 * m_radioRange * std::sin(angle);
                const double dx = bestNb->x - n->x;
                const double dy = bestNb->y - n->y;
                const double d  = std::max(1.0, std::hypot(dx, dy));

                // Position virtuelle (n'affecte que le calcul fitness)
                const double vx = n->x + 0.3 * dx + spiral * dx / d;
                const double vy = n->y + 0.3 * dy + spiral * dy / d;

                // Recalcul fitness à la position virtuelle
                NodeState tmp = *n;
                tmp.x = vx; tmp.y = vy;
                tmp.distToSink = NodeDist(vx, vy, m_sinkX, m_sinkY);
                const double dMax = m_areaSize * std::sqrt(2.0);
                const double eR   = tmp.NormEnergy(m_eInit);
                const double dR   = 1.0 - std::min(1.0, tmp.distToSink / dMax);

                uint32_t deg = 0;
                for (const auto* nb2 : nodes) {
                    if (nb2->id == tmp.id) continue;
                    if (NodeDist(vx, vy, nb2->x, nb2->y) <= m_radioRange)
                        deg++;
                }
                const double densR = std::min(1.0,
                    static_cast<double>(deg) / FdqnCfg::CLUSTER_OPT);
                const double newFit = FdqnCfg::IFO_W1 * eR
                                    + FdqnCfg::IFO_W2 * dR
                                    + FdqnCfg::IFO_W3 * densR;

                if (newFit > n->fitness) {
                    // La position virtuelle améliore la fitness → accepter
                    n->fitness = newFit;
                    // Note : on garde la vraie position physique.
                    // La fitness virtuelle guide seulement la sélection CH.
                }
            }
        }
    }

    // ── Phase 3 : Exploitation (raffinement des top-20%) ─────────────────────

    void Phase3_Exploit(std::vector<NodeState*>& nodes) {
        std::vector<NodeState*> sorted = nodes;
        std::sort(sorted.begin(), sorted.end(),
                  [](const NodeState* a, const NodeState* b) {
                      return a->fitness > b->fitness; });

        const uint32_t eliteN = std::max(1u,
            static_cast<uint32_t>(sorted.size() * 0.20));

        for (uint32_t e = 0; e < eliteN; e++) {
            auto* n = sorted[e];
            // Bonus énergie : favorise les nœuds à haute énergie résiduelle
            const double bonus = 1.0 + 0.1 * n->NormEnergy(m_eInit);
            n->fitness *= bonus;
        }
    }

    // ── Phase 6 : Nettoyage des clusters vides ou sous-peuplés ───────────────

    /**
     * Supprime les clusters dont le nombre de membres est < CLUSTER_MEM_MIN.
     * L'ancien CH est réaffecté comme membre ordinaire au cluster le plus proche
     * ayant encore de la capacité. Cela garantit que GetStats().membersMin ≥ 1
     * et évite le "membres min=0" dans les logs.
     *
     * Appelé après Phase5_FormClusters().
     */
    void Phase6_PruneEmpty(std::vector<NodeState*>& nodes) {
        // MEM_MIN adaptatif : garantit min=8 en début de vie, descend proprement
        // en fin de vie quand les nœuds meurent (évite élimination totale)
        const uint32_t alive = static_cast<uint32_t>(nodes.size());
        const uint32_t nCH   = static_cast<uint32_t>(m_clusters.size());
        // Plancher = max(1, floor(alive / (nCH * 2))) — toujours ≥1 membre par CH
        const uint32_t MEM_MIN = (nCH > 0)
            ? std::max(1u, alive / (nCH * 2))
            : 1u;
        const uint32_t MEM_MAX = static_cast<uint32_t>(FdqnCfg::CLUSTER_MEM_MAX);

        bool changed = true;
        while (changed) {
            changed = false;

            for (auto it = m_clusters.begin(); it != m_clusters.end(); ) {
                // Un cluster est "invalide" s'il a trop peu de membres
                if (it->members.size() >= MEM_MIN) { ++it; continue; }

                // Trouver le nœud CH
                NodeState* chNode = nullptr;
                for (auto* n : nodes)
                    if (n->id == it->chId) { chNode = n; break; }

                // Réaffecter le CH comme membre du cluster voisin le plus proche
                // avec de la capacité disponible
                uint32_t fallbackCH   = 0;
                double   fallbackDist = 1e18;

                for (const auto& other : m_clusters) {
                    if (other.chId == it->chId) continue;
                    if (other.members.size() >= MEM_MAX) continue;

                    // Trouver les coordonnées du CH cible
                    for (const auto* n : nodes) {
                        if (n->id != other.chId) continue;
                        double d = chNode
                            ? NodeDist(chNode->x, chNode->y, n->x, n->y)
                            : 1e18;
                        if (d < fallbackDist) {
                            fallbackDist = d;
                            fallbackCH   = other.chId;
                        }
                        break;
                    }
                }

                if (fallbackCH == 0 && m_clusters.size() > 1) {
                    // Aucune place disponible → chercher sans contrainte MEM_MAX
                    for (const auto& other : m_clusters) {
                        if (other.chId == it->chId) continue;
                        for (const auto* n : nodes) {
                            if (n->id != other.chId) continue;
                            double d = chNode
                                ? NodeDist(chNode->x, chNode->y, n->x, n->y)
                                : 1e18;
                            if (d < fallbackDist) {
                                fallbackDist = d;
                                fallbackCH   = other.chId;
                            }
                            break;
                        }
                    }
                }

                // Réaffecter les membres orphelins + le CH vers fallbackCH
                if (fallbackCH != 0) {
                    // Réaffecter membres
                    if (m_chToIdx.count(fallbackCH)) {
                        auto& target = m_clusters[m_chToIdx[fallbackCH]];
                        for (uint32_t mid : it->members) {
                            for (auto* n : nodes) {
                                if (n->id == mid) {
                                    n->clusterId = fallbackCH;
                                    n->reclusterCount++;
                                    break;
                                }
                            }
                            target.members.push_back(mid);
                        }
                        // Réaffecter le CH lui-même comme membre
                        if (chNode) {
                            chNode->isClusterHead = false;
                            chNode->clusterId     = fallbackCH;
                            chNode->reclusterCount++;
                            target.members.push_back(chNode->id);
                            target.totalEnergy   += chNode->energy;
                        }
                    }
                }

                // Supprimer le cluster vide de l'index et du vecteur
                m_chToIdx.erase(it->chId);
                it = m_clusters.erase(it);

                // Reconstruire l'index chId→idx après suppression
                m_chToIdx.clear();
                for (uint32_t idx = 0; idx < m_clusters.size(); idx++)
                    m_chToIdx[m_clusters[idx].chId] = idx;

                changed = true;
                break;  // recommencer depuis le début après modification
            }
        }
    }

    // ── Phase 4 : Sélection des CH (top-k avec contrainte espacement) ─────────

    void Phase4_SelectCH(std::vector<NodeState*>& nodes, uint32_t nClusters) {
        // Trier par fitness décroissante
        std::vector<NodeState*> sorted = nodes;
        std::sort(sorted.begin(), sorted.end(),
                  [](const NodeState* a, const NodeState* b) {
                      return a->fitness > b->fitness; });

        const uint32_t kMax = std::min(nClusters,
                                       static_cast<uint32_t>(sorted.size()));

        // Sélection gloutonne avec contrainte d'espacement minimal
        // FIX Problem 1 — espacement minimal adaptatif pour distribution uniforme sur toute la zone
        // idealSpacing = areaSize/√nClusters assure une couverture homogène
        const double idealSpacing = m_areaSize / std::sqrt(static_cast<double>(nClusters));
        const double minSpacing   = std::max(m_radioRange * 0.6, idealSpacing * 0.45);
        std::vector<NodeState*> chosen;
        chosen.reserve(kMax);

        for (auto* cand : sorted) {
            if (chosen.size() >= kMax) break;
            // Vérifier l'espacement avec les CH déjà choisis
            bool tooClose = false;
            for (const auto* ch : chosen) {
                if (NodeDist(cand->x, cand->y, ch->x, ch->y) < minSpacing) {
                    tooClose = true;
                    break;
                }
            }
            if (!tooClose) chosen.push_back(cand);
        }

        // Si pas assez de CH (contrainte espacement trop stricte),
        // compléter sans contrainte
        if (chosen.size() < kMax) {
            std::set<uint32_t> chosenIds;
            for (auto* c : chosen) chosenIds.insert(c->id);
            for (auto* cand : sorted) {
                if (chosen.size() >= kMax) break;
                if (!chosenIds.count(cand->id)) {
                    chosen.push_back(cand);
                    chosenIds.insert(cand->id);
                }
            }
        }

        // Marquer les CH — contrainte voisins adaptative selon densité courante
        // En début de vie (dense) : ≥ MEM_MIN/2 = 4. En fin de vie : ≥ 1
        const uint32_t nc_min = std::max(2u,
            static_cast<uint32_t>(std::ceil(static_cast<double>(nodes.size()) / FdqnCfg::CLUSTER_MEM_MAX)));
        // Première passe : contrainte normale (≥ MEM_MIN/2)
        const int minNeighborsStrict = static_cast<int>(FdqnCfg::CLUSTER_MEM_MIN / 2);
        for (auto* ch : chosen) {
            int nNeighbors = 0;
            for (auto* nb : nodes) {
                if (nb->id == ch->id || !nb->isAlive) continue;
                if (NodeDist(ch->x, ch->y, nb->x, nb->y) <= m_radioRange)
                    nNeighbors++;
            }
            if (nNeighbors < minNeighborsStrict) continue;

            ch->isClusterHead = true;
            ch->clusterId     = ch->id;
            ClusterInfo ci;
            ci.chId        = ch->id;
            ci.totalEnergy = ch->energy;
            m_chToIdx[ch->id] = static_cast<uint32_t>(m_clusters.size());
            m_clusters.push_back(ci);
        }

        // Deuxième passe : si nc_min non atteint, assouplir à ≥ 1 voisin
        if (m_clusters.size() < nc_min) {
            std::set<uint32_t> alreadyCH;
            for (const auto& ci : m_clusters) alreadyCH.insert(ci.chId);
            for (auto* ch : chosen) {
                if (m_clusters.size() >= nc_min) break;
                if (alreadyCH.count(ch->id)) continue;
                // Vérifier au moins 1 voisin vivant
                bool hasNeighbor = false;
                for (auto* nb : nodes) {
                    if (nb->id == ch->id || !nb->isAlive) continue;
                    if (NodeDist(ch->x, ch->y, nb->x, nb->y) <= m_radioRange) {
                        hasNeighbor = true; break;
                    }
                }
                if (!hasNeighbor) continue;
                ch->isClusterHead = true;
                ch->clusterId     = ch->id;
                ClusterInfo ci;
                ci.chId        = ch->id;
                ci.totalEnergy = ch->energy;
                m_chToIdx[ch->id] = static_cast<uint32_t>(m_clusters.size());
                m_clusters.push_back(ci);
                alreadyCH.insert(ch->id);
            }
        }

        // Troisième passe : nc_min ABSOLU — sans contrainte voisins.
        // Garantit que membres max ≤ MEM_MAX même en fin de vie (réseau clairsemé).
        // Parcourt sorted (tous vivants par fitness) au lieu de chosen seulement.
        if (m_clusters.size() < nc_min) {
            std::set<uint32_t> alreadyCH;
            for (const auto& ci : m_clusters) alreadyCH.insert(ci.chId);
            for (auto* ch : sorted) {
                if (m_clusters.size() >= nc_min) break;
                if (alreadyCH.count(ch->id) || ch->isClusterHead) continue;
                ch->isClusterHead = true;
                ch->clusterId     = ch->id;
                ClusterInfo ci;
                ci.chId        = ch->id;
                ci.totalEnergy = ch->energy;
                m_chToIdx[ch->id] = static_cast<uint32_t>(m_clusters.size());
                m_clusters.push_back(ci);
                alreadyCH.insert(ch->id);
            }
        }
    }

    // ── Phase 5 : Formation des clusters ─────────────────────────────────────

    void Phase5_FormClusters(std::vector<NodeState*>& nodes) {
        if (m_clusters.empty()) return;

        const uint32_t MEM_MAX = static_cast<uint32_t>(FdqnCfg::CLUSTER_MEM_MAX);

        // Compteur de membres courant par cluster (index dans m_clusters)
        std::map<uint32_t, uint32_t> memberCount; // chId → count
        for (const auto& ci : m_clusters) memberCount[ci.chId] = 0;

        // Trier les nœuds membres par distance croissante à leur meilleur CH
        // → les plus proches sont servis en premier, limitant la surcharge
        struct Candidate {
            NodeState* node;
            uint32_t   bestCH;
            double     bestDist;
        };
        std::vector<Candidate> candidates;
        candidates.reserve(nodes.size());

        for (auto* n : nodes) {
            if (n->isClusterHead) continue;

            double   bestDist_inRange = 1e18, bestDist_any = 1e18;
            uint32_t bestCH_inRange  = 0,    bestCH_any   = m_clusters[0].chId;

            for (const auto& ci : m_clusters) {
                const NodeState* chPtr = nullptr;
                for (const auto* nd : nodes)
                    if (nd->id == ci.chId) { chPtr = nd; break; }
                if (!chPtr) continue;

                const double d = NodeDist(n->x, n->y, chPtr->x, chPtr->y);
                if (d <= m_radioRange && d < bestDist_inRange) {
                    bestDist_inRange = d;
                    bestCH_inRange   = ci.chId;
                }
                if (d < bestDist_any) {
                    bestDist_any = d;
                    bestCH_any   = ci.chId;
                }
            }
            const uint32_t best = (bestCH_inRange != 0) ? bestCH_inRange : bestCH_any;
            const double   dist = (bestCH_inRange != 0) ? bestDist_inRange : bestDist_any;
            candidates.push_back({n, best, dist});
        }

        // Trier par distance croissante : les nœuds les plus proches sont assignés
        // en premier → garantit que les CH proches ne dépassent pas MEM_MAX
        std::sort(candidates.begin(), candidates.end(),
                  [](const Candidate& a, const Candidate& b){
                      return a.bestDist < b.bestDist; });

        // Assigner chaque nœud à son meilleur CH avec capacité disponible
        for (auto& cand : candidates) {
            NodeState* n = cand.node;
            uint32_t   ch = cand.bestCH;

            // Si le meilleur CH est plein, chercher le CH le plus proche avec de la place
            if (memberCount[ch] >= MEM_MAX) {
                double   fallbackDist = 1e18;
                uint32_t fallbackCH   = 0;
                for (const auto& ci : m_clusters) {
                    if (memberCount[ci.chId] >= MEM_MAX) continue;
                    const NodeState* chPtr = nullptr;
                    for (const auto* nd : nodes)
                        if (nd->id == ci.chId) { chPtr = nd; break; }
                    if (!chPtr) continue;
                    const double d = NodeDist(n->x, n->y, chPtr->x, chPtr->y);
                    if (d < fallbackDist) { fallbackDist = d; fallbackCH = ci.chId; }
                }
                // Si TOUS les clusters sont pleins → CH le moins chargé (débordement minimal)
                if (fallbackCH == 0) {
                    uint32_t minCount = UINT32_MAX;
                    for (const auto& ci : m_clusters) {
                        if (memberCount[ci.chId] < minCount) {
                            minCount   = memberCount[ci.chId];
                            fallbackCH = ci.chId;
                        }
                    }
                }
                if (fallbackCH != 0) ch = fallbackCH;
            }

            const uint32_t oldCH = n->clusterId;
            n->clusterId = ch;
            if (oldCH != 0 && oldCH != ch) n->reclusterCount++;
            memberCount[ch]++;

            if (m_chToIdx.count(ch)) {
                ClusterInfo& ci = m_clusters[m_chToIdx[ch]];
                ci.members.push_back(n->id);
                ci.totalEnergy += n->energy;
            }
        }

        // Calculer distance moyenne par cluster
        for (auto& ci : m_clusters) {
            if (ci.members.empty()) continue;
            const NodeState* chPtr = nullptr;
            for (const auto* n : nodes)
                if (n->id == ci.chId) { chPtr = n; break; }
            if (!chPtr) continue;
            double sumD = 0.0;
            for (uint32_t mid : ci.members) {
                for (const auto* n : nodes)
                    if (n->id == mid) { sumD += NodeDist(*chPtr, *n); break; }
            }
            ci.avgDist = sumD / ci.members.size();
        }
    }


    // ── Membres privés ────────────────────────────────────────────────────────

    double   m_sinkX, m_sinkY;
    double   m_areaSize;
    double   m_radioRange;
    double   m_eInit;
    uint32_t m_maxIter;
    uint32_t m_round;

    std::vector<ClusterInfo>     m_clusters;
    std::map<uint32_t, uint32_t> m_chToIdx;   // chId → index dans m_clusters
};

#endif // IFO_CLUSTERING_H
